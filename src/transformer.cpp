#include "transformer.h"
#include <fstream>
#include <numeric>
#include "layer.h"

const char *const DICT_PATH = "../models/dict/jieba.dict.utf8";
const char *const HMM_PATH = "../models/dict/hmm_model.utf8";
const char *const USER_DICT_PATH = "../models/dict/user.dict.utf8";
const char *const IDF_PATH = "../models/dict/idf.utf8";
const char *const STOP_WORD_PATH = "../models/dict/stop_words.utf8";

cppjieba::Jieba jieba(DICT_PATH,
                      HMM_PATH,
                      USER_DICT_PATH,
                      IDF_PATH,
                      STOP_WORD_PATH);


class TransformerMask : public ncnn::Layer {
public:
    TransformerMask() {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w;
        int outh = h;
        int outc = channels;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++) {
            const float *ptr = bottom_blob.channel(p);
            float *outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    if (j > i) {
                        *outptr = -10000000.f;
                    } else {
                        *outptr = *ptr;
                    }
                    outptr += 1;
                    ptr += 1;
                }

//                ptr += w;
            }
        }


        return 0;
    }
};


class TransformerGemm : public ncnn::Layer {
public:
    TransformerGemm() {
        one_blob_only = false;
    }

    virtual int forward(const std::vector<ncnn::Mat> &bottom_blobs, std::vector<ncnn::Mat> &top_blobs,
                        const ncnn::Option &opt) const {
        const ncnn::Mat &A0 = bottom_blobs[0];
        const ncnn::Mat &B0 = bottom_blobs[1];
        size_t elemsize = A0.elemsize;
        ncnn::Mat A = A0;


        ncnn::Mat B;

        B.create(B0.h, B0.w, B0.c, elemsize, opt.workspace_allocator);
        // transpose B to col-major

        for (int p = 0; p < B0.c; p++) {
            const float *ptr_pb = B0.channel(p);
            float *ptr = B.channel(p);
            int h = B.h;
            int w = B.w;
            for (int i = 0; i < h; i++) {
//                float *ptr = B.row(i);i
                for (int j = 0; j < w; j++) {
                    ptr[i * w + j] = ptr_pb[j * h + i];
                }
            }
        }


        int M = A.h;
        int K = A.w; // assert A.w == B.w
        int N = B.h;
        int C = A.c;

        ncnn::Mat &top_blob = top_blobs[0];
        top_blob.create(N, M, C, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        for (int p = 0; p < C; p++) {

            float *outptr = top_blob.channel(p);
            const float *ptr_pa = A.channel(p);
            const float *ptr_pb = B.channel(p);

            for (int i = 0; i < M; i++) {
//                const float *ptrA = tempA.row(i);

                for (int j = 0; j < N; j++) {
//                    const float *ptrB = tempB.row(j);

                    float sum = 0.f;

                    for (int k = 0; k < K; k++) {
//                        sum += ptrA[k] * ptrB[k];
                        sum += ptr_pa[i * K + k] * ptr_pb[j * K + k];
                    }

                    *outptr++ = sum;
                }
            }
        }


        return 0;
    }
};


DEFINE_LAYER_CREATOR(TransformerMask)

DEFINE_LAYER_CREATOR(TransformerGemm)


void TransFormer::setGpuIndex(int gpuIndex) {
#ifdef __VULKAN__
    if (gpuIndex >= 0) {
        net.opt.use_vulkan_compute = true;
        net.set_vulkan_device(gpuIndex);
        printf("CrnnNet try to use Gpu%d\n", gpuIndex);
    } else {
        net.opt.use_vulkan_compute = false;
        printf("CrnnNet use Cpu\n");
    }
#endif
}

TransFormer::~TransFormer() {
    encoder_net.clear();
    decoder_net.clear();
}

void TransFormer::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

bool TransFormer::initModel() {

    encoder_net.register_custom_layer("TransformerGemm", TransformerGemm_layer_creator);
    decoder_net.register_custom_layer("TransformerGemm", TransformerGemm_layer_creator);
    decoder_net.register_custom_layer("TransformerMask", TransformerMask_layer_creator);

    encoder_net.load_param("../models/encoder.param");
    encoder_net.load_model("../models/encoder.bin");

    decoder_net.load_param("../models/decoder.param");
    decoder_net.load_model("../models/decoder.bin");

    std::ifstream src_vocab_in("../models/src_vocab.txt");
    std::string line;
    if (src_vocab_in) {
        int count = 0;
        while (getline(src_vocab_in, line)) {// line中不包括每行的换行符
            encoder_map[line] = count;
            count++;
        }
    }

    std::ifstream tgt_vocab_in("../models/tgt_vocab.txt");
    if (tgt_vocab_in) {
        int count = 0;
        while (getline(tgt_vocab_in, line)) {// line中不包括每行的换行符
            decoder_map[count] = line;
            count++;
        }
    }
    encoder_map_size = encoder_map.size();
    decoder_map_size = decoder_map.size();

    enc_tok_embedding_weight.create(encoder_hidden_size, encoder_map_size);
    dec_tok_embedding_weight.create(decoder_hidden_size, decoder_map_size);

    std::ifstream in_enc_tok("../models/enc_tok_embedding_weight.bin", std::ios::in | std::ios::binary);
    in_enc_tok.read((char *) enc_tok_embedding_weight, sizeof(float) * encoder_hidden_size * encoder_map_size);

    std::ifstream in_dec_tok("../models/dec_tok_embedding_weight.bin", std::ios::in | std::ios::binary);
    in_dec_tok.read((char *) dec_tok_embedding_weight, sizeof(float) * decoder_hidden_size * decoder_map_size);


    enc_pos_embedding_weight.create(encoder_hidden_size, max_len);
    dec_pos_embedding_weight.create(decoder_hidden_size, max_len);

    std::ifstream in_enc_pos("../models/enc_pos_embedding_weight.bin", std::ios::in | std::ios::binary);
    in_enc_pos.read((char *) enc_pos_embedding_weight, sizeof(float) * encoder_hidden_size * max_len);

    std::ifstream in_dec_pos("../models/dec_pos_embedding_weight.bin", std::ios::in | std::ios::binary);
    in_dec_pos.read((char *) dec_pos_embedding_weight, sizeof(float) * decoder_hidden_size * max_len);

    return true;
}

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

std::vector<std::string> TransFormer::forward(std::string &input_str) {
    std::vector<std::string> inputs;

    jieba.Cut(input_str, inputs, true);

    std::vector<std::string> sos = {"<sos>"};
    inputs.insert(inputs.begin(), sos.begin(), sos.end());
    inputs.emplace_back("<eos>");

    ncnn::Mat enc_tok_embedding, enc_pos_embedding;
    int word_size = inputs.size();

    int input_len = 0;
    if (word_size % 4 == 0) input_len = word_size;
    else input_len = (word_size / 4 + 1) * 4;


    enc_tok_embedding.create(encoder_hidden_size, input_len);
    enc_pos_embedding.create(encoder_hidden_size, input_len);
    enc_tok_embedding.fill(0.f);
    enc_pos_embedding.fill(0.f);


    memcpy(enc_pos_embedding, enc_pos_embedding_weight, word_size * encoder_hidden_size * sizeof(float));

    for (int i = 0; i < word_size; i++) {
        float *pt_t = enc_tok_embedding.row(i);
        const float *pt_tok_en = enc_tok_embedding_weight.row(encoder_map[inputs[i]]);
        memcpy(pt_t, pt_tok_en, encoder_hidden_size * sizeof(float));
    }
//    printf("111\n");
    ncnn::Extractor ex = encoder_net.create_extractor();
    ex.input("0", enc_tok_embedding);
    ex.input("1", enc_pos_embedding);

    ncnn::Mat encode_outputs;

    ex.extract("out1", encode_outputs);

    std::vector<int> decode_input_indexs = {sos_index, sos_index};
    std::vector<std::string> output;
    for (int i = 1; i < max_len; i++) {

        ncnn::Mat decode_tok_input, decode_pos_input;
        int size = 0;
        if ((i + 1) % 4 == 0) size = i + 1;
        else size = ((i + 1) / 4 + 1) * 4;

        decode_tok_input.create(decoder_hidden_size, size);
        decode_pos_input.create(decoder_hidden_size, size);
        decode_tok_input.fill(0.f);
        decode_pos_input.fill(0.f);
        for (int j = 0; j < i + 1; j++) {
            float *pt_d = decode_tok_input.row(j);
            const float *pt_tok_dc = dec_tok_embedding_weight.row(decode_input_indexs[j]);
            memcpy(pt_d, pt_tok_dc, decoder_hidden_size * sizeof(float));
        }

        memcpy(decode_pos_input, dec_pos_embedding_weight, (i + 1) * decoder_hidden_size * sizeof(float));

        ncnn::Extractor ex2 = decoder_net.create_extractor();
        ex2.input("0", decode_tok_input);
        ex2.input("1", decode_pos_input);
        ex2.input("input.8", encode_outputs);


        ncnn::Mat decode_output;
        ex2.extract("out1", decode_output);

//        printf("h:%d w:%d c:%d  size:%d\n", decode_output.h, decode_output.w, decode_output.c,decoder_map_size);
        const float *ptr = decode_output.row(i);
//        printf("h:%d w:%d c:%d \n",decode_hidden.h,decode_hidden.w,decode_hidden.c);
        int max_index = 0;
        float max_value = -100000.f;
        for (int j = 0; j < decoder_map_size; j++) {
            if (ptr[j] > max_value) {
                max_index = j;
                max_value = ptr[j];
            }
        }

        decode_input_indexs.push_back(max_index);


//        printf("%s \n",decoder_map[max_index] .c_str());
        if (decoder_map[max_index] == "<eos>") break;
        if (decoder_map[max_index] == "<unk>") continue;
        output.push_back(decoder_map[max_index]);
    }
    return output;
}
