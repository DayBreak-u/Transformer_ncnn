#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__

#include "net.h"
#include <map>
#include <vector>
#include "cppjieba/Jieba.hpp"


class TransFormer {
public:

    ~TransFormer();

    void setNumThread(int numOfThread);

    void setGpuIndex(int gpuIndex);

    bool initModel();

    std::vector<std::string> forward(std::string &input_str);

private:


    int numThread;
    ncnn::Net encoder_net;
    ncnn::Net decoder_net;

    ncnn::Mat enc_tok_embedding_weight;
    ncnn::Mat enc_pos_embedding_weight;

    ncnn::Mat dec_tok_embedding_weight;
    ncnn::Mat dec_pos_embedding_weight;

    int encoder_map_size;
    int decoder_map_size;

    int encoder_hidden_size = 256;
    int decoder_hidden_size = 256;

    int sos_index = 2;
    int eos_index = 3;


    int max_len = 100;
//    std::ifstream in_encode("../models/encode_embedding_weight.bin",std::ios::in | std::ios::binary);;
//    std::ifstream in_decode("../models/decode_embedding_weight.bin",std::ios::in | std::ios::binary);;
    std::map<std::string,int> encoder_map;
    std::map<int,std::string> decoder_map;

};


#endif //__TRANSFORMER_H__
