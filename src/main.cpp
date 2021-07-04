#include "transformer.h"
#include <stdio.h>
#include <sys/time.h>
double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv) {

    TransFormer transformer;
    transformer.setNumThread(1);
    transformer.initModel();


    char str[1000];
    while (true){
        printf("question:");
        if(scanf("%s",&str) == EOF) break;
        std::string intput = str;
//        printf("111111%s\n",intput.c_str());
        double t1 = get_current_time();
        std::vector<std::string> res = transformer.forward(intput);
        double t2 = get_current_time();
         printf("cost: %f ms\n",t2-t1);
         printf("answer:");
        for (int i = 0; i< res.size();i++)
        {
            printf("%s",res[i].c_str());
        }
        printf("\n");

    }

    return 0;
}