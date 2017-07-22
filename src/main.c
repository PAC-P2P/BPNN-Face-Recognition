#include "backprop.h"
#include "pgmimage.h"
#include "imagenet.h"
#include "time.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstl/cmap.h>

#define false 0
#define true 1

// 训练集名
#define TRAINNAME "all_train.list"

// 测试集名
#define TESTNAME "all_test.list"

// 网络名
#define NETNAME "BPNN.net"

// 随机产生器的种子
#define SEED 102194

// 学习速率
#define LEARNRATE 0.3

// 冲量
#define IMPULSE 0.3


// 评估表现
int evaluate_performance(BPNN *net, double *err)
{
    bool flag = true; // 样例匹配成功为true

    *err = 0.0;
    double delta;

    // 计算输出层均方误差之和
    for (int j = 1; j <= net->output_n; j++)
    {
        delta = net->target[j] - net->output_units[j];
        *err += (0.5 * delta * delta);
    }


    for (int j = 1; j <= net->output_n; j++) {
        /*** If the target unit is on... ***/
        if (net->target[j] > 0.5) {
            if (net->output_units[j] > 0.5) {
                /*** If the output unit is on, then we correctly recognized me! ***/
            } else /*** otherwise, we didn't think it was me... ***/
            {
                flag = false;
            }
        } else /*** Else, the target unit is off... ***/
        {
            if (net->output_units[j] > 0.5) {
                /*** If the output unit is on, then we mistakenly thought it was me ***/
                flag = false;
            } else {
                /*** else, we correctly realized that it wasn't me ***/
            }
        }
    }

    if (flag)
        return 1;
    else
        return 0;
}

/*** Computes the performance of a net on the images in the imagelist. ***/
/*** Prints out the percentage correct on the image set, and the
     average error between the target and the output units for the set. ***/
int performance_on_imagelist(BPNN *net, IMAGELIST *il, int list_errors, map_t *map_user)
{
    double err, val;
    int i, n, j, correct;

    err = 0.0;
    correct = 0;
    n = il->n;  // n：图片集中图片张数
    if (n > 0) {
        // 遍历图片列表中每张图片
        for (i = 0; i < n; i++) {

            /*** Load the image into the input layer. **/
            load_input_with_image(il->list[i], net);

            /*** Run the net on this input. **/
            bpnn_feedforward(net);

            /*** Set up the target vector for this image. **/
            load_target(il->list[i], net, map_user);

            /*** See if it got it right. ***/
            if (evaluate_performance(net, &val)) {
                //匹配成功，计数器加1
                correct++;
            }
            else if (list_errors)
            {
                printf("%s", NAME(il->list[i]));

                printf("\n");
            }
            err += val; // 列表中所有图片 输出层 均方误差之和
        }

        err = err / (double)n;  // 列表中所有图片 输出层 均方误差之和 的平均数

        if (!list_errors)
            // 输出 匹配准确率 和 误差
            printf("%g%%  %g \n", ((double)correct / (double)n) * 100.0, err);
    } else {
        if (!list_errors)
            printf("0.0 0.0 ");
    }
    return 0;
}

// 评估图片集的匹配情况
void result_on_imagelist(BPNN *net, IMAGELIST *il, int list_errors, map_t *map_user)
{
    double err, val;
    int i, n, j, correct;

    err = 0.0;
    correct = 0;

    n = il->n; // 图片集元素个数

    if (n > 0) {
        for (i = 0; i < n; i++) {
            /*** Load the image into the input layer. **/
            // 装载图片到输入层
            load_input_with_image(il->list[i], net);

            /*** Run the net on this input. **/
            // 在此输入的基础上运行这个网络
            bpnn_feedforward(net);

            /*** Set up the target vector for this image. **/
            // 设置目标向量
            load_target(il->list[i], net, map_user);

            // 输出图片的名称
            printf("Picture name: %s\n", NAME(il->list[i]));

            size_t map_userNum = map_size(map_user), i_flag_num = 0, i_flag_i = 0;

            // map迭代器
            map_iterator_t iterator;

            for(size_t i = 1; i  <= map_userNum; ++i)
            {
                printf("--output_units-->> %f\n", net->output_units[i]);
                if(net->output_units[i] > 0.5)
                {
                    // 统计输出权值大于0.5的输出单元个数和索引
                    i_flag_num ++;
                    i_flag_i = i;
                }
            }

            if(1 == i_flag_num)
            {
                // 遍历map
                for (iterator = map_begin(map_user); !iterator_equal(iterator, map_end(map_user)); iterator = iterator_next(iterator)) {

                    if(i_flag_i == *(int *) pair_second((const pair_t *) iterator_get_pointer(iterator)))
                    {
                        printf("He is --> %s \n", (char *) pair_first((const pair_t *) iterator_get_pointer(iterator)));
                    }
                }
            }
            else
            {
                printf("I do not know who he is...\n");
            }

            /*** See if it got it right. ***/
            if (evaluate_performance(net, &val)) {
                correct++;
                printf("Yes\n");
            } else {
                printf("No\n");
            }

            printf("\n");

            err += val;
        }

        err = err / (double)n;

        // 输出 匹配准确率 和 平均误差
        if (!list_errors)
            printf("Accuracy rate of: %g%%  Average error: %g \n\n",
                   ((double)correct / (double)n) * 100.0, err);
    } else {
        if (!list_errors)
            printf("0.0 0.0 ");
    }
    return;
}

int backprop_face(IMAGELIST *trainlist, IMAGELIST *test1list, int epochs, int savedelta,
                  char *netname, int list_errors, map_t *map_user) {
    IMAGE *iimg;
    BPNN *net;
    int train_n, epoch, i, imgsize;
    double out_err, hid_err, sumerr;

    int userNum = map_size(map_user);

    train_n = trainlist->n;

    /*** Read network in if it exists, otherwise make one from scratch ***/
    if ((net = bpnn_read(netname)) == NULL) {
        if (train_n > 0) {
            printf("Creating new network '%s'\n", netname);
            iimg = trainlist->list[0];
            imgsize = ROWS(iimg) * COLS(iimg);
            // 创建一个 imgsize * userNum * userNum 大小的网络
            net = bpnn_create(imgsize, userNum, userNum);
        } else {
            printf("Need some images to train on!\n");
            return -1;
        }
    }

    if (epochs > 0) {
        /*** 训练进行中（epochs次） ***/
        printf("Training underway (going to %d epochs)\n", epochs);
        /*** 每epochs次保存网络 ***/
        printf("Will save network every %d epochs\n", savedelta);
        fflush(stdout);
    }

    /*** 迭代前输出测试表现 ***/
    /*** Print out performance before any epochs have been completed. ***/
    printf("\n迭代前：\n");
    printf("训练集误差和：0.0\n");
    printf("评估训练集的表现： ");
    performance_on_imagelist(net, trainlist, 0, map_user);
    printf("评估测试集1的表现：");
    performance_on_imagelist(net, test1list, 0, map_user);
    //printf("评估测试集2的表现：");
    //performance_on_imagelist(net, test2list, 0, map_user);
    printf("\n");
    fflush(stdout);
    if (list_errors) {
        printf("\n训练集中的这些图片分类失败:\n");
        performance_on_imagelist(net, trainlist, 1, map_user);
        printf("\n测试集1中的这些图片分类失败:\n");
        performance_on_imagelist(net, test1list, 1, map_user);
        //printf("\n测试集2中的这些图片分类失败:\n");
        //performance_on_imagelist(net, test2list, 1, map_user);
    }

    /************** 开始训练！ ****************************/
    /************** Train it *****************************/
    for (epoch = 1; epoch <= epochs; epoch++) {

        // 输出迭代次数
        printf("Iteration number: %d \n", epoch);
        fflush(stdout);

        sumerr = 0.0;
        for (i = 0; i < train_n; i++) {

            /** Set up input units on net with image i **/
            // 用训练集中图片i来设置输入层单元
            load_input_with_image(trainlist->list[i], net);

            /** Set up target vector for image i **/
            // 为图片i设置目标向量
            load_target(trainlist->list[i], net, map_user);

            /** Run backprop, learning rate 0.3, momentum 0.3 **/
            /** 运行反向传播算法，学习速率0.3，冲量0.3 **/
            bpnn_train(net, LEARNRATE, IMPULSE, &out_err, &hid_err);

            // 训练集中所有图片作为输入，网络的 输出层 和 隐藏层 的误差之和
            sumerr += (out_err + hid_err);
        }
        printf("训练集误差和: %g \n", sumerr);

        // 评估测试集，测试集1，测试集2 的表现
        /*** Evaluate performance on train, test, test2, and print perf ***/
        printf("评估训练集的表现： ");
        performance_on_imagelist(net, trainlist, 0, map_user);
        printf("评估测试集1的表现：");
        performance_on_imagelist(net, test1list, 0, map_user);
        //printf("评估测试集2的表现：");
        //performance_on_imagelist(net, test2list, 0, map_user);
        printf("\n");
        fflush(stdout);

        /*** Save network every 'savedelta' epochs ***/
        if (!(epoch % savedelta)) {
            bpnn_save(net, netname
            );
        }
    }
    printf("\n");
    fflush(stdout);
    /************** 迭代结束 ****************************/

    /************** 预测结果 ****************************/

    // 输出测试集中每张图片的匹配情况
    printf("迭代结束后的匹配情况：\n\n");
    printf("测试集1：\n\n");
    result_on_imagelist(net, test1list, 0, map_user);
    //printf("测试集2：\n\n");
    //result_on_imagelist(net, test2list, 0, map_user);

    /** Save the trained network **/
    if (epochs > 0) {
        bpnn_save(net, netname
        );
    }
    return 0;
}

int main(int argc, char *argv[]) {

    IMAGELIST *trainlist, *testlist;
    int traintimes, seed, savedelta, list_errors;
    clock_t start, finish;

    char netname[30] = NETNAME;
    char trainname[256] = TRAINNAME;
    char testname[256] = TESTNAME;

    // 种子
    seed = SEED; /*** today's date seemed like a good default ***/

    // 初始化
    savedelta = 100;    // 保存网络的周期
    list_errors = 0;

    printf("please input the times of train:\n");
    scanf("%d", &traintimes);

    // 创建存储用户名字的map
    map_t *map_user = create_map(char*,int);
    if (map_user == NULL) {
        printf("Failed to create map\n");
        exit(1);
    }

    // map初始化
    map_init(map_user);

    /*** Create imagelists ***/
    trainlist = imgl_alloc();
    testlist = imgl_alloc();

    //---开始计时---
    start = clock();

    /*** If any train, test1, or test2 sets have been specified, then
     load them in. ***/
    if (trainname[0] != '\0')
    {
    	printf("\n-------load trainlist--------\n");
        imgl_load_images_from_textfile_map(trainlist, trainname, map_user);
    }
    if (testname[0] != '\0')
     {
     	printf("\n-------load testlist--------\n");
     	imgl_load_images_from_textfile_map(testlist, testname, map_user);
     }

    /*** 初始化神经网络包 ***/
    /*** Initialize the neural net package ***/
    bpnn_initialize(seed);

    /*** 显示训练集，测试集1，测试集2中图片数量 ***/
    /*** Show number of images in train, test1, test2 ***/
    printf("%d images in training set\n", trainlist->n);
    printf("%d images in test1 set\n", testlist->n);

    /*** If we've got at least one image to train on, go train the net ***/
    // 假如我们至少有1张图片来训练，那么就开始训练吧！
    backprop_face(trainlist, testlist, traintimes, savedelta, netname, list_errors, map_user);

    //----结束计时-----
    finish = clock();
   	printf( "\nUse %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    exit(0);
}
