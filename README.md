CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Qirui (Chiray) Fu
  * [personal website](https://qiruifu.github.io/)
  * I also write a [blog](https://qiruifu.github.io/prefix-sum-blog/) for this project
* Tested on my own laptop: Windows 11, i5-13500HX @ 2.5GHz 16GB, GTX 4060 8GB

### README

This is a project working on exclusive scan and stream compaction algorithms implemented by CUDA. We also compare the efficience of our method with CPU implementation and thrust library.

#### My Implementation
##### Part 1~4
For these basic parts, CPU part and thrust part are not very hard. I spent more time on naive and work-efficient part. At first I wrote all loops outside the kernel, which means I invoked the kernel many times. The advantage of it is every time I can only raise the threads that work in this layer. But I found that invoking kernel takes a lot of time, so I changed it and put all loops inside kernel. 

At this step, the efficience is really bad. The naive version can be slightly faster than the CPU implementation, while the work-efficient version takes hundreds of times longer than the CPU.

##### Part 5/ 6 Extra Credit
To accelerate the program, I implemented the work-efficient with shared memory. It's a little annoying because we have to scan the recorded summations of other blocks. To achieve that, I have to implement a recursive process so it's a little hard to debug at first. Now, the efficience of my code is closed to thrust implementation, I am really satisfied with that.

It's reasonable that work-efficient method was so bad at first. We had to visit so much non-contiguous global memory, which is extremelly time consuming. And, of course, some threads were lazy, they were raised and did nothing.

#### Performance Analysis

##### Block Sizes
For naive implementation, best block size is 256 or 512. If you choose 1024 it's much slower.

For work-efficient implementation with shared memory, best block size is 128 or 256. It's smaller than naive method. 

##### Comparison of Different Implementations

The length is power of 2:

<img src="/img/p-4.png" style="width: 100%">

Only compare work-efficient and thrust:

<img src="/img/p-2.png" style="width: 100%">

The length is not power of 2:

<img src="/img/np-4.png" style="width: 100%">

Only compare work-efficient and thrust:

<img src="/img/np-2.png" style="width: 100%">

From the comparisons, we can say:
* My implementation of work-efficient with shared memory is as good as thrust when length of array is not very high. When the array gets longer, the time of my method increases more.
* We all know that when the length of an array is power of 2, we would obtain a really bad cache properties ([You can find so many articles about that like this](https://blog.tteles.dev/posts/why-powers-of-two-are-bad-for-cache/)). From this aspect, we can analyse the bottleneck is memory access or computation. For example, the time spent by CPU method are always similar between two lengths, which means CPU method is limited by computational ability. For naive method, before $2^{22}$ it's limited by memory visit. When array gets larger, its bottleneck is computing. For our optimized method and thrust, they are limited by memory visit, since the differences between two lengths are significant.

#### My Output
This output is generated in length of $2^{25}$.
```
****************
** SCAN TESTS **
****************
    [  25  24   7  28  20  13  13   3  25  12  40  29  43 ...   9   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 63.1529ms    (std::chrono Measured)
    [   0  25  49  56  84 104 117 130 133 158 170 210 239 ... 821828477 821828486 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 68.7827ms    (std::chrono Measured)
    [   0  25  49  56  84 104 117 130 133 158 170 210 239 ... 821750162 821750177 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 37.6891ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 35.5439ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 4.15898ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 3.68474ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 2.2831ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.19594ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   2   3   0   3   0   2   1   2   2   2   1   1 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 67.8244ms    (std::chrono Measured)
    [   3   2   3   3   2   1   2   2   2   1   1   2   2 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 68.7491ms    (std::chrono Measured)
    [   3   2   3   3   2   1   2   2   2   1   1   2   2 ...   1   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 194.502ms    (std::chrono Measured)
    [   3   2   3   3   2   1   2   2   2   1   1   2   2 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 6.80982ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 6.33053ms    (CUDA Measured)
    passed
```
