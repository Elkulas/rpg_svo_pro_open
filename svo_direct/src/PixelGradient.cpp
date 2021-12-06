//
// updated by jiang 21/11/09
//
#include "PixelGradient.h"
#include "Utils.h"

PixelGradient::PixelGradient() {
  pyrLevelsUsed = 4;
}


PixelGradient::~PixelGradient() {
}

void PixelGradient::computeGradents(const Mat img)
{
  // 初始的长宽
  int w = img.cols;
  int h = img.rows;

  // 0层情况
  // wG和hG是int的数组
  wG[0] = w;
  hG[0] = h;
  // 构建金字塔
  // 四层,在构造函数中定义
  // 将金字塔各个层的长宽存下来
  for (int level = 1; level < pyrLevelsUsed; ++level) {
    wG[level] = wG[level - 1] / 2;
    hG[level] = hG[level - 1] / 2;
  }

  // // 所有的像素数量
  // int tolPixel = img.cols * img.rows;

  // // 指向uchar的指针
  // unsigned char *color = new unsigned char[tolPixel];

  // // 将原来在img中的数据拷贝到color分配的内存中
  // memcpy(color, img.data, img.rows*img.cols);


  // 为每一层图像梯度分配存储空间
  // 用dIp来存储各个层的梯度情况,sq的存储梯度的平方
  // NOTICE:其中dIp是存储了所有金字塔各层的像素,梯度的信息
  // 其中[0]表示就是原始像素的信息
  for (int i = 0; i < pyrLevelsUsed; i++) {
    dIp[i] = new Eigen::Vector3f[wG[i] * hG[i]];
    absSquaredGrad[i] = new float[wG[i] * hG[i]];
  }

  // dI指针指向了第0层的第一个元素
  // 原来他们指向同一个地方,图像导数
  // dI也就表示了第0层的情况
  dI = dIp[0]; 

  // 将img中的像素信息存到dI中
  // for (int i = 0; i < w * h; i++){
  //   dI[i][0] = (float)color[i];
  // }


  if(img.type() == 0) {
    for (int j = 0; j < img.rows; ++j) {
      for (int i = 0; i < img.cols; ++i) {
        dI[i + j * img.cols][0] = (img.at<uchar>(j, i));
      }
    }
  }
  else if(img.type() == 2) {
    std::cout << "16bit!" << std::endl;
    for (int j = 0; j < img.rows; ++j) {
      for (int i = 0; i < img.cols; ++i) {
        dI[i + j * img.cols][0] = (img.at<ushort>(j, i));
      }
    }
  }
  else {
    std::cout << "Not Proper type!" << std::endl;
  }



  // 遍历金字塔
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    // 该层图像大小
    int wl = wG[lvl], hl = hG[lvl]; 
    // 指向当前层的像素
    Eigen::Vector3f *dI_l = dIp[lvl];

    // 对于平方梯度的一个指针
    float *dabs_l = absSquaredGrad[lvl];

    // 如果不是第一层
    if (lvl > 0) {
      // 上一层
      int lvlm1 = lvl - 1;
      // 上一层width
      int wlm1 = wG[lvlm1]; // 列数
      Eigen::Vector3f *dI_lm = dIp[lvlm1];


      // 像素4合1, 生成金字塔
      // 在此处 上采样 ,生成上层金字塔所有的像素值,存放在[0]
      for (int y = 0; y < hl; y++)
        for (int x = 0; x < wl; x++) {
          dI_l[x + y * wl][0] = 0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] +
              dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
              dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
              dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
        }
    }

    // 平均gradients
    float mean_grident = 0.0f;
    // 第二行开始,倒数第二行结束
    for (int idx = wl; idx < wl * (hl - 1); idx++)
    {
      float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
      float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

      if (!std::isfinite(dx)) dx = 0;
      if (!std::isfinite(dy)) dy = 0;

      // 0:原始像素值 1:x方向梯度值 2:y方向梯度值
      dI_l[idx][1] = dx;
      dI_l[idx][2] = dy;

      dabs_l[idx] = dx * dx + dy * dy; // 梯度平方
      mean_grident += sqrtf(dabs_l[idx]);
    }
    // std::cout << "mean grident of level" << lvl << "=" << mean_grident / (wl * hl) << std::endl;

  } // 遍历完毕各个层

  // int wl = wG[level], hl = hG[level]; // 该层图像大小
  // Eigen::Vector3f *dI_l = dIp[level];
  // float *dabs_l = absSquaredGrad[level];

}

void PixelGradient::computeGradentsWithMask(const Mat img, const Mat mask)
{

  // 判断mask情况
  if(mask.empty())
      std::cout << "mask is empty " << std::endl;
  if (mask.type() != CV_8UC1)
      std::cout << "mask type wrong " << std::endl;
  if (mask.size() != img.size())
      std::cout << "wrong size " << std::endl;
  

  // 初始的长宽
  int w = img.cols;
  int h = img.rows;

  // 0层情况
  // wG和hG是int的数组
  wG[0] = w;
  hG[0] = h;
  // 构建金字塔
  // 四层,在构造函数中定义
  // 将金字塔各个层的长宽存下来
  for (int level = 1; level < pyrLevelsUsed; ++level) {
    wG[level] = wG[level - 1] / 2;
    hG[level] = hG[level - 1] / 2;
  }

  // // 所有的像素数量
  // int tolPixel = img.cols * img.rows;

  // // 指向uchar的指针
  // unsigned char *color = new unsigned char[tolPixel];
  // 存放mask data指针
  unsigned char *dmask[PYR_LEVELS];

  // // 将原来在img中的数据拷贝到color分配的内存中
  // memcpy(color, img.data, img.rows*img.cols);
  // memcpy(dmask, mask.data, mask.rows*mask.cols);


  // 为每一层图像梯度分配存储空间
  // 用dIp来存储各个层的梯度情况,sq的存储梯度的平方
  // NOTICE:其中dIp是存储了所有金字塔各层的像素,梯度的信息
  // 其中[0]表示就是原始像素的信息
  for (int i = 0; i < pyrLevelsUsed; i++) {
    dIp[i] = new Eigen::Vector3f[wG[i] * hG[i]];
    absSquaredGrad[i] = new float[wG[i] * hG[i]];
    dmask[i] = new unsigned char[wG[i] * hG[i]];
  }

  // 拷贝地一层mask
  memcpy(dmask[0], mask.data, mask.rows*mask.cols);


  // dI指针指向了第0层的第一个元素
  // 原来他们指向同一个地方,图像导数
  // dI也就表示了第0层的情况
  dI = dIp[0]; 

  // // 将img中的像素信息存到dI中
  // for (int i = 0; i < w * h; i++)
  //   dI[i][0] = (float)color[i];

  if(img.type() == 0) {
    for (int j = 0; j < img.rows; ++j) {
      for (int i = 0; i < img.cols; ++i) {
        dI[i + j * img.cols][0] = (img.at<uchar>(j, i));
      }
    }
  }
  else if(img.type() == 2) {
    for (int j = 0; j < img.rows; ++j) {
      for (int i = 0; i < img.cols; ++i) {
        dI[i + j * img.cols][0] = (img.at<ushort>(j, i));
      }
    }
  }
  else {
    std::cout << "Not Proper type!" << std::endl;
  }


  // 遍历金字塔
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    // 该层图像大小
    int wl = wG[lvl], hl = hG[lvl]; 
    // 指向当前层的像素
    Eigen::Vector3f *dI_l = dIp[lvl];

    unsigned char *dmask_l = dmask[lvl];

    // 对于平方梯度的一个指针
    float *dabs_l = absSquaredGrad[lvl];

    // 如果不是第一层
    if (lvl > 0) {
      // 上一层
      int lvlm1 = lvl - 1;
      // 上一层width
      int wlm1 = wG[lvlm1]; // 列数
      Eigen::Vector3f *dI_lm = dIp[lvlm1];
      unsigned char *dmask_lm = dmask[lvlm1];


      // 像素4合1, 生成金字塔
      // 在此处 上采样 ,生成上层金字塔所有的像素值,存放在[0]
      for (int y = 0; y < hl; y++)
        for (int x = 0; x < wl; x++) {
          dI_l[x + y * wl][0] = 0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] +
              dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
              dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
              dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
          
          dmask_l[x + y * wl] = 0.25f * (dmask_lm[2 * x + 2 * y * wlm1] +
              dmask_lm[2 * x + 1 + 2 * y * wlm1] +
              dmask_lm[2 * x + 2 * y * wlm1 + wlm1] +
              dmask_lm[2 * x + 1 + 2 * y * wlm1 + wlm1]);
        }
    }

    // 平均gradients
    float mean_grident = 0.0f;
    // 第二行开始,倒数第二行结束
    for (int idx = wl; idx < wl * (hl - 1); idx++)
    {
      float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
      float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

      if (!std::isfinite(dx)) dx = 0;
      if (!std::isfinite(dy)) dy = 0;

      // 0:原始像素值 1:x方向梯度值 2:y方向梯度值
      dI_l[idx][1] = dx;
      dI_l[idx][2] = dy;

      if(dmask_l[idx] == 255)
        dabs_l[idx] = dx * dx + dy * dy; // 梯度平方
      else
        dabs_l[idx] = 0; // 梯度平方

      mean_grident += sqrtf(dabs_l[idx]);
    }
    // std::cout << "With Mask mean grident of level" << lvl << "=" << mean_grident / (wl * hl) << std::endl;

  } // 遍历完毕各个层

  // int wl = wG[level], hl = hG[level]; // 该层图像大小
  // Eigen::Vector3f *dI_l = dIp[level];
  // float *dabs_l = absSquaredGrad[level];

}