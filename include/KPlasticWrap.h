#pragma once

#include <opencv2/opencv.hpp>
#include <array>

class KPlasticWrap {
public:
	// 塑料包装（Plastic Wrap）滤镜参数
	struct Params {
		// 细节级别（对应 PhotoDemon 的 detail 滑块）；内部会自动 +2
		int detail = 4;               
		// 平滑半径（像素，越大越平滑）
		double smoothness = 20.0;     
		// 厚度/位移距离（像素，决定浮雕位移采样距离）
		double thickness = 2.0;       
		// 光照角度（度）
		double angle = 0.0;           
		// 光照强度（浮雕强度，建议 >= 0.1）
		double depth = 5.0;           
		// 末端柔化半径（像素，用于弱化像素化感）
		double postBlur = 3.0;        

		void validate();
	};

	// 设置参数
	void setParams(int detail, double smoothness, double thickness, double angleDeg, double depth, double postBlur = 3.0);

	// 获取参数
	Params getParams() const { return m_params; }

	// 应用滤镜。输入支持 CV_8UC3（BGR）或 CV_8UC4（BGRA）。输出与输入通道数一致。
	cv::Mat apply(const cv::Mat& input) const;

private:
	Params m_params;

	static void buildTriangularLUT(std::array<uint8_t, 256>& lut, int detail);
	static inline uint8_t clampToByte(int v) { return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v)); }
	static int computeKernelSizeFromSigma(double sigma);

	static cv::Mat createGrayscale(const cv::Mat& srcBGRorBGRA);
	static void normalizeMinMax(cv::Mat& gray);
	static void applyGaussianSmooth(cv::Mat& gray, double radius);
	static cv::Mat applyDetailCurve(const cv::Mat& gray, const std::array<uint8_t, 256>& lut);
	static cv::Mat embossWithLighting(const cv::Mat& gray, double angleDeg, double distance, double depth);
	static void gentlePostBlur(cv::Mat& gray, double radius);
	static cv::Mat screenBlend(const cv::Mat& base, const cv::Mat& highlightMapGray);
};