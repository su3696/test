#include "KPlasticWrap.h"
#include <cmath>
#include <vector>

namespace {
	static inline double deg2rad(double deg) {
		constexpr double kPi = 3.1415926535897932384626433832795;
		return deg * kPi / 180.0;
	}
}

void KPlasticWrap::Params::validate() {
	if (detail < 0) detail = 0;
	if (smoothness < 0.0) smoothness = 0.0;
	if (thickness < 0.0) thickness = 0.0;
	if (depth < 0.1) depth = 0.1;
	if (postBlur < 0.0) postBlur = 0.0;
}

void KPlasticWrap::setParams(int detail, double smoothness, double thickness, double angleDeg, double depth, double postBlur) {
	m_params.detail = detail;
	m_params.smoothness = smoothness;
	m_params.thickness = thickness;
	m_params.angle = angleDeg;
	m_params.depth = depth;
	m_params.postBlur = postBlur;
	m_params.validate();
}

int KPlasticWrap::computeKernelSizeFromSigma(double sigma) {
	if (sigma <= 0.0) return 0; // use no blur
	int k = static_cast<int>(std::ceil(sigma * 3.0) * 2 + 1); // ~3*sigma each side
	if (k % 2 == 0) ++k;
	if (k < 3) k = 3;
	return k;
}

cv::Mat KPlasticWrap::createGrayscale(const cv::Mat& src) {
	CV_Assert(!src.empty());
	cv::Mat bgr;
	if (src.type() == CV_8UC3) {
		bgr = src;
	} else if (src.type() == CV_8UC4) {
		cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
	} else if (src.type() == CV_8UC1) {
		return src.clone();
	} else {
		CV_Error(cv::Error::StsUnsupportedFormat, "KPlasticWrap: only CV_8UC1/3/4 supported");
	}
	cv::Mat gray;
	cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
	return gray;
}

void KPlasticWrap::normalizeMinMax(cv::Mat& gray) {
	CV_Assert(gray.type() == CV_8UC1);
	double minV = 0.0, maxV = 0.0;
	cv::minMaxLoc(gray, &minV, &maxV);
	if (maxV <= minV + 1e-6) return;
	cv::Mat f;
	gray.convertTo(f, CV_32F);
	f = (f - static_cast<float>(minV)) * (255.0f / static_cast<float>(maxV - minV));
	f.convertTo(gray, CV_8U);
}

void KPlasticWrap::applyGaussianSmooth(cv::Mat& gray, double radius) {
	if (radius <= 0.0) return;
	// Map radius to sigma. PhotoDemon uses a direct pixel radius in its custom blur.
	// We approximate by sigma ~= radius / 2.
	double sigma = std::max(0.1, radius * 0.5);
	int k = computeKernelSizeFromSigma(sigma);
	if (k > 0)
		cv::GaussianBlur(gray, gray, cv::Size(k, k), sigma, sigma, cv::BORDER_REPLICATE);
}

void KPlasticWrap::buildTriangularLUT(std::array<uint8_t, 256>& lut, int detail) {
	// VB: wrapDetail = userDetail + 2, then create points alternating 0,255 across [0..255]
	int steps = std::max(0, detail) + 2;
	if (steps < 2) steps = 2;
	// Build piecewise linear curve with 'steps' segments alternating between 0 and 255, starting at 0.
	// We'll fill LUT by mapping x in [0..255] to the corresponding segment.
	for (int x = 0; x < 256; ++x) {
		double t = x / 255.0; // [0,1]
		double pos = t * steps; // [0,steps]
		int i = static_cast<int>(std::floor(pos));
		if (i < 0) i = 0;
		if (i >= steps) i = steps - 1;
		double frac = pos - i; // [0,1)
		// Segment i goes from value at i to value at i+1
		int v0 = (i % 2 == 0) ? 0 : 255;
		int v1 = ((i + 1) % 2 == 0) ? 0 : 255;
		int v = static_cast<int>(std::round(v0 * (1.0 - frac) + v1 * frac));
		lut[x] = static_cast<uint8_t>(v);
	}
}

cv::Mat KPlasticWrap::applyDetailCurve(const cv::Mat& gray, const std::array<uint8_t, 256>& lut) {
	cv::Mat mapped(gray.size(), CV_8UC1);
	// Use LUT for efficiency
	cv::Mat lutMat(1, 256, CV_8UC1);
	for (int i = 0; i < 256; ++i) lutMat.at<uint8_t>(0, i) = lut[i];
	cv::LUT(gray, lutMat, mapped);
	return mapped;
}

cv::Mat KPlasticWrap::embossWithLighting(const cv::Mat& gray, double angleDeg, double distance, double depth) {
	// Compute bilinear-sampled shifted image at direction (angle) with distance.
	const int width = gray.cols;
	const int height = gray.rows;

	double rad = deg2rad(angleDeg);
	double dx = std::cos(rad) * distance;
	double dy = std::sin(rad) * distance;

	cv::Mat shifted;
	{
		// Build affine transform for pure translation by (dx, dy)
		cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
		cv::warpAffine(gray, shifted, M, gray.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
	}

	// Emboss value = (src - shifted) * depth, clamped [0,255]
	cv::Mat src32f, shifted32f, diff32f, out8u;
	gray.convertTo(src32f, CV_32F);
	shifted.convertTo(shifted32f, CV_32F);
	diff32f = (src32f - shifted32f) * static_cast<float>(depth);
	cv::threshold(diff32f, diff32f, 255.0, 255.0, cv::THRESH_TRUNC);
	cv::threshold(diff32f, diff32f, 0.0, 0.0, cv::THRESH_TOZERO);
	diff32f.convertTo(out8u, CV_8U);
	return out8u;
}

void KPlasticWrap::gentlePostBlur(cv::Mat& gray, double radius) {
	if (radius <= 0.0) return;
	// Use separable Gaussian with sigma ~= radius.
	double sigma = std::max(0.1, radius);
	int k = computeKernelSizeFromSigma(sigma);
	if (k > 0)
		cv::GaussianBlur(gray, gray, cv::Size(k, k), sigma, sigma, cv::BORDER_REPLICATE);
}

cv::Mat KPlasticWrap::screenBlend(const cv::Mat& base, const cv::Mat& highlightMapGray) {
	// Screen blend: out = 1 - (1 - A) * (1 - B)
	// Assume base is BGR/BGRA and highlight is GRAY. We'll apply the gray as a 3-channel map.
	CV_Assert(base.type() == CV_8UC3 || base.type() == CV_8UC4);
	CV_Assert(highlightMapGray.type() == CV_8UC1);
	cv::Mat highlight3;
	if (base.channels() == 3) {
		cv::cvtColor(highlightMapGray, highlight3, cv::COLOR_GRAY2BGR);
		cv::Mat base32f, high32f, out32f;
		base.convertTo(base32f, CV_32F, 1.0 / 255.0);
		highlight3.convertTo(high32f, CV_32F, 1.0 / 255.0);
		out32f = 1.0f - (1.0f - base32f).mul(1.0f - high32f);
		cv::Mat out8u;
		out32f.convertTo(out8u, CV_8U, 255.0);
		return out8u;
	} else {
		// BGRA: apply screen on RGB, preserve alpha as-is
		std::vector<cv::Mat> ch;
		cv::split(base, ch); // B,G,R,A
		cv::Mat rgb;
		cv::merge(std::vector<cv::Mat>{ch[0], ch[1], ch[2]}, rgb);

		cv::Mat high3; 
		cv::cvtColor(highlightMapGray, high3, cv::COLOR_GRAY2BGR);

		cv::Mat rgb32f, high32f, out32f;
		rgb.convertTo(rgb32f, CV_32F, 1.0 / 255.0);
		high3.convertTo(high32f, CV_32F, 1.0 / 255.0);
		out32f = 1.0f - (1.0f - rgb32f).mul(1.0f - high32f);
		cv::Mat outRgb8u;
		out32f.convertTo(outRgb8u, CV_8U, 255.0);

		std::vector<cv::Mat> rgbOutCh;
		cv::split(outRgb8u, rgbOutCh);
		cv::Mat outBGRA;
		cv::merge(std::vector<cv::Mat>{rgbOutCh[0], rgbOutCh[1], rgbOutCh[2], ch[3]}, outBGRA);
		return outBGRA;
	}
}

cv::Mat KPlasticWrap::apply(const cv::Mat& input) const {
	CV_Assert(!input.empty());
	CV_Assert(input.type() == CV_8UC3 || input.type() == CV_8UC4);

	Params p = m_params;
	p.validate();

	// 1) 灰度 + 可选平滑
	cv::Mat gray = createGrayscale(input);
	applyGaussianSmooth(gray, p.smoothness);
	// 可选：重归一化（与 VB 注释一致，默认不开启；如需使用，解除下一行注释）
	// normalizeMinMax(gray);

	// 2) 构造细节 LUT 并应用到灰度图
	std::array<uint8_t, 256> lut{};
	buildTriangularLUT(lut, p.detail);
	cv::Mat grayDetail = applyDetailCurve(gray, lut);

	// 3) 浮雕（按角度和厚度进行位移采样），乘以光照强度 depth
	cv::Mat embossed = embossWithLighting(grayDetail, p.angle, p.thickness, p.depth);

	// 4) 柔和后处理以减轻像素化
	gentlePostBlur(embossed, p.postBlur);

	// 5) 使用 Screen 模式与原图融合（更现代、较轻）
	cv::Mat result = screenBlend(input, embossed);
	return result;
}