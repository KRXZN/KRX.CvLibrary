using CvCommon;
using Leaf.ColorDetector.Models;
using OpenCvSharp;

namespace Leaf.ColorDetector.Detectors;

/// <summary>
/// 颜色检测器接口。
/// </summary>
public interface IColorDetector
{
    /// <summary>
    /// 对指定的 ROI 区域进行颜色检测。
    /// </summary>
    /// <param name="imageMat">完整的 BGR 图像</param>
    /// <param name="roi">感兴趣区域</param>
    /// <param name="expectedColor">期望的颜色名称</param>
    /// <param name="colorDefinitions">颜色定义列表（含 Lab 参考色）</param>
    /// <returns>检测结果</returns>
    ColorDetectResult Detect(
        Mat imageMat,
        CvRect roi,
        string expectedColor,
        IReadOnlyList<ColorDefinition> colorDefinitions);

    /// <summary>
    /// 对已裁剪的 ROI 图像直接进行颜色检测。
    /// </summary>
    /// <param name="roiMat">BGR 格式的 ROI 图像</param>
    /// <param name="expectedColor">期望的颜色名称</param>
    /// <param name="colorDefinitions">颜色定义列表（含 Lab 参考色）</param>
    /// <returns>检测结果</returns>
    ColorDetectResult Detect(
        Mat roiMat,
        string expectedColor,
        IReadOnlyList<ColorDefinition> colorDefinitions);
}
