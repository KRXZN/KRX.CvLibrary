using CvCommon;
using CvLibrary.OpenCV;
using Leaf.ColorDetector.ColorScience;
using Leaf.ColorDetector.Models;
using Leaf.ColorDetector.Preprocessing;
using OpenCvSharp;

namespace Leaf.ColorDetector.Calibration;

/// <summary>
/// 颜色校准工具。
/// <para>
/// 从样本图像的 ROI 中自动学习 Lab 参考色，用于快速创建或更新颜色定义。
/// </para>
/// <para>
/// <b>典型现场使用流程：</b><br/>
/// 1. 操作员将标准保险丝放入工位，拍一张照片<br/>
/// 2. 对每个保险丝框选 ROI，指定颜色名称<br/>
/// 3. 调用 <see cref="LearnFromRoi"/> 自动计算 Lab 参考色<br/>
/// 4. 结果直接可用于检测，如需微调只需改 MaxDeltaE（容差）
/// </para>
/// </summary>
public static class ColorCalibrator
{
    private static readonly ColorDetectorOptions s_defaultOptions = new();

    /// <summary>
    /// 从单个 ROI 样本图像中学习 Lab 参考色。
    /// <para>
    /// 当 maxDeltaE 使用默认值时，会根据样本内部散布度（Dispersion）自动推算合理容差，
    /// 确保单样本校准也能获得安全的容差范围。
    /// </para>
    /// </summary>
    /// <param name="roiBgr">BGR 格式的 ROI 图像</param>
    /// <param name="colorName">颜色名称</param>
    /// <param name="ratingLabel">额定标签（如 "10A"），可选</param>
    /// <param name="maxDeltaE">容差，默认 12；设为 0 时完全由散布度自动推算</param>
    /// <param name="options">预处理选项，为 null 时使用默认值</param>
    /// <returns>校准得到的颜色定义</returns>
    public static ColorDefinition LearnFromRoi(
        Mat roiBgr,
        string colorName,
        string ratingLabel = "",
        double maxDeltaE = 12.0,
        ColorDetectorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(roiBgr);
        if (roiBgr.Empty())
            throw new ArgumentException("ROI image is empty.", nameof(roiBgr));

        var opts = options ?? s_defaultOptions;
        var (labMat, validMask) = ImagePreprocessor.Preprocess(roiBgr, opts);

        try
        {
            var labResult = LabStatistics.ComputeRobustLab(labMat, validMask)
                ?? throw new InvalidOperationException(
                    $"No valid pixels found in ROI for color '{colorName}'. Check image quality or preprocessing parameters.");

            // 根据样本内部散布度自动推算容差下限
            // Dispersion × 2.0 + 1.5 给出安全范围，确保同色正常变化不会误判
            var dispersionBased = Math.Clamp(labResult.Dispersion * 2.0 + 1.5, 4.0, 10.0);
            var effectiveMaxDeltaE = maxDeltaE > 0
                ? Math.Max(maxDeltaE, dispersionBased)
                : dispersionBased;

            var singleLightnessTol = Math.Clamp(effectiveMaxDeltaE * 1.2, 3.0, 15.0);
            var singleChromaTol = Math.Clamp(effectiveMaxDeltaE * 0.6, 4, 10.0);

            return new ColorDefinition
            {
                ColorName = colorName,
                RefL = Math.Round(labResult.L, 2),
                RefA = Math.Round(labResult.A, 2),
                RefB = Math.Round(labResult.B, 2),
                MaxDeltaE = Math.Round(effectiveMaxDeltaE, 1),
                LightnessTolerance = Math.Round(singleLightnessTol, 1),
                ChromaTolerance = Math.Round(singleChromaTol, 1)
            };
        }
        finally
        {
            labMat.Dispose();
            validMask.Dispose();
        }
    }

    /// <summary>
    /// 从同一颜色的多个 ROI 样本中学习 Lab 参考色（更鲁棒）。
    /// <para>
    /// 取各样本 Lab 值的均值作为参考色，并根据样本间离散度自动建议容差。
    /// </para>
    /// </summary>
    /// <param name="roiImages">多个 BGR 格式的样本 ROI 图像</param>
    /// <param name="colorName">颜色名称</param>
    /// <param name="ratingLabel">额定标签，可选</param>
    /// <param name="options">预处理选项，为 null 时使用默认值</param>
    /// <returns>合并后的颜色定义（含自动建议的容差）</returns>
    public static ColorDefinition LearnFromMultipleRois(
        IReadOnlyList<Mat> roiImages,
        string colorName,
        string ratingLabel = "",
        ColorDetectorOptions? options = null)
    {
        if (roiImages.Count == 0)
            throw new ArgumentException("At least one ROI image is required.", nameof(roiImages));

        if (roiImages.Count == 1)
            return LearnFromRoi(roiImages[0], colorName, ratingLabel, options: options);

        var opts = options ?? s_defaultOptions;
        var labValues = new List<(double L, double A, double B)>();
        var dispersions = new List<double>();

        foreach (var roi in roiImages)
        {
            var (labMat, validMask) = ImagePreprocessor.Preprocess(roi, opts);
            try
            {
                var result = LabStatistics.ComputeRobustLab(labMat, validMask);
                if (result is not null)
                {
                    labValues.Add((result.Value.L, result.Value.A, result.Value.B));
                    dispersions.Add(result.Value.Dispersion);
                }
            }
            finally
            {
                labMat.Dispose();
                validMask.Dispose();
            }
        }

        if (labValues.Count == 0)
            throw new InvalidOperationException(
                $"No valid pixels found in any ROI for color '{colorName}'.");

        var avgL = labValues.Average(v => v.L);
        var avgA = labValues.Average(v => v.A);
        var avgB = labValues.Average(v => v.B);
        var avgChroma = Math.Sqrt(avgA * avgA + avgB * avgB);

        var maxInternalDeltaE = 0.0;
        foreach (var (l, a, b) in labValues)
        {
            var de = DeltaE.Calculate(avgL, avgA, avgB, l, a, b);
            maxInternalDeltaE = Math.Max(maxInternalDeltaE, de);
        }

        var avgDispersion = dispersions.Count > 0 ? dispersions.Average() : 0.0;
        var suggestedDeltaE = Math.Clamp(
            Math.Max(maxInternalDeltaE * 2.0 + 1.5, avgDispersion * 2.0 + 1.5),
            4.0,
            10.0);

        var lStd = StdDev(labValues.Select(v => v.L));
        var cStd = StdDev(labValues.Select(v => Math.Sqrt(v.A * v.A + v.B * v.B) - avgChroma));

        var lightnessTol = Math.Clamp(Math.Max(3.0 * lStd + 2.0, suggestedDeltaE * 0.7), 4.0, 20.0);
        var chromaTol = Math.Clamp(Math.Max(3.0 * cStd + 3.0, suggestedDeltaE * 0.9), 6.0, 25.0);

        return new ColorDefinition
        {
            ColorName = colorName,
            RefL = Math.Round(avgL, 2),
            RefA = Math.Round(avgA, 2),
            RefB = Math.Round(avgB, 2),
            MaxDeltaE = Math.Round(suggestedDeltaE, 1),
            LightnessTolerance = Math.Round(lightnessTol, 1),
            ChromaTolerance = Math.Round(chromaTol, 1)
        };
    }

    /// <summary>
    /// 从完整图像中批量训练多种颜色。
    /// <para>
    /// 操作员在一张图像上标注多个 ROI 及其颜色名称，一次性完成所有颜色的校准。
    /// </para>
    /// </summary>
    /// <param name="imageMat">完整的 BGR 图像</param>
    /// <param name="colorRois">颜色名称 → ROI 区域的映射</param>
    /// <param name="options">预处理选项，为 null 时使用默认值</param>
    /// <returns>校准得到的颜色定义列表</returns>
    public static List<ColorDefinition> LearnBatch(
        Mat imageMat,
        Dictionary<string, List<CvRect>> colorRois,
        ColorDetectorOptions? options = null)
    {
        var results = new List<ColorDefinition>(colorRois.Count);

        foreach (var (colorName, rois) in colorRois)
        {
            var roiMats = new List<Mat>();
            try
            {
                foreach (var roi in rois)
                    roiMats.Add(CvTool.CropImage(imageMat,roi));
                var definition = LearnFromMultipleRois(roiMats, colorName, options: options);
                results.Add(definition);
            }
            finally
            {
                foreach (var mat in roiMats)
                    mat.Dispose();
            }
        }

        return results;
    }

    private static double StdDev(IEnumerable<double> values)
    {
        var arr = values.ToArray();
        if (arr.Length <= 1)
            return 0;

        var mean = arr.Average();
        var varSum = 0.0;
        foreach (var v in arr)
        {
            var d = v - mean;
            varSum += d * d;
        }

        return Math.Sqrt(varSum / (arr.Length - 1));
    }
}
