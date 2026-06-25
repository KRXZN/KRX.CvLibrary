using System.Diagnostics;
using CvCommon;
using CvLibrary.OpenCV;
using Leaf.ColorDetector.ColorScience;
using Leaf.ColorDetector.Models;
using Leaf.ColorDetector.Preprocessing;
using Microsoft.Extensions.Logging;
using OpenCvSharp;

namespace Leaf.ColorDetector.Detectors;

/// <summary>
/// 颜色检测器。
/// <para>
/// 检测流程：
/// 1. ROI 裁剪 → 中心区域采样
/// 2. 高斯模糊降噪
/// 3. HSV 生成掩码（排除非消色低亮度像素，消色系豁免）
/// 4. BGR → Lab 转换
/// 5. 对有效像素提取鲁棒 Lab 统计量（MAD 离群值剔除 + 中位数 a*b* + 截断均值 L*）
/// 6. 对每种候选颜色计算 ΔE2000 色差 → 高斯评分函数转为得分
/// 7. 质量门控（像素不足/空间不一致/未知颜色/模糊判定）
/// 8. 输出最佳匹配及诊断信息
/// </para>
/// </summary>
public class ColorDetector : IColorDetector
{
    private readonly ColorDetectorOptions _options;
    private readonly ILogger<ColorDetector>? _logger;

    public ColorDetector(
        ColorDetectorOptions? options = null,
        ILogger<ColorDetector>? logger = null)
    {
        _options = options ?? new ColorDetectorOptions();
        _logger = logger;
    }

    /// <inheritdoc />
    public ColorDetectResult Detect(
        Mat imageMat,
        CvRect roi,
        string expectedColor,
        IReadOnlyList<ColorDefinition> colorDefinitions)
    {
        using var roiMat = CvTool.CropImage(imageMat, roi);
        return Detect(roiMat, expectedColor, colorDefinitions);
    }

    /// <inheritdoc />
    public ColorDetectResult Detect(
        Mat roiMat,
        string expectedColor,
        IReadOnlyList<ColorDefinition> colorDefinitions)
    {
        var timestamp = Stopwatch.GetTimestamp();
        var totalPixels = roiMat.Rows * roiMat.Cols;

        if (roiMat.Empty() || totalPixels == 0)
            return BuildResult(
                expectedColor, totalPixels, 0,
                0, 0, 0, 0, true,
                [], DetectQuality.Insufficient, timestamp);

        // 预处理 + Lab 统计量提取
        var (labMat, validMask) = ImagePreprocessor.Preprocess(roiMat, _options);
        totalPixels = labMat.Rows * labMat.Cols;

        LabStatistics.LabResult? labResult;
        try
        {
            labResult = LabStatistics.ComputeRobustLab(labMat, validMask);
        }
        finally
        {
            labMat.Dispose();
            validMask.Dispose();
        }

        if (labResult is null || labResult.Value.ValidCount < _options.MinValidPixelCount)
            return BuildResult(
                expectedColor, totalPixels, labResult?.ValidCount ?? 0,
                0, 0, 0, 0, true,
                [], DetectQuality.Insufficient, timestamp);

        var lab = labResult.Value;

        // 对所有候选颜色计算匹配分数
        var scores = ComputeMatchScores(lab, colorDefinitions);

        // 按 ΔE 升序排列
        scores.Sort((a, b) => a.DeltaE.CompareTo(b.DeltaE));

        // 质量门控
        var quality = EvaluateQuality(scores, lab.ValidCount, totalPixels, lab.IsSpatiallyConsistent);

        return BuildResult(
            expectedColor, totalPixels, lab.ValidCount,
            lab.L, lab.A, lab.B, lab.Dispersion, lab.IsSpatiallyConsistent,
            scores, quality, timestamp);
    }

    /// <summary>
    /// 对所有候选颜色计算 ΔE2000 + 自适应椭球距离 + 高斯评分。
    /// </summary>
    private List<ColorScore> ComputeMatchScores(
        LabStatistics.LabResult lab,
        IReadOnlyList<ColorDefinition> definitions)
    {
        var scores = new List<ColorScore>(definitions.Count);

        foreach (var colorDef in definitions)
        {
            var deltaE = DeltaE.Calculate(
                colorDef.RefL, colorDef.RefA, colorDef.RefB,
                lab.L, lab.A, lab.B);

            // 椭球容差：将 L* 与 a*b* 分离建模，增强近色区分与亮度漂移鲁棒性
            var adaptiveDistance = ComputeAdaptiveDistance(colorDef, lab.L, lab.A, lab.B);

            var score = CalculateCompositeScore(
                deltaE,
                adaptiveDistance,
                colorDef.MaxDeltaE,
                lab.Dispersion,
                lab.IsSpatiallyConsistent);

            scores.Add(new ColorScore
            {
                ColorName = colorDef.ColorName,
                DeltaE = Math.Round(deltaE, 2),
                Score = Math.Round(score, 4)
            });
        }

        return scores;
    }

    /// <summary>
    /// 评估检测质量。
    /// </summary>
    private DetectQuality EvaluateQuality(
        List<ColorScore> scores, int validCount, int totalPixels, bool isSpatiallyConsistent)
    {
        // 有效像素比例过低
        var validRatio = totalPixels > 0 ? (double)validCount / totalPixels : 0;
        if (validRatio < _options.MinValidPixelRatio)
            return DetectQuality.LowPixelRatio;

        // 所有颜色得分都为 0（ΔE 全部超出容差）
        if (scores.Count == 0 || scores[0].Score <= 0)
            return DetectQuality.Unknown;

        // 第1名和第2名差距太小
        if (scores.Count >= 2
            && scores[0].Score > 0
            && scores[1].Score > 0
            && scores[0].Score - scores[1].Score < _options.MinConfidenceGap)
        {
            return DetectQuality.Ambiguous;
        }

        return scores[0].Score >= 0.5 ? DetectQuality.High : DetectQuality.Medium;
    }

    /// <summary>
    /// 计算融合评分：ΔE2000 + 自适应椭球距离 + 空间一致性惩罚。
    /// </summary>
    private double CalculateCompositeScore(
        double deltaE,
        double adaptiveDistance,
        double maxDeltaE,
        double dispersion,
        bool isSpatiallyConsistent)
    {
        if (maxDeltaE <= 0)
            return 0;

        var sigmaFactor = Math.Max(_options.ScoreSigmaFactor, 0.1);
        var sigma = maxDeltaE / sigmaFactor;

        var normalizedDe = deltaE / maxDeltaE;
        var normalizedAdaptive = adaptiveDistance;

        var adaptiveWeight = Math.Clamp(_options.AdaptiveToleranceWeight, 0, 1);
        var combinedNormalized =
            normalizedDe * (1.0 - adaptiveWeight) + normalizedAdaptive * adaptiveWeight;

        if (combinedNormalized >= 1.0)
            return 0.0;

        var baseScore = Math.Exp(-0.5 * (combinedNormalized * maxDeltaE / sigma) * (combinedNormalized * maxDeltaE / sigma));

        var spatialPenalty = isSpatiallyConsistent
            ? 1.0
            : Math.Clamp(_options.SpatialPenaltyFactor, 0, 1);

        return Math.Clamp(baseScore * spatialPenalty, 0, 1);
    }

    /// <summary>
    /// 计算自适应椭球距离（L* 轴 + a*b* 平面）。
    /// 返回值≈1 表示达到容差边界。
    /// </summary>
    private static double ComputeAdaptiveDistance(
        ColorDefinition colorDef,
        double measuredL,
        double measuredA,
        double measuredB)
    {
        var dL = measuredL - colorDef.RefL;
        var dA = measuredA - colorDef.RefA;
        var dB = measuredB - colorDef.RefB;

        var dC = Math.Sqrt(dA * dA + dB * dB);

        var maxDe = Math.Max(colorDef.MaxDeltaE, 1e-6);
        var lightnessTol = colorDef.LightnessTolerance > 0
            ? colorDef.LightnessTolerance
            : maxDe * 0.9;
        var chromaTol = colorDef.ChromaTolerance > 0
            ? colorDef.ChromaTolerance
            : maxDe * 1.0;

        var nL = dL / Math.Max(lightnessTol, 1e-6);
        var nC = dC / Math.Max(chromaTol, 1e-6);

        return Math.Sqrt(nL * nL + nC * nC);
    }

    private static ColorDetectResult BuildResult(
        string expectedColor, int totalPixels, int validPixelCount,
        double measuredL, double measuredA, double measuredB,
        double dispersion, bool isSpatiallyConsistent,
        List<ColorScore> scores, DetectQuality quality,
        long timestamp)
    {
        var best = scores.Count > 0 ? scores[0] : null;
        var bestColor = best?.ColorName ?? string.Empty;
        var bestDeltaE = best?.DeltaE ?? double.MaxValue;
        var bestScore = best?.Score ?? 0;

        // 只有质量可接受且颜色匹配时才算 OK
        var isQualityAcceptable = quality is DetectQuality.High or DetectQuality.Medium;
        var isMatch = isQualityAcceptable
                      && bestScore > 0
                      && string.Equals(bestColor, expectedColor, StringComparison.OrdinalIgnoreCase);

        return new ColorDetectResult
        {
            IsMatch = isMatch,
            DetectedColor = bestColor,
            ExpectedColor = expectedColor,
            DeltaE = bestDeltaE,
            Confidence = bestScore,
            Quality = quality,
            ValidPixelCount = validPixelCount,
            TotalPixelCount = totalPixels,
            MeasuredL = Math.Round(measuredL, 2),
            MeasuredA = Math.Round(measuredA, 2),
            MeasuredB = Math.Round(measuredB, 2),
            Dispersion = Math.Round(dispersion, 2),
            IsSpatiallyConsistent = isSpatiallyConsistent,
            AllScores = scores,
            Elapsed = Stopwatch.GetElapsedTime(timestamp)
        };
    }
}
