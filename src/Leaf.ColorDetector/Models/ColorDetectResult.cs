namespace Leaf.ColorDetector.Models;

/// <summary>
/// 单个颜色检测的结果。
/// </summary>
public class ColorDetectResult
{
    /// <summary>是否检测成功（匹配到预期颜色且质量可接受）</summary>
    public bool IsMatch { get; set; }

    /// <summary>检测到的最佳匹配颜色名称</summary>
    public string DetectedColor { get; set; } = string.Empty;

    /// <summary>期望的颜色名称</summary>
    public string ExpectedColor { get; set; } = string.Empty;

    /// <summary>最佳匹配的 ΔE2000 色差值（越小越接近参考色）</summary>
    public double DeltaE { get; set; }

    /// <summary>最佳匹配的得分 (0.0-1.0)，由 ΔE 转换而来</summary>
    public double Confidence { get; set; }

    /// <summary>检测质量等级</summary>
    public DetectQuality Quality { get; set; }

    /// <summary>ROI 中的有效像素数量</summary>
    public int ValidPixelCount { get; set; }

    /// <summary>ROI 中的总像素数量</summary>
    public int TotalPixelCount { get; set; }

    /// <summary>有效像素占总像素的比例</summary>
    public double ValidPixelRatio => TotalPixelCount > 0
        ? (double)ValidPixelCount / TotalPixelCount
        : 0;

    /// <summary>样本的 Lab 测量值 (L*)</summary>
    public double MeasuredL { get; set; }

    /// <summary>样本的 Lab 测量值 (a*)</summary>
    public double MeasuredA { get; set; }

    /// <summary>样本的 Lab 测量值 (b*)</summary>
    public double MeasuredB { get; set; }

    /// <summary>有效像素的内部散布度（平均 ΔE76 到中心，越小表示颜色越均匀）</summary>
    public double Dispersion { get; set; }

    /// <summary>ROI 内颜色空间分布是否一致（false 可能表示半插入/倾斜/背景混入）</summary>
    public bool IsSpatiallyConsistent { get; set; } = true;

    /// <summary>各候选颜色的匹配得分（按 ΔE 升序排列）</summary>
    public List<ColorScore> AllScores { get; set; } = [];

    /// <summary>检测耗时</summary>
    public TimeSpan Elapsed { get; set; }

    /// <summary>
    /// 结果摘要信息
    /// </summary>
    public string Summary => Quality switch
    {
        DetectQuality.Insufficient => $"NG: 有效像素不足 ({ValidPixelCount})",
        DetectQuality.Unknown => $"NG: 未知颜色 (Lab={MeasuredL:F1},{MeasuredA:F1},{MeasuredB:F1})",
        DetectQuality.Ambiguous => $"NG: 颜色模糊 {DetectedColor} ΔE={DeltaE:F1}",
        _ when IsMatch => $"OK: {DetectedColor} ΔE={DeltaE:F1} [{Quality}]",
        _ => $"NG: 期望={ExpectedColor}, 检测={DetectedColor} ΔE={DeltaE:F1}"
    };
}

/// <summary>
/// 单个候选颜色的匹配得分。
/// </summary>
public class ColorScore
{
    /// <summary>颜色名称</summary>
    public string ColorName { get; set; } = string.Empty;

    /// <summary>与该颜色参考值的 ΔE2000 色差</summary>
    public double DeltaE { get; set; }

    /// <summary>由 ΔE 转换的得分 (0.0-1.0)</summary>
    public double Score { get; set; }
}
