namespace Leaf.ColorDetector.Models;

/// <summary>
/// 保险丝颜色定义。
/// <para>
/// 每种颜色由一个 Lab 色彩空间的参考色中心点表示，
/// 检测时通过 ΔE2000 色差公式计算样本与参考色的距离。
/// 操作员只需调整一个参数：<see cref="MaxDeltaE"/>（容差）。
/// </para>
/// <para>
/// Lab 参考色可通过 <see cref="Calibration.ColorCalibrator"/> 从样本图像自动学习获得。
/// </para>
/// </summary>
public class ColorDefinition
{
    /// <summary>颜色名称（如 "Red", "Blue", "Yellow"）</summary>
    public string ColorName { get; set; } = string.Empty;

    /// <summary>Lab 参考色 — L* 明度分量 (0~100)</summary>
    public double RefL { get; set; }

    /// <summary>Lab 参考色 — a* 红绿分量 (-128~127)</summary>
    public double RefA { get; set; }

    /// <summary>Lab 参考色 — b* 黄蓝分量 (-128~127)</summary>
    public double RefB { get; set; }

    /// <summary>
    /// 最大允许色差（ΔE2000）。
    /// <para>
    /// 这是操作员唯一需要调整的参数。
    /// 参考值：5=严格匹配，12=常规工业检测，20=宽松容差。
    /// </para>
    /// </summary>
    public double MaxDeltaE { get; set; } = 5;

    /// <summary>
    /// 亮度方向的自适应容差（L* 轴，0 表示自动回退到 MaxDeltaE 推导）。
    /// 用于增强对明暗漂移和近色区分的鲁棒性。
    /// </summary>
    public double LightnessTolerance { get; set; }

    /// <summary>
    /// 色度方向的自适应容差（a*b* 平面，0 表示自动回退到 MaxDeltaE 推导）。
    /// 与 <see cref="LightnessTolerance"/> 共同形成椭球容差。
    /// </summary>
    public double ChromaTolerance { get; set; }
}
