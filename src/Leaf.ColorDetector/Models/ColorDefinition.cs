using System.Text.Json.Serialization;

namespace Leaf.ColorDetector.Models;

/// <summary>
/// 颜色定义。
/// <para>
/// 采样时 Lab 参考色（<see cref="RefL"/>/<see cref="RefA"/>/<see cref="RefB"/>）与
/// RGB 标称值（<see cref="NomR"/>/<see cref="NomG"/>/<see cref="NomB"/>）同时写入，
/// 检测时按 <see cref="MatchMode"/> 取用对应的一组容差生成像素掩码：
/// </para>
/// <para>
/// 1. <see cref="ColorMatchMode.LabTolerance"/>：亮度轴（<see cref="LightnessTolerance"/>）
/// 与色度平面（<see cref="ChromaTolerance"/>）分离建模的椭圆容差；<br/>
/// 2. <see cref="ColorMatchMode.RgbRange"/>：各分量落在 标称值±上下容差 区间内。
/// </para>
/// <para>
/// 统计掩码面积（像素数）并与 <see cref="AreaMin"/>/<see cref="AreaMax"/> 比较判定。
/// 参考值可通过 <see cref="Calibration.ColorCalibrator"/> 框选采样或点选取色获得。
/// </para>
/// </summary>
public class ColorDefinition
{
    /// <summary>稳定标识：配方单元通过它引用颜色库中的条目。</summary>
    public Guid Id { get; set; } = Guid.NewGuid();

    /// <summary>颜色名称（如 "Red", "Blue", "Yellow"）</summary>
    public string ColorName { get; set; } = string.Empty;

    /// <summary>匹配模式，决定检测时用哪组参考值与容差。</summary>
    public ColorMatchMode MatchMode { get; set; } = ColorMatchMode.LabTolerance;

    // ===== Lab 模式（Lab 参考色 + 椭圆容差） =====

    /// <summary>Lab 参考色 — L* 明度分量 (0~100)</summary>
    public double RefL { get; set; }

    /// <summary>Lab 参考色 — a* 红绿分量 (-128~127)</summary>
    public double RefA { get; set; }

    /// <summary>Lab 参考色 — b* 黄蓝分量 (-128~127)</summary>
    public double RefB { get; set; }

    /// <summary>
    /// 最大允许色差（ΔE2000）。
    /// <para>
    /// 亮度/色度容差为 0 时按其自动推导，也用于兼容旧版平均色匹配路径。
    /// 参考值：5=严格匹配，12=常规工业检测，20=宽松容差。
    /// </para>
    /// </summary>
    public double MaxDeltaE { get; set; } = 5;

    /// <summary>
    /// 亮度方向的容差（L* 轴，0 表示自动回退到 MaxDeltaE 推导）。
    /// 用于增强对明暗漂移和近色区分的鲁棒性。
    /// </summary>
    public double LightnessTolerance { get; set; }

    /// <summary>
    /// 色度方向的容差（a*b* 平面，0 表示自动回退到 MaxDeltaE 推导）。
    /// 与 <see cref="LightnessTolerance"/> 共同形成椭球容差。
    /// </summary>
    public double ChromaTolerance { get; set; }

    // ===== RGB 模式（分量标称值 + 每分量低/高容差） =====

    /// <summary>RGB 标称值 — R 分量 (0~255)</summary>
    public double NomR { get; set; }

    /// <summary>RGB 标称值 — G 分量 (0~255)</summary>
    public double NomG { get; set; }

    /// <summary>RGB 标称值 — B 分量 (0~255)</summary>
    public double NomB { get; set; }

    /// <summary>R 分量低容差：下界 = NomR - 此值。</summary>
    public double TolRLow { get; set; } = 10;

    /// <summary>R 分量高容差：上界 = NomR + 此值。</summary>
    public double TolRHigh { get; set; } = 10;

    /// <summary>G 分量低容差：下界 = NomG - 此值。</summary>
    public double TolGLow { get; set; } = 10;

    /// <summary>G 分量高容差：上界 = NomG + 此值。</summary>
    public double TolGHigh { get; set; } = 10;

    /// <summary>B 分量低容差：下界 = NomB - 此值。</summary>
    public double TolBLow { get; set; } = 10;

    /// <summary>B 分量高容差：上界 = NomB + 此值。</summary>
    public double TolBHigh { get; set; } = 10;

    // ===== 面积判定（跟随颜色：同一颜色在多个 ROI 共用同一套上下限） =====

    /// <summary>
    /// 是否启用该颜色。禁用的颜色不参与面积判定（引用它的 ROI 跳过颜色检测）。
    /// </summary>
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// 面积下限（像素数）。检测时该颜色在 ROI 内的分割面积低于此值判 NG。
    /// 0 表示不限制下限。
    /// </summary>
    public int AreaMin { get; set; }

    /// <summary>
    /// 面积上限（像素数）。检测时该颜色在 ROI 内的分割面积高于此值判 NG。
    /// 0 表示不限制上限（AreaMin 与 AreaMax 均为 0 时该颜色面积不参与判定）。
    /// </summary>
    public int AreaMax { get; set; }

    /// <summary>参考色的 RGB 十六进制串（UI 色块显示用，由 RGB 标称值生成）</summary>
    [JsonIgnore]
    public string RgbHex =>
        $"#{(int)Math.Clamp(NomR, 0, 255):X2}{(int)Math.Clamp(NomG, 0, 255):X2}{(int)Math.Clamp(NomB, 0, 255):X2}";

    /// <summary>参考色文本（采样后供核对）。</summary>
    [JsonIgnore]
    public string RefColorText =>
        $"R:{NomR:F0} G:{NomG:F0} B:{NomB:F0}  L*:{RefL:F1} a*:{RefA:F1} b*:{RefB:F1}";

    /// <summary>
    /// 深拷贝（颜色列表在配方单元间复制时避免共享实例）。
    /// </summary>
    public ColorDefinition Clone() =>
        new()
        {
            Id = Id,
            ColorName = ColorName,
            MatchMode = MatchMode,
            RefL = RefL,
            RefA = RefA,
            RefB = RefB,
            MaxDeltaE = MaxDeltaE,
            LightnessTolerance = LightnessTolerance,
            ChromaTolerance = ChromaTolerance,
            NomR = NomR,
            NomG = NomG,
            NomB = NomB,
            TolRLow = TolRLow,
            TolRHigh = TolRHigh,
            TolGLow = TolGLow,
            TolGHigh = TolGHigh,
            TolBLow = TolBLow,
            TolBHigh = TolBHigh,
            IsEnabled = IsEnabled,
            AreaMin = AreaMin,
            AreaMax = AreaMax,
        };
}
