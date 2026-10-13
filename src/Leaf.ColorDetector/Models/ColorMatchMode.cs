using System.ComponentModel;

namespace Leaf.ColorDetector.Models;

/// <summary>
/// 颜色匹配模式：决定像素"属于某个颜色"的判定方式（用于分割面积统计）。
/// </summary>
public enum ColorMatchMode
{
    /// <summary>
    /// Lab 容差：亮度轴（L*）与色度平面（a*b*）分离建模的椭圆容差。
    /// 光照鲁棒，容差可按亮度/色度分别调整。
    /// </summary>
    [Description("Lab容差")]
    LabTolerance,

    /// <summary>
    /// RGB 分量范围：像素各分量落在 标称值±容差 区间内即计入面积。
    /// 直观，但对光照变化敏感。
    /// </summary>
    [Description("RGB分量范围")]
    RgbRange,

    /// <summary>
    /// ΔE2000 色差容差：像素与参考色的 CIEDE2000 色差 ≤ 容差即计入面积。
    /// 感知均匀，容差可跨颜色复用；不能单独放宽亮度方向，逐像素计算较重。
    /// <para>追加在末尾：保持既有枚举数值不变（配方 JSON 按数值序列化）。</para>
    /// </summary>
    [Description("ΔE色差")]
    DeltaE,
}
