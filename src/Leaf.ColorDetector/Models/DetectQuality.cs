namespace Leaf.ColorDetector.Models;

/// <summary>
/// 颜色检测结果的质量等级。
/// </summary>
public enum DetectQuality
{
    /// <summary>高置信度匹配，第1名与第2名差距明显</summary>
    High,

    /// <summary>匹配成功但第1名与第2名差距较小，建议关注</summary>
    Medium,

    /// <summary>有效像素比例偏低（可能遮挡/半插入），结果可信度降低</summary>
    LowPixelRatio,

    /// <summary>第1名与第2名得分过于接近，无法可靠区分</summary>
    Ambiguous,

    /// <summary>所有候选颜色的 ΔE 均超出容差，不属于任何已知颜色</summary>
    Unknown,

    /// <summary>有效像素数量不足，无法判定</summary>
    Insufficient
}
