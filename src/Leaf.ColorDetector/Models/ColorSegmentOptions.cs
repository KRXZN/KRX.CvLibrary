namespace Leaf.ColorDetector.Models;

/// <summary>
/// 颜色分割（面积判定）的配置参数。
/// <para>
/// 分割流程：BGR → Lab → 逐像素容差判定生成掩码 → 形态学开运算去噪 → 统计像素面积。
/// 操作员通常只需调整每种颜色的容差与面积上下限，此处的参数仅供调试和特殊环境适配。
/// </para>
/// </summary>
public class ColorSegmentOptions
{
    /// <summary>
    /// 高斯模糊核大小（必须为奇数）。用于抑制噪声对面积统计的影响。
    /// 设为 0 或 1 则跳过模糊。
    /// </summary>
    public int GaussianKernelSize { get; set; } = 3;

    /// <summary>
    /// 形态学开运算核大小。用于去除掩码中的孤立小噪点。
    /// 设为 0 或 1 则跳过形态学处理。
    /// </summary>
    public int MorphologyKernelSize { get; set; } = 3;
}
