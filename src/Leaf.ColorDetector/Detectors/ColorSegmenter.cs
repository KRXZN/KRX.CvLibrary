using Leaf.ColorDetector.ColorScience;
using Leaf.ColorDetector.Models;
using OpenCvSharp;

namespace Leaf.ColorDetector.Detectors;

/// <summary>
/// 颜色分割器（面积统计）。
/// <para>
/// 流程：
/// 1. ROI 图高斯模糊（抑制噪声，采样与检测共用同一预处理链）
/// 2. 按 <see cref="ColorDefinition.MatchMode"/> 生成掩码：
///    Lab 容差（亮度轴 + 色度平面椭圆容差）或 RGB 分量范围（标称值±上下容差）
/// 3. 形态学开运算去除孤立小噪点
/// 4. 统计掩码像素数（面积）
/// </para>
/// <para>
/// 面积口径：ROI 内该颜色像素的绝对数量。面积上下限由调用方（配方单元）持有，
/// 用 <see cref="IsAreaOK"/> 判定。
/// </para>
/// </summary>
public class ColorSegmenter
{
    private readonly ColorSegmentOptions _options;

    public ColorSegmenter(ColorSegmentOptions? options = null)
    {
        _options = options ?? new ColorSegmentOptions();
    }

    /// <summary>
    /// 生成单个颜色的分割掩码（CV_8UC1，匹配像素为 255）。
    /// 调用方负责 Dispose。用于预览叠加与面积测量。
    /// </summary>
    public Mat BuildMask(
        Mat roiBgr,
        ColorDefinition color,
        ColorSegmentOptions? options = null
    )
    {
        var opts = options ?? _options;
        using var blurred = Blur(roiBgr, opts);
        if (color.MatchMode == ColorMatchMode.RgbRange)
        {
            var rgbMask = BuildRgbMask(blurred, color);
            ApplyMorphology(rgbMask, opts);
            return rgbMask;
        }
        using var labMat = ToLab(blurred);
        var mask = BuildRawMask(blurred, labMat, color);
        ApplyMorphology(mask, opts);
        return mask;
    }

    /// <summary>
    /// 测量单个颜色在 ROI 内的分割面积（像素数，形态学去噪后）。
    /// </summary>
    public int MeasureArea(
        Mat roiBgr,
        ColorDefinition color,
        ColorSegmentOptions? options = null
    )
    {
        using var mask = BuildMask(roiBgr, color, options);
        return Cv2.CountNonZero(mask);
    }

    /// <summary>
    /// 面积是否合格：上下限均为 0 表示不参与判定；单边为 0 表示该边不限制。
    /// </summary>
    public static bool IsAreaOK(int area, int areaMin, int areaMax)
    {
        if (areaMin == 0 && areaMax == 0)
            return true;
        if (areaMin > 0 && area < areaMin)
            return false;
        if (areaMax > 0 && area > areaMax)
            return false;
        return true;
    }

    /// <summary>高斯模糊（核大小 &lt; 3 时跳过）。</summary>
    private static Mat Blur(Mat roiBgr, ColorSegmentOptions options)
    {
        var blurred = new Mat();
        if (options.GaussianKernelSize >= 3)
        {
            var kSize = options.GaussianKernelSize | 1;
            Cv2.GaussianBlur(roiBgr, blurred, new Size(kSize, kSize), 0);
        }
        else
        {
            roiBgr.CopyTo(blurred);
        }
        return blurred;
    }

    /// <summary>
    /// BGR → Lab（8bit 编码：L∈[0,255] 对应 L*∈[0,100]，a/b 为 a*/b* + 128）。
    /// 不做亮度归一化，保证参考色与实测像素在同一色空间可比。
    /// </summary>
    private static Mat ToLab(Mat bgrMat)
    {
        var labMat = new Mat();
        Cv2.CvtColor(bgrMat, labMat, ColorConversionCodes.BGR2Lab);
        return labMat;
    }

    /// <summary>按匹配模式分派生成掩码（含形态学去噪）。</summary>
    private static Mat BuildMaskCore(
        Mat blurredBgr,
        Mat? labMat,
        ColorDefinition color,
        ColorSegmentOptions options
    )
    {
        var mask = BuildRawMask(blurredBgr, labMat, color);
        ApplyMorphology(mask, options);
        return mask;
    }

    /// <summary>按匹配模式生成原始掩码（不含形态学）。</summary>
    private static Mat BuildRawMask(Mat blurredBgr, Mat? labMat, ColorDefinition color) =>
        color.MatchMode switch
        {
            ColorMatchMode.RgbRange => BuildRgbMask(blurredBgr, color),
            ColorMatchMode.DeltaE => BuildDeltaEMask(
                labMat ?? throw new InvalidOperationException("ΔE 模式需要 Lab 图像"),
                color
            ),
            _ => BuildLabMask(
                labMat ?? throw new InvalidOperationException("Lab 模式需要 Lab 图像"),
                color
            ),
        };

    /// <summary>形态学开运算去噪（核大小 &lt; 2 时跳过）。</summary>
    private static void ApplyMorphology(Mat mask, ColorSegmentOptions options)
    {
        if (options.MorphologyKernelSize < 2)
            return;

        using var kernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse,
            new Size(options.MorphologyKernelSize, options.MorphologyKernelSize)
        );
        Cv2.MorphologyEx(mask, mask, MorphTypes.Open, kernel);
    }

    /// <summary>
    /// RGB 分量范围掩码（向量化）：像素各分量落在 标称值±容差 区间内即匹配。
    /// </summary>
    private static Mat BuildRgbMask(Mat bgrMat, ColorDefinition color)
    {
        // OpenCV Scalar 为 BGR 顺序
        var lower = new Scalar(
            Clamp255(color.NomB - color.TolBLow),
            Clamp255(color.NomG - color.TolGLow),
            Clamp255(color.NomR - color.TolRLow)
        );
        var upper = new Scalar(
            Clamp255(color.NomB + color.TolBHigh),
            Clamp255(color.NomG + color.TolGHigh),
            Clamp255(color.NomR + color.TolRHigh)
        );

        var mask = new Mat();
        Cv2.InRange(bgrMat, lower, upper, mask);
        return mask;
    }

    /// <summary>
    /// Lab 容差掩码：逐像素椭圆容差判定（亮度轴 L* 与色度平面 a*b* 分离建模）。
    /// </summary>
    private static Mat BuildLabMask(Mat labMat, ColorDefinition color)
    {
        // 容差回退规则与 ColorDetector 的评分椭球一致（tol<=0 时按 MaxDeltaE 推导）
        var maxDe = Math.Max(color.MaxDeltaE, 1e-6);
        var lightnessTol = color.LightnessTolerance > 0
            ? color.LightnessTolerance
            : maxDe * 0.9;
        var chromaTol = color.ChromaTolerance > 0 ? color.ChromaTolerance : maxDe * 1.0;

        // 转到 8bit Lab 编码空间做整数比较（L8 = L* * 255/100，a8 = a* + 128，b8 = b* + 128）
        var refL8 = color.RefL * 255.0 / 100.0;
        var refA8 = color.RefA + 128.0;
        var refB8 = color.RefB + 128.0;
        var tolL8 = Math.Max(lightnessTol * 255.0 / 100.0, 1e-6);
        var tolC = Math.Max(chromaTol, 1e-6);

        // 预计算倒数平方，逐像素避免开方
        var invTolL8Sq = 1.0 / (tolL8 * tolL8);
        var invTolCSq = 1.0 / (tolC * tolC);

        var mask = new Mat(labMat.Rows, labMat.Cols, MatType.CV_8UC1, Scalar.All(0));
        var labIndexer = labMat.GetGenericIndexer<Vec3b>();
        var maskIndexer = mask.GetGenericIndexer<byte>();

        for (var y = 0; y < labMat.Rows; y++)
        {
            for (var x = 0; x < labMat.Cols; x++)
            {
                var pixel = labIndexer[y, x];
                var dL = pixel.Item0 - refL8;
                var dA = pixel.Item1 - refA8;
                var dB = pixel.Item2 - refB8;

                var nL2 = dL * dL * invTolL8Sq;
                var nC2 = (dA * dA + dB * dB) * invTolCSq;
                if (nL2 + nC2 <= 1.0)
                    maskIndexer[y, x] = 255;
            }
        }

        return mask;
    }

    /// <summary>
    /// ΔE2000 色差掩码：逐像素与参考色算 CIEDE2000 色差，≤ 容差即匹配。
    /// <para>
    /// 逐像素 ΔE 计算较重（Pow/三角函数），但 8bit Lab 像素取值有限（同一 ROI 内重复度高），
    /// 故按像素 Lab 值缓存色差结果，避免重复计算——全图预览（几十万像素）下提速明显。
    /// </para>
    /// </summary>
    private static Mat BuildDeltaEMask(Mat labMat, ColorDefinition color)
    {
        var maxDeltaE = Math.Max(color.MaxDeltaE, 1e-6);
        var refL = color.RefL;
        var refA = color.RefA;
        var refB = color.RefB;

        var mask = new Mat(labMat.Rows, labMat.Cols, MatType.CV_8UC1, Scalar.All(0));
        var labIndexer = labMat.GetGenericIndexer<Vec3b>();
        var maskIndexer = mask.GetGenericIndexer<byte>();
        // 8bit Lab 三通道打包为缓存键（每 ROI 不同色数通常远小于像素数）
        var deltaECache = new Dictionary<int, double>();

        for (var y = 0; y < labMat.Rows; y++)
        {
            for (var x = 0; x < labMat.Cols; x++)
            {
                var pixel = labIndexer[y, x];
                var key = (pixel.Item0 << 16) | (pixel.Item1 << 8) | pixel.Item2;
                if (!deltaECache.TryGetValue(key, out var deltaE))
                {
                    deltaE = DeltaE.Calculate(
                        refL,
                        refA,
                        refB,
                        pixel.Item0 * 100.0 / 255.0,
                        pixel.Item1 - 128.0,
                        pixel.Item2 - 128.0
                    );
                    deltaECache[key] = deltaE;
                }
                if (deltaE <= maxDeltaE)
                    maskIndexer[y, x] = 255;
            }
        }

        return mask;
    }

    private static double Clamp255(double value) => Math.Clamp(value, 0, 255);
}
