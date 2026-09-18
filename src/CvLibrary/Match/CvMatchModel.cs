using CvCommon;
using OpenCvSharp;

namespace CvLibrary.OpenCV.Match
{
    /// <summary>
    /// 模板匹配模型抽象基类。
    /// 持有预计算的模板金字塔数据和角度配置，子类实现具体的匹配算法。
    /// </summary>
    public abstract class CvMatchModel : IDisposable
    {
        #region Protected Fields

        /// <summary>金字塔层数。</summary>
        protected int NumLevels;

        /// <summary>金字塔模板数据 [level][angleIndex]。</summary>
        protected TemplateEntry[][]? Pyramid;

        /// <summary>每层的角度列表（度）。</summary>
        protected double[][]? LayerAngles;

        /// <summary>原始模板尺寸（L0，无旋转）。</summary>
        protected CvSize TemplateSize;

        /// <summary>原始模板中心（L0，无旋转）。</summary>
        protected Point2d TemplateCenter;

        /// <summary>角度配置 — 起始角度（度）。</summary>
        protected double ModelAngleStart;

        /// <summary>角度配置 — 角度范围（度）。</summary>
        protected double ModelAngleExtent;

        /// <summary>L0 角度步长（度）。</summary>
        protected double ModelAngleStep;

        /// <summary>极性模式。</summary>
        protected MatchMetric ModelMetric;

        /// <summary>是否已释放。</summary>
        protected bool Disposed;

        #endregion

        #region Internal Types

        /// <summary>
        /// 预计算的单个模板条目。
        /// </summary>
        protected sealed class TemplateEntry
        {
            /// <summary>模板图像 (CV_8UC1)。</summary>
            public required Mat Image { get; init; }

            /// <summary>有效像素掩码 (CV_8UC1)，255=有效，0=填充。</summary>
            public required Mat Mask { get; init; }

            /// <summary>该角度的模板中心偏移（从 Image 左上角到模板中心的偏移）。</summary>
            public required Point2d CenterOffset { get; init; }

            /// <summary>该条目的角度（度）。</summary>
            public double Angle { get; init; }
        }

        #endregion

        #region Public Properties

        /// <summary>金字塔层数（只读）。</summary>
        public int Levels => NumLevels;

        /// <summary>模板原始尺寸。</summary>
        public CvSize TemplateDimensions => TemplateSize;

        #endregion

        #region IDisposable

        /// <inheritdoc/>
        public void Dispose()
        {
            if (Disposed) return;
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// 释放托管和非托管资源。
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (Disposed) return;

            if (disposing)
            {
                ReleasePyramid();
            }

            Disposed = true;
        }

        /// <summary>
        /// 析构函数 — 安全网，防止忘记 Dispose。
        /// </summary>
        ~CvMatchModel()
        {
            Dispose(disposing: false);
        }

        /// <summary>
        /// 释放金字塔中所有 Mat 资源。
        /// </summary>
        protected void ReleasePyramid()
        {
            if (Pyramid == null) return;

            for (int level = 0; level < Pyramid.Length; level++)
            {
                if (Pyramid[level] == null) continue;
                for (int a = 0; a < Pyramid[level].Length; a++)
                {
                    Pyramid[level][a]?.Image?.Dispose();
                    Pyramid[level][a]?.Mask?.Dispose();
                }
            }
            Pyramid = null;
            LayerAngles = null;
        }

        #endregion

        #region Public API

        /// <summary>
        /// 在搜索图中查找模板的所有匹配实例。
        /// </summary>
        /// <param name="searchImage">搜索图像。</param>
        /// <param name="options">查找参数。</param>
        /// <returns>匹配结果列表（按得分降序）。</returns>
        /// <exception cref="ObjectDisposedException">模型已释放。</exception>
        /// <exception cref="ArgumentException">搜索图为空。</exception>
        public abstract IReadOnlyList<CvMatchResult> FindMatches(
            Mat searchImage,
            FindMatchesOptions? options = null);

        #endregion

        #region Pyramid Building Helpers

        /// <summary>
        /// 构建图像金字塔。L0 = 原始图，后续逐层降采样。
        /// </summary>
        protected static Mat[] BuildSearchPyramid(Mat src, int levels)
        {
            var pyramid = new Mat[levels];
            pyramid[0] = src.Clone();

            for (int i = 1; i < levels; i++)
            {
                pyramid[i] = DownsampleLayer(pyramid[i - 1]);
            }

            return pyramid;
        }

        /// <summary>
        /// 自适应核大小的降采样。
        /// 最小边 ≥ 30px 用 PyrDown（5×5 核），否则用 3×3 高斯核。
        /// </summary>
        protected static Mat DownsampleLayer(Mat src)
        {
            int minSide = Math.Min(src.Width, src.Height);

            if (minSide >= 30)
            {
                var dst = new Mat();
                Cv2.PyrDown(src, dst);
                return dst;
            }
            else
            {
                var blurred = new Mat();
                Cv2.GaussianBlur(src, blurred, new Size(3, 3), 0);
                var dst = new Mat();
                Cv2.Resize(blurred, dst, new Size(src.Width / 2, src.Height / 2),
                    0, 0, InterpolationFlags.Cubic);
                blurred.Dispose();
                return dst;
            }
        }

        /// <summary>
        /// 对模板和 mask 做旋转（统一使用 WarpAffine，不使用特殊角度路径）。
        /// </summary>
        protected static (Mat rotatedImage, Mat rotatedMask) RotateTemplateWithMask(
            Mat template, Mat mask, double angle)
        {
            // 0° 快捷路径：直接 clone，避免 WarpAffine 插值误差
            double normalizedAngle = ((angle % 360) + 360) % 360;
            if (Math.Abs(normalizedAngle) < 1e-6)
                return (template.Clone(), mask.Clone());

            Point2f center = new(template.Width / 2f, template.Height / 2f);
            Mat rotationMatrix = Cv2.GetRotationMatrix2D(center, angle, 1.0);
            var bbox = new RotatedRect(center, template.Size(), (float)angle).BoundingRect();

            // 调整平移，防止裁剪
            rotationMatrix.Set(0, 2,
                rotationMatrix.At<double>(0, 2) + bbox.Width / 2.0 - center.X);
            rotationMatrix.Set(1, 2,
                rotationMatrix.At<double>(1, 2) + bbox.Height / 2.0 - center.Y);

            var rotatedImage = new Mat();
            var rotatedMask = new Mat();

            Cv2.WarpAffine(template, rotatedImage, rotationMatrix, bbox.Size,
                InterpolationFlags.Cubic, BorderTypes.Constant, Scalar.Black);
            Cv2.WarpAffine(mask, rotatedMask, rotationMatrix, bbox.Size,
                InterpolationFlags.Linear, BorderTypes.Constant, Scalar.Black);  // mask OK with Linear

            rotationMatrix.Dispose();

            return (rotatedImage, rotatedMask);
        }

        /// <summary>
        /// 计算自动金字塔层数。
        /// 保守策略：顶层模板（旋转前）不小于 25px，避免过度降采样导致 NCC 不可靠。
        /// </summary>
        protected static int CalculateAutoLevels(double templateWidth, double templateHeight)
        {
            double minSide = Math.Min(templateWidth, templateHeight);
            int levels = (int)Math.Floor(Math.Log(minSide / 25.0, 2.0));
            levels = Math.Max(1, levels);

            // 确保最顶层（旋转膨胀后）≥ 35px — 留足余量给大角度旋转。
            double topSide = minSide / Math.Pow(2, levels - 1);
            double topRotated = topSide * Math.Sqrt(2);
            if (topRotated < 35 && levels > 1)
                levels--;

            return levels;
        }

        /// <summary>
        /// 生成某一层的角度列表。
        /// </summary>
        protected static double[] GenerateAnglesForLevel(
            double angleStart, double angleExtent, double baseStep, int level)
        {
            double levelStep = baseStep * Math.Pow(2, level);
            if (angleExtent <= 0 || levelStep >= angleExtent)
            {
                // 只有一个角度或不需要旋转
                return new[] { angleStart + angleExtent / 2.0 };
            }

            int count = (int)Math.Ceiling(angleExtent / levelStep) + 1;
            var angles = new double[count];
            for (int i = 0; i < count; i++)
            {
                angles[i] = angleStart + i * levelStep;
                if (angles[i] > angleStart + angleExtent)
                    angles[i] = angleStart + angleExtent;
            }

            return angles;
        }

        #endregion

        #region Sub-Pixel & Angle Refinement

        /// <summary>
        /// 亚像素位置精炼（3×3 邻域抛物线拟合）。
        /// protected virtual 允许子类覆盖。
        /// </summary>
        /// <param name="responseMap">NCC 响应图 (CV_32FC1)。</param>
        /// <param name="peakPixel">整数峰值像素坐标。</param>
        /// <returns>亚像素精炼后的坐标。</returns>
        protected virtual Point2d RefineSubPixel(Mat responseMap, Point peakPixel)
        {
            int x = peakPixel.X;
            int y = peakPixel.Y;

            if (x <= 0 || x >= responseMap.Cols - 1 ||
                y <= 0 || y >= responseMap.Rows - 1)
            {
                // 在边界上，不做精炼
                return new Point2d(x, y);
            }

            // 垂直方向（Y）抛物线拟合
            float v0 = responseMap.At<float>(y - 1, x);
            float v1 = responseMap.At<float>(y, x);
            float v2 = responseMap.At<float>(y + 1, x);

            double denomY = v0 + v2 - 2 * v1;
            double dy = Math.Abs(denomY) > 1e-10 ? (v0 - v2) / (2 * denomY) : 0;

            // 水平方向（X）抛物线拟合
            float h0 = responseMap.At<float>(y, x - 1);
            float h1 = v1; // same as center
            float h2 = responseMap.At<float>(y, x + 1);

            double denomX = h0 + h2 - 2 * h1;
            double dx = Math.Abs(denomX) > 1e-10 ? (h0 - h2) / (2 * denomX) : 0;

            // 限制偏移量在合理范围 [-1, 1]
            dx = Math.Max(-1, Math.Min(dx, 1));
            dy = Math.Max(-1, Math.Min(dy, 1));

            return new Point2d(x + dx, y + dy);
        }

        /// <summary>
        /// 角度亚像素精炼（三个邻接角度的分数抛物线插值）。
        /// </summary>
        /// <param name="responseMaps">三个相邻角度的响应图。</param>
        /// <param name="angles">三个角度值。</param>
        /// <param name="position">在每张响应图上的位置（模板中心坐标）。</param>
        /// <returns>亚像素精炼后的角度（度）。</returns>
        protected static double RefineAngle(
            Mat[] responseMaps, double[] angles, Point position)
        {
            if (responseMaps.Length != 3 || angles.Length != 3)
                return angles.Length > 0 ? angles[0] : 0;

            // 在邻域 3×3 范围内取最高分
            float s0 = GetLocalMax(responseMaps[0], position);
            float s1 = GetLocalMax(responseMaps[1], position);
            float s2 = GetLocalMax(responseMaps[2], position);

            double denom = s0 + s2 - 2 * s1;
            if (Math.Abs(denom) < 1e-10)
                return angles[1];

            double fraction = (s0 - s2) / (2 * denom);
            double step = angles[1] - angles[0];
            return angles[1] + fraction * step;
        }

        private static float GetLocalMax(Mat responseMap, Point pos)
        {
            float maxVal = responseMap.At<float>(pos.Y, pos.X);
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = pos.X + dx;
                    int ny = pos.Y + dy;
                    if (nx >= 0 && nx < responseMap.Cols &&
                        ny >= 0 && ny < responseMap.Rows)
                    {
                        maxVal = Math.Max(maxVal, responseMap.At<float>(ny, nx));
                    }
                }
            }
            return maxVal;
        }

        #endregion

        #region NMS Helpers

        /// <summary>
        /// 非极大值抑制（后处理方式，用于候选数较少的层）。
        /// </summary>
        protected static List<(CvRect Box, double Score)> ApplyNMS(
            List<(CvRect Box, double Score)> candidates, double overlapThreshold)
        {
            if (candidates.Count == 0)
                return candidates;

            var sorted = new List<(CvRect Box, double Score)>(
                candidates.OrderByDescending(c => c.Score));
            var result = new List<(CvRect Box, double Score)>();

            while (sorted.Count > 0)
            {
                var top = sorted[0];
                result.Add(top);
                sorted.RemoveAt(0);

                var remaining = new List<(CvRect Box, double Score)>();
                foreach (var other in sorted)
                {
                    double iou = CalculateIoU(top.Box, other.Box);
                    if (iou < overlapThreshold)
                        remaining.Add(other);
                }
                sorted = remaining;
            }

            return result;
        }

        /// <summary>
        /// 计算两个矩形的 IoU（交并比）。
        /// </summary>
        protected static double CalculateIoU(CvRect a, CvRect b)
        {
            double interLeft = Math.Max(a.Left, b.Left);
            double interTop = Math.Max(a.Top, b.Top);
            double interRight = Math.Min(a.Right, b.Right);
            double interBottom = Math.Min(a.Bottom, b.Bottom);

            double interWidth = interRight - interLeft;
            double interHeight = interBottom - interTop;

            if (interWidth <= 0 || interHeight <= 0)
                return 0;

            double interArea = interWidth * interHeight;
            double unionArea = a.Width * a.Height + b.Width * b.Height - interArea;

            return interArea / unionArea;
        }

        /// <summary>
        /// 归一化 NCC 原始分数到 [0, 1]。
        /// </summary>
        protected double NormalizeScore(float rawNcc)
        {
            // NaN/Infinity can arise from MatchTemplate on zero-variance regions (e.g., black borders).
            if (float.IsNaN(rawNcc) || float.IsInfinity(rawNcc))
                return 0.0;

            double value = ModelMetric == MatchMetric.IgnoreGlobalPolarity
                ? Math.Abs(rawNcc)
                : rawNcc;
            return (value + 1.0) / 2.0;
        }

        /// <summary>
        /// 计算某层的自适应分数阈值。
        /// 越粗的层级（高 level）越宽松——降采样会自然降低 NCC 质量。
        /// </summary>
        protected double GetAdaptiveThreshold(double baseMinScore, int level, int topLevel)
        {
            // Coarser (higher) levels get more relaxation because downsampling degrades NCC.
            double relaxation = level * 0.12;
            return Math.Max(0.4, baseMinScore - relaxation);
        }

        #endregion

        #region Validation

        /// <summary>
        /// 校验模板输入。
        /// </summary>
        protected static void ValidateTemplate(Mat template)
        {
            if (template == null || template.Empty())
                throw new ArgumentException("Template image is null or empty.");

            if (template.Width < 10 || template.Height < 10)
                throw new ArgumentException(
                    $"Template too small ({template.Width}×{template.Height}). Minimum is 10×10.");
        }

        /// <summary>
        /// 校验参数合法性后执行查找。子类应在 FindMatches 开头调用。
        /// </summary>
        protected void ValidateFindMatches(Mat searchImage, FindMatchesOptions options)
        {
            // net472 迁移：ObjectDisposedException 无 ThrowIf
            if (Disposed)
                throw new ObjectDisposedException(GetType().Name);

            if (searchImage == null || searchImage.Empty())
                throw new ArgumentException("Search image is null or empty.");

            if (options.MinScore < 0 || options.MinScore > 1)
                throw new ArgumentOutOfRangeException(
                    nameof(options.MinScore), "MinScore must be in [0, 1].");
        }

        #endregion
    }
}
