using OpenCvSharp;

namespace CvLibrary.Tests.Match
{
    /// <summary>
    /// 合成测试图像生成器。
    /// 在大图上以精确的位置和角度贴入模板图，用于自动化验证模板匹配算法。
    /// </summary>
    public static class TestImageGenerator
    {
        /// <summary>
        /// 创建一张合成搜索图——将旋转后的模板贴在背景图的指定位置。
        /// </summary>
        /// <param name="backgroundWidth">背景图宽度。</param>
        /// <param name="backgroundHeight">背景图高度。</param>
        /// <param name="template">模板图像。</param>
        /// <param name="centerX">模板中心在背景图中的 X 坐标。</param>
        /// <param name="centerY">模板中心在背景图中的 Y 坐标。</param>
        /// <param name="angle">模板旋转角度（度）。</param>
        /// <param name="backgroundColor">背景灰度值。</param>
        /// <returns>合成后的搜索图像。</returns>
        public static Mat CreateSearchImage(
            int backgroundWidth, int backgroundHeight,
            Mat template,
            double centerX, double centerY,
            double angle = 0,
            byte backgroundColor = 128)
        {
            var background = new Mat(backgroundHeight, backgroundWidth,
                MatType.CV_8UC1, new Scalar(backgroundColor));

            var rotatedTemplate = RotateTemplate(template, angle);

            // 贴入位置（左上角）
            int pasteX = (int)(centerX - rotatedTemplate.Width / 2.0);
            int pasteY = (int)(centerY - rotatedTemplate.Height / 2.0);

            // 计算有效区域
            int srcStartX = Math.Max(0, -pasteX);
            int srcStartY = Math.Max(0, -pasteY);
            int dstStartX = Math.Max(0, pasteX);
            int dstStartY = Math.Max(0, pasteY);
            int copyW = Math.Min(rotatedTemplate.Width - srcStartX,
                backgroundWidth - dstStartX);
            int copyH = Math.Min(rotatedTemplate.Height - srcStartY,
                backgroundHeight - dstStartY);

            if (copyW > 0 && copyH > 0)
            {
                var srcRect = new Rect(srcStartX, srcStartY, copyW, copyH);
                var dstRect = new Rect(dstStartX, dstStartY, copyW, copyH);

                using var srcRoi = new Mat(rotatedTemplate, srcRect);
                var dstRoi = new Mat(background, dstRect);
                srcRoi.CopyTo(dstRoi);
            }

            rotatedTemplate.Dispose();
            return background;
        }

        /// <summary>
        /// 使用 CvTool.RotateImage 旋转模板。
        /// </summary>
        private static Mat RotateTemplate(Mat template, double angle)
        {
            return CvLibrary.OpenCV.CvTool.RotateImage(template, angle);
        }

        /// <summary>
        /// 创建包含多个目标实例的合成搜索图。
        /// </summary>
        /// <returns>搜索图和各实例的 ground truth。</returns>
        public static (Mat image, List<InstanceGroundTruth> truths) CreateMultiInstanceImage(
            int width, int height,
            Mat template,
            List<InstanceGroundTruth> instances,
            byte backgroundColor = 128)
        {
            var image = new Mat(height, width, MatType.CV_8UC1, new Scalar(backgroundColor));

            foreach (var inst in instances)
            {
                var rotated = RotateTemplate(template, inst.Angle);

                int pasteX = (int)(inst.CenterX - rotated.Width / 2.0);
                int pasteY = (int)(inst.CenterY - rotated.Height / 2.0);

                int srcStartX = Math.Max(0, -pasteX);
                int srcStartY = Math.Max(0, -pasteY);
                int dstStartX = Math.Max(0, pasteX);
                int dstStartY = Math.Max(0, pasteY);
                int copyW = Math.Min(rotated.Width - srcStartX, width - dstStartX);
                int copyH = Math.Min(rotated.Height - srcStartY, height - dstStartY);

                if (copyW > 0 && copyH > 0)
                {
                    using var srcRoi = new Mat(rotated, new Rect(srcStartX, srcStartY, copyW, copyH));
                    var dstRoi = new Mat(image, new Rect(dstStartX, dstStartY, copyW, copyH));
                    srcRoi.CopyTo(dstRoi);
                }

                rotated.Dispose();
            }

            return (image, instances);
        }

        /// <summary>
        /// 对图像添加高斯噪声。
        /// </summary>
        public static Mat AddGaussianNoise(Mat image, double sigma = 5.0)
        {
            var noise = new Mat(image.Size(), MatType.CV_8UC1);
            Cv2.Randn(noise, new Scalar(0), new Scalar(sigma));

            var result = new Mat();
            Cv2.Add(image, noise, result);
            noise.Dispose();
            return result;
        }

        /// <summary>
        /// 创建亮度反转的图像。
        /// </summary>
        public static Mat InvertBrightness(Mat image)
        {
            var inverted = new Mat();
            Cv2.BitwiseNot(image, inverted);
            return inverted;
        }
    }

    /// <summary>
    /// 合成图像中的目标实例 ground truth。
    /// </summary>
    public class InstanceGroundTruth
    {
        public double CenterX { get; init; }
        public double CenterY { get; init; }
        public double Angle { get; init; }

        public InstanceGroundTruth(double centerX, double centerY, double angle = 0)
        {
            CenterX = centerX;
            CenterY = centerY;
            Angle = angle;
        }
    }
}
