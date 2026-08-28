using CvCommon;
using CvLibrary.OpenCV;
using OpenCvSharp;

namespace CvLibrary.Tests
{
    /// <summary>
    /// MapRotatedRectToSource 与 RotateImage 的一致性验证：
    /// 在源图已知位置画白色矩形 -> RotateImage 旋转 -> 在旋转图中定位白矩形 ->
    /// MapRotatedRectToSource 映射回源坐标系 -> 与原矩形比对。
    /// </summary>
    public class CvToolImageTests
    {
        [Theory]
        [InlineData(0)]
        [InlineData(90)]
        [InlineData(180)]
        [InlineData(270)]
        [InlineData(30)]
        [InlineData(-45)]
        public void MapRotatedRectToSource_RotateRoundTrip_ShouldRecoverSourceRect(double angle)
        {
            const int W = 120;
            const int H = 80;
            var srcRect = new Rect(30, 20, 40, 24);

            using var src = new Mat(H, W, MatType.CV_8UC1, Scalar.Black);
            Cv2.Rectangle(src, srcRect, Scalar.White, -1);

            using var rotated = CvTool.RotateImage(src, angle);

            // 在旋转图中定位白色矩形（阈值 200 避开 warpAffine 的灰色 128 边界填充）
            using var bin = new Mat();
            Cv2.Threshold(rotated, bin, 200, 255, ThresholdTypes.Binary);
            using var idx = new Mat();
            Cv2.FindNonZero(bin, idx);
            var rRect = Cv2.BoundingRect(idx);

            var mapped = CvTool.MapRotatedRectToSource(
                new CvRect(rRect.X, rRect.Y, rRect.Width, rRect.Height),
                angle,
                W,
                H
            );

            bool isRightAngle = Math.Abs(angle % 90) < 1e-9;
            if (isRightAngle)
            {
                // 90° 整数倍：精确映射，仅允许像素取整误差
                Assert.True(
                    Math.Abs(mapped.X - srcRect.X) <= 1.5
                        && Math.Abs(mapped.Y - srcRect.Y) <= 1.5
                        && Math.Abs(mapped.Width - srcRect.Width) <= 3
                        && Math.Abs(mapped.Height - srcRect.Height) <= 3,
                    $"angle={angle}: mapped=({mapped.X},{mapped.Y},{mapped.Width},{mapped.Height}), "
                        + $"src=({srcRect.X},{srcRect.Y},{srcRect.Width},{srcRect.Height})"
                );
            }
            else
            {
                // 任意角度：白矩形在旋转图中是倾斜四边形，经验定位得到其包围盒；
                // 映射回源坐标后应包含源矩形，且尺寸不超过其对角线长度
                const double tol = 2.0;
                Assert.True(
                    mapped.X <= srcRect.X + tol
                        && mapped.Y <= srcRect.Y + tol
                        && mapped.Right >= srcRect.Right - tol
                        && mapped.Bottom >= srcRect.Bottom - tol,
                    $"angle={angle}: mapped=({mapped.X},{mapped.Y},{mapped.Width},{mapped.Height}) "
                        + $"should contain src=({srcRect.X},{srcRect.Y},{srcRect.Width},{srcRect.Height})"
                );
                // 白矩形在旋转图中是倾斜四边形，经验定位得到其包围盒；包围盒 4 角
                // 逆映射回去的外扩上限为 srcW+srcH（45° 时取到），大于对角线
                double slackBound = srcRect.Width + srcRect.Height + tol;
                Assert.True(
                    mapped.Width <= slackBound && mapped.Height <= slackBound,
                    $"angle={angle}: mapped size ({mapped.Width}x{mapped.Height}) "
                        + $"exceeds bound {slackBound}"
                );
            }
        }
        [Theory]
        [InlineData(0)]
        [InlineData(90)]
        [InlineData(180)]
        [InlineData(270)]
        [InlineData(30)]
        [InlineData(-45)]
        public void MapSourceRectToRotated_RoundTripWithInverse_ShouldRecoverSourceRect(double angle)
        {
            const int W = 120;
            const int H = 80;
            var srcRect = new CvRect(30, 20, 40, 24);

            var rotatedRect = CvTool.MapSourceRectToRotated(srcRect, angle, W, H);
            var roundTrip = CvTool.MapRotatedRectToSource(rotatedRect, angle, W, H);

            bool isRightAngle = Math.Abs(angle % 90) < 1e-9;
            if (isRightAngle)
            {
                // 90° 整数倍：正逆映射互为精确逆变换（CvRect 保持 double 精度）
                Assert.True(
                    Math.Abs(roundTrip.X - srcRect.X) < 1e-9
                        && Math.Abs(roundTrip.Y - srcRect.Y) < 1e-9
                        && Math.Abs(roundTrip.Width - srcRect.Width) < 1e-9
                        && Math.Abs(roundTrip.Height - srcRect.Height) < 1e-9,
                    $"angle={angle}: roundTrip=({roundTrip.X},{roundTrip.Y},{roundTrip.Width},{roundTrip.Height}), "
                        + $"src=({srcRect.X},{srcRect.Y},{srcRect.Width},{srcRect.Height})"
                );
            }
            else
            {
                // 任意角度：两次包围盒外扩，应包含源矩形且有界
                const double tol = 2.0;
                Assert.True(
                    roundTrip.X <= srcRect.X + tol
                        && roundTrip.Y <= srcRect.Y + tol
                        && roundTrip.Right >= srcRect.Right - tol
                        && roundTrip.Bottom >= srcRect.Bottom - tol,
                    $"angle={angle}: roundTrip=({roundTrip.X},{roundTrip.Y},{roundTrip.Width},{roundTrip.Height}) "
                        + $"should contain src=({srcRect.X},{srcRect.Y},{srcRect.Width},{srcRect.Height})"
                );
                double slackBound = srcRect.Width + srcRect.Height + tol;
                Assert.True(
                    roundTrip.Width <= slackBound && roundTrip.Height <= slackBound,
                    $"angle={angle}: roundTrip size ({roundTrip.Width}x{roundTrip.Height}) "
                        + $"exceeds bound {slackBound}"
                );
            }
        }
    }
}
