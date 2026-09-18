using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace CvLibraryExtensions.OpenCV
{
    public static class DrawTextExtensions
    {
        public static void DrawWatermark(this Mat image, string context)
        {
            if (image is null) throw new ArgumentNullException(nameof(image));

            if (image.Empty())
            {
                throw new ArgumentException("The image is empty.", nameof(image));
            }
            if (string.IsNullOrEmpty(context))
            {
                throw new ArgumentException(
                    "The context cannot be null or empty.",
                    nameof(context)
                );
            }
            // 如果内容为非ASCII字符，则抛出异常
            if (context.Any(c => c > 127))
            {
                throw new ArgumentException(
                    "the context contains non-ASCII characters, which are not supported.",
                    nameof(context)
                );
            }

            var font = OpenCvSharp.HersheyFonts.HersheySimplex;
            var fontScale = Math.Max(0.6, image.Width / 1800.0);
            var thickness = Math.Max(1, (int)Math.Round(fontScale * 2));
            var margin = Math.Max(12, image.Width / 80);

            var textSize = OpenCvSharp.Cv2.GetTextSize(
                context,
                font,
                fontScale,
                thickness,
                out var baseline
            );
            var x = Math.Max(margin, image.Width - textSize.Width - margin);
            var y = Math.Max(textSize.Height + margin, image.Height - margin);

            var bgTopLeft = new OpenCvSharp.Point(
                Math.Max(0, x - 8),
                Math.Max(0, y - textSize.Height - 8)
            );
            var bgBottomRight = new OpenCvSharp.Point(
                Math.Min(image.Width - 1, x + textSize.Width + 8),
                Math.Min(image.Height - 1, y + baseline + 8)
            );

            OpenCvSharp.Cv2.Rectangle(
                image,
                bgTopLeft,
                bgBottomRight,
                new OpenCvSharp.Scalar(0, 0, 0),
                -1
            );
            OpenCvSharp.Cv2.PutText(
                image,
                context,
                new OpenCvSharp.Point(x, y),
                font,
                fontScale,
                new OpenCvSharp.Scalar(255, 255, 255),
                thickness,
                OpenCvSharp.LineTypes.AntiAlias
            );
        }
    }
}
