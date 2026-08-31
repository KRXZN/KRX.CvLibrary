using CvCommon;

namespace CvLibrary.Extensions
{
    public static class CvRectExtension
    {
        public static CvRect ClampRectToImage(this CvRect rect, int imageWidth, int imageHeight)
        {
            double x = Math.Max(0, rect.X);
            double y = Math.Max(0, rect.Y);
            double right = Math.Min(rect.Right, imageWidth);
            double bottom = Math.Min(rect.Bottom, imageHeight);
            double w = Math.Max(0, right - x);
            double h = Math.Max(0, bottom - y);
            return new CvRect(x, y, w, h);
        }

    }
}
