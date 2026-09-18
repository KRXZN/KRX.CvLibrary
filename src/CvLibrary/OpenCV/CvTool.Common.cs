using System.Diagnostics;
using CvCommon;
using OpenCvSharp;

namespace CvLibrary.OpenCV
{
    public static partial class CvTool
    {        
        public static (double, double) GetMinMaxGrayValue(Mat mat)
        {
            Cv2.MinMaxLoc(mat, out double min, out double max);
            return (min, max);
        }
     
        public static CvRegion SelecteMaxAreaRegion(CvRegion region)
        {
            // net472 迁移：无 Enumerable.MaxBy，改用 OrderByDescending 取最大
            var maxAreaProps = region.RegionProperties.OrderByDescending(p => p.Area).FirstOrDefault();
            Mat selected = new Mat();
            Cv2.Compare(region.Mask, maxAreaProps!.Label, selected, CmpType.EQ);
            region.Mask.MinMaxLoc(out double min, out double max);
            return new CvRegion(selected, maxAreaProps);
        }
    }
}
