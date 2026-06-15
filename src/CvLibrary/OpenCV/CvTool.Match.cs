using CvCommon;
using OpenCvSharp;

namespace CvLibrary.OpenCV
{
    public static partial class CvTool
    {
        public static Mat CreateMatchTemplate(Mat src, CvRect rect)
        {
            Mat template = CropImage(src, rect);
            if (template.Empty())
                throw new ArgumentException("Template image is empty.");
            if (template.Type() != MatType.CV_8UC1)
                template = ConvertToGray(template);
            return template;
        }

        public static IEnumerable<CvPoint> MatchTemplate(
            Mat src,
            Mat template,
            TemplateMatchModes mode = TemplateMatchModes.CCoeffNormed,
            double[]? rotateAngle = null,
            double threshold = 0.7,
            double nmsThreshold = 0.3
        )
        {
            if (src.Empty() || template.Empty())
                throw new ArgumentException("Source or template image is empty.");
            if (src.Type() != MatType.CV_8UC1)
                src = ConvertToGray(src);
            if (template.Type() != MatType.CV_8UC1)
                template = ConvertToGray(template);

            var allMatchResults = new List<(CvRect Box, float Score)>();

            Action<Mat, Mat> findMatches = (source, currentTemplate) =>
            {
                if (currentTemplate.Cols > source.Cols || currentTemplate.Rows > source.Rows)
                    return;

                int resultCols = source.Cols - currentTemplate.Cols + 1;
                int resultRows = source.Rows - currentTemplate.Rows + 1;
                if (resultRows <= 0 || resultCols <= 0)
                    return;

                using Mat result = new Mat(resultRows, resultCols, MatType.CV_32FC1);
                Cv2.MatchTemplate(source, currentTemplate, result, mode);

                for (int y = 0; y < result.Rows; y++)
                {
                    for (int x = 0; x < result.Cols; x++)
                    {
                        float score = result.At<float>(y, x);
                        if (score >= threshold)
                        {
                            allMatchResults.Add(
                                (
                                    new CvRect(x, y, currentTemplate.Width, currentTemplate.Height),
                                    score
                                )
                            );
                        }
                    }
                }
            };

            if (rotateAngle == null || rotateAngle.Length == 0)
            {
                findMatches(src, template);
            }
            else
            {
                foreach (double angle in rotateAngle)
                {
                    using Mat rotatedTemplate = RotateImage(template, angle);
                    findMatches(src, rotatedTemplate);
                }
            }

            // Apply Non-Maximum Suppression (NMS)
            var sortedMatches = allMatchResults.OrderByDescending(m => m.Score).ToList();
            var finalLocations = new List<CvPoint>();

            while (sortedMatches.Count > 0)
            {
                var topMatch = sortedMatches[0];
                finalLocations.Add(topMatch.Box.TopLeft);
                sortedMatches.RemoveAt(0);

                var remainingMatches = new List<(CvRect Box, float Score)>();
                foreach (var otherMatch in sortedMatches)
                {
                    var intersection = CvRect.Intersect(topMatch.Box, otherMatch.Box);
                    if (intersection.IsEmpty)
                    {
                        remainingMatches.Add(otherMatch);
                        continue;
                    }
                    double iou =
                        intersection.Width
                        * intersection.Height
                        / (
                            topMatch.Box.Width * topMatch.Box.Height
                            + otherMatch.Box.Width * otherMatch.Box.Height
                            - intersection.Width * intersection.Height
                        );
                    if (iou < nmsThreshold)
                    {
                        remainingMatches.Add(otherMatch);
                    }
                }
                sortedMatches = remainingMatches;
            }

            return finalLocations;
        }
    }
}
