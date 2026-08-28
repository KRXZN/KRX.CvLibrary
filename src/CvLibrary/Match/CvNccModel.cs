using System.Collections.Concurrent;
using CvCommon;
using OpenCvSharp;

namespace CvLibrary.OpenCV.Match
{
    /// <summary>
    /// NCC（归一化互相关）模板匹配模型。
    /// 仿 HALCON create_ncc_model / find_ncc_model API。
    /// </summary>
    public sealed class CvNccModel : CvMatchModel
    {
        #region Factory Methods

        /// <summary>
        /// 从已裁剪的模板图像创建 NCC 模型。
        /// </summary>
        /// <param name="template">模板图像（任意通道，自动转灰度）。</param>
        /// <param name="options">创建参数。</param>
        /// <param name="templateMask">模板掩码（CV_8UC1，与模板同尺寸；白色参与匹配，黑色忽略）。null = 全白（模板矩形内全有效）。</param>
        /// <returns>NCC 匹配模型。</returns>
        public static CvNccModel Create(Mat template, NccModelOptions? options = null, Mat? templateMask = null)
        {
            options ??= new NccModelOptions();
            ValidateTemplate(template);
            ValidateTemplateMask(template, templateMask);

            // 自动转灰度
            Mat grayTemplate;
            if (template.Type() != MatType.CV_8UC1)
            {
                grayTemplate = new Mat();
                Cv2.CvtColor(template, grayTemplate, ColorConversionCodes.BGR2GRAY);
            }
            else
            {
                grayTemplate = template.Clone();
            }

            // 可选平滑
            if (options.SmoothKernelSize.HasValue)
            {
                int ksize = options.SmoothKernelSize.Value;
                if (ksize < 3 || ksize % 2 == 0)
                    throw new ArgumentException("SmoothKernelSize must be odd and >= 3.");
                var smoothed = new Mat();
                Cv2.GaussianBlur(grayTemplate, smoothed, new Size(ksize, ksize), 0);
                grayTemplate.Dispose();
                grayTemplate = smoothed;
            }

            try
            {
                var model = new CvNccModel();
                model.Initialize(grayTemplate, options, templateMask);
                return model;
            }
            finally
            {
                grayTemplate.Dispose();
            }
        }

        /// <summary>
        /// 从源图像中按矩形区域裁剪模板并创建 NCC 模型（便捷重载）。
        /// </summary>
        public static CvNccModel Create(Mat sourceImage, CvRect templateRoi,
            NccModelOptions? options = null)
        {
            Mat cropped = CropTemplateFromSource(sourceImage, templateRoi);
            try
            {
                return Create(cropped, options);
            }
            finally
            {
                cropped.Dispose();
            }
        }

        #endregion

        #region Initialization

        private CvNccModel() { }

        private void Initialize(Mat grayTemplate, NccModelOptions options, Mat? templateMask)
        {
            TemplateSize = new CvSize(grayTemplate.Width, grayTemplate.Height);
            TemplateCenter = new Point2d(grayTemplate.Width / 2.0, grayTemplate.Height / 2.0);

            // 角度配置
            ModelAngleStart = options.AngleStart;
            ModelAngleExtent = options.AngleExtent;
            ModelAngleStep = options.AngleStep ?? 1.0;
            ModelMetric = options.Metric;

            // 金字塔层数
            NumLevels = options.NumLevels
                ?? CalculateAutoLevels(TemplateSize.Width, TemplateSize.Height);

            // mask：外部传入则克隆使用（白色参与匹配，黑色忽略），否则全白（模板矩形内全有效）
            var fullMask = templateMask != null
                ? templateMask.Clone()
                : new Mat(grayTemplate.Size(), MatType.CV_8UC1, Scalar.White);

            // 逐层构建预旋转模板
            var currentLevelTemplate = grayTemplate.Clone();
            var currentLevelMask = fullMask.Clone();

            Pyramid = new TemplateEntry[NumLevels][];
            LayerAngles = new double[NumLevels][];

            for (int level = 0; level < NumLevels; level++)
            {
                double[] angles = GenerateAnglesForLevel(
                    ModelAngleStart, ModelAngleExtent, ModelAngleStep, level);
                LayerAngles[level] = angles;

                // 生成 mask：先对原始 mask 做所有角度的旋转，然后降采样（对 level > 0）
                Mat levelMaskSource;
                if (level == 0)
                {
                    levelMaskSource = fullMask;
                }
                else
                {
                    levelMaskSource = DownsampleLayer(currentLevelMask);
                    currentLevelMask.Dispose();
                    currentLevelMask = levelMaskSource;
                }

                Pyramid[level] = new TemplateEntry[angles.Length];

                for (int a = 0; a < angles.Length; a++)
                {
                    // 旋转模板和 mask
                    var (rotated, rotatedMask) = RotateTemplateWithMask(
                        currentLevelTemplate, levelMaskSource, angles[a]);

                    Pyramid[level][a] = new TemplateEntry
                    {
                        Image = rotated,
                        Mask = rotatedMask,
                        CenterOffset = new Point2d(rotated.Width / 2.0, rotated.Height / 2.0),
                        Angle = angles[a],
                    };
                }

                // 非最后一层时降采样模板（L0 的灰度模板）
                if (level < NumLevels - 1)
                {
                    var nextLvl = DownsampleLayer(currentLevelTemplate);
                    currentLevelTemplate.Dispose();
                    currentLevelTemplate = nextLvl;
                }
            }

            currentLevelTemplate.Dispose();
            currentLevelMask.Dispose();
            fullMask.Dispose();
        }

        #endregion

        #region FindMatches

        /// <inheritdoc/>
        public override IReadOnlyList<CvMatchResult> FindMatches(
            Mat searchImage, FindMatchesOptions? options = null)
        {
            options ??= new FindMatchesOptions();
            ValidateFindMatches(searchImage, options);

            // Clamp 角度子范围
            double effectiveAngleStart = options.AngleStart ?? ModelAngleStart;
            double effectiveAngleExtent = options.AngleExtent ?? ModelAngleExtent;
            effectiveAngleStart = Math.Max(effectiveAngleStart, ModelAngleStart);
            effectiveAngleExtent = Math.Min(effectiveAngleExtent, ModelAngleExtent);

            // 处理搜索区域（物理裁剪 + padding）
            Mat actualSearchImage;
            double offsetX = 0, offsetY = 0;
            CvRect? effectiveRoi = null;

            if (options.SearchRegion.HasValue)
            {
                var roi = options.SearchRegion.Value;
                int padW = (int)TemplateSize.Width;
                int padH = (int)TemplateSize.Height;
                int cropX = Math.Max(0, (int)roi.X - padW);
                int cropY = Math.Max(0, (int)roi.Y - padH);
                int cropW = Math.Min(searchImage.Width - cropX, (int)roi.Width + 2 * padW);
                int cropH = Math.Min(searchImage.Height - cropY, (int)roi.Height + 2 * padH);

                actualSearchImage = CropFromSource(searchImage, cropX, cropY, cropW, cropH);
                offsetX = cropX;
                offsetY = cropY;
                effectiveRoi = new CvRect(
                    (int)roi.X - cropX,
                    (int)roi.Y - cropY,
                    (int)roi.Width,
                    (int)roi.Height);
            }
            else
            {
                actualSearchImage = searchImage.Clone();
                effectiveRoi = null;
            }

            try
            {
                // 转灰度
                Mat graySearch;
                if (actualSearchImage.Type() != MatType.CV_8UC1)
                {
                    graySearch = new Mat();
                    Cv2.CvtColor(actualSearchImage, graySearch, ColorConversionCodes.BGR2GRAY);
                }
                else
                {
                    graySearch = actualSearchImage.Clone();
                }

                try
                {
                    // 构建搜索图金字塔
                    var searchPyramid = BuildSearchPyramid(graySearch, NumLevels);

                    try
                    {
                        return SearchPyramidCascade(searchPyramid, options,
                            effectiveAngleStart, effectiveAngleExtent,
                            offsetX, offsetY, effectiveRoi);
                    }
                    finally
                    {
                        foreach (var m in searchPyramid) m.Dispose();
                    }
                }
                finally
                {
                    graySearch.Dispose();
                }
            }
            finally
            {
                actualSearchImage.Dispose();
            }
        }

        #endregion

        #region Pyramid Cascade Search

        private IReadOnlyList<CvMatchResult> SearchPyramidCascade(
            Mat[] searchPyramid, FindMatchesOptions options,
            double angleStart, double angleExtent,
            double offsetX, double offsetY, CvRect? roi)
        {
            int topLevel = NumLevels - 1;
            // For single-level: just what was asked.  For pyramid: keep a few
            // extra at coarse levels since some won't survive refinement.
            int maxCandidatesPerLevel = NumLevels == 1
                ? options.MaxMatches
                : Math.Min(20, options.MaxMatches + 3);

            // === 顶层：贪心抑制 ===
            var candidates = GreedySuppressionSearch(
                searchPyramid[topLevel], topLevel,
                angleStart, angleExtent,
                GetAdaptiveThreshold(options.MinScore, topLevel, topLevel),
                options.MaxOverlap, maxCandidatesPerLevel);

            if (candidates.Count == 0)
                return Array.Empty<CvMatchResult>();

            // === 逐层向下精炼 ===
            for (int level = topLevel - 1; level >= 0; level--)
            {
                double levelThreshold = GetAdaptiveThreshold(options.MinScore, level, topLevel);

                candidates = RefineCandidates(
                    searchPyramid[level], level,
                    candidates, angleStart, angleExtent,
                    levelThreshold, options.MaxOverlap, maxCandidatesPerLevel);

                if (candidates.Count == 0)
                    return Array.Empty<CvMatchResult>();
            }

            // === L0 亚像素精炼 ===
            var results = new List<CvMatchResult>(Math.Min(candidates.Count, options.MaxMatches));

            foreach (var candidate in candidates.OrderByDescending(c => c.Score))
            {
                Point2d refinedPos = candidate.Center;
                Point refinedTopLeft = candidate.TopLeft;

                if (options.SubPixelRefinement)
                {
                    // 亚像素位置精炼
                    var angleEntry = GetBestAngleEntry(0, candidate.Angle);
                    if (angleEntry == null) continue;
                    using var resultMat = MatchAtPosition(
                        searchPyramid[0], angleEntry, candidate.TopLeft);
                    refinedPos = RefineSubPixel(resultMat,
                        new Point((int)candidate.Center.X, (int)candidate.Center.Y));

                    // 用精炼后的位置更新 topLeft 供角度精炼使用
                    refinedTopLeft = new Point(
                        (int)(refinedPos.X - angleEntry.CenterOffset.X + 0.5),
                        (int)(refinedPos.Y - angleEntry.CenterOffset.Y + 0.5));
                }

                // 角度亚像素精炼（使用精炼后的位置）
                double refinedAngle = candidate.Angle;
                if (options.SubPixelRefinement && LayerAngles![0].Length >= 3)
                {
                    refinedAngle = RefineAngleAtLevel(
                        searchPyramid[0], refinedTopLeft, candidate.Angle, 0);
                }

                // 按 ROI 过滤（裁剪图坐标系：refinedPos 与 effectiveRoi 在同一坐标系）
                if (roi.HasValue)
                {
                    if (refinedPos.X < roi.Value.X || refinedPos.X > roi.Value.X + roi.Value.Width ||
                        refinedPos.Y < roi.Value.Y || refinedPos.Y > roi.Value.Y + roi.Value.Height)
                        continue;
                }

                // 转换到源图坐标
                double globalX = refinedPos.X + offsetX;
                double globalY = refinedPos.Y + offsetY;

                results.Add(new CvMatchResult
                {
                    Position = new CvPoint(globalX, globalY),
                    Angle = refinedAngle,
                    Score = candidate.Score,
                });
            }

            return results.Take(options.MaxMatches).ToList();
        }

        #endregion

        #region Top-Level Greedy Suppression

        private List<Candidate> GreedySuppressionSearch(
            Mat searchMat, int level,
            double angleStart, double angleExtent,
            double minScore, double maxOverlap, int maxCandidates)
        {
            var candidates = new List<Candidate>();
            var allowMask = new Mat(searchMat.Size(), MatType.CV_8UC1, Scalar.White);

            try
            {
                double[] angles = GetAnglesInRange(LayerAngles![level], angleStart, angleExtent);

                // Pre-filter: only consider angles whose template fits the search image.
                var validAngles = new List<(double Angle, TemplateEntry Entry)>();
                foreach (double angle in angles)
                {
                    var entry = GetAngleEntry(level, angle);
                    if (entry == null) continue;
                    if (entry.Image.Width > searchMat.Width ||
                        entry.Image.Height > searchMat.Height) continue;
                    validAngles.Add((angle, entry));
                }


                while (candidates.Count < maxCandidates)
                {
                    Candidate? bestInLoop;

                    if (candidates.Count == 0)
                    {
                        // === First iteration: process all angles in parallel ===
                        bestInLoop = ParallelScanAngles(searchMat, validAngles,
                            allowMask, minScore);
                    }
                    else
                    {
                        // === Subsequent iterations: sequential scan with allowMask ===
                        bestInLoop = SequentialScanAngles(searchMat, validAngles,
                            allowMask, minScore);
                    }

                    if (bestInLoop == null) break;
                    candidates.Add(bestInLoop);

                    var entryForMask = GetAngleEntry(level, bestInLoop.Angle)!;
                    Cv2.Rectangle(allowMask,
                        new Rect(bestInLoop.TopLeft.X, bestInLoop.TopLeft.Y,
                            entryForMask.Image.Width, entryForMask.Image.Height),
                        Scalar.Black, -1);
                }
            }
            finally
            {
                allowMask.Dispose();
            }

            return candidates;
        }

        /// <summary>
        /// Parallel angle scan for the first iteration (allowMask is all white).
        /// Each angle is processed independently — big speedup on multi-core CPUs.
        /// </summary>
        private Candidate? ParallelScanAngles(
            Mat searchMat,
            List<(double Angle, TemplateEntry Entry)> angles,
            Mat allowMask, double minScore)
        {
            // Disable OpenCV internal threading to avoid oversubscription.
            int origThreads = Cv2.GetNumThreads();
            Cv2.SetNumThreads(1);

            try
            {
                var bestLock = new object();
                Candidate? best = null;
                double bestScore = minScore - 1;

                Parallel.ForEach(angles, item =>
                {
                    var (angle, entry) = item;

                    using var allowSub = new Mat(allowMask,
                        new Rect(0, 0,
                            searchMat.Width - entry.Image.Width + 1,
                            searchMat.Height - entry.Image.Height + 1));
                    if (Cv2.CountNonZero(allowSub) == 0) return;

                    using var resultMat = new Mat();
                    Cv2.MatchTemplate(searchMat, entry.Image, resultMat,
                        TemplateMatchModes.CCoeffNormed, entry.Mask);

                    SanitizeResultMat(resultMat);

                    Cv2.MinMaxLoc(resultMat, out _, out double maxVal,
                        out _, out Point maxLoc, allowSub);

                    if (Math.Abs(maxVal) < 0.01f) return;

                    double score = NormalizeScore((float)maxVal);

                    if (score > minScore)
                    {
                        lock (bestLock)
                        {
                            if (score > bestScore)
                            {
                                bestScore = score;
                                best = new Candidate
                                {
                                    TopLeft = maxLoc,
                                    Center = new Point2d(
                                        maxLoc.X + entry.CenterOffset.X,
                                        maxLoc.Y + entry.CenterOffset.Y),
                                    Angle = angle,
                                    Score = score,
                                };
                            }
                        }
                    }
                });

                return best;
            }
            finally
            {
                Cv2.SetNumThreads(origThreads);
            }
        }

        /// <summary>
        /// Sequential angle scan for subsequent iterations (allowMask partially blocked).
        /// </summary>
        private Candidate? SequentialScanAngles(
            Mat searchMat,
            List<(double Angle, TemplateEntry Entry)> angles,
            Mat allowMask, double minScore)
        {
            Candidate? best = null;

            foreach (var (angle, entry) in angles)
            {
                int resultW = searchMat.Width - entry.Image.Width + 1;
                int resultH = searchMat.Height - entry.Image.Height + 1;
                if (resultW <= 0 || resultH <= 0) continue;

                using var allowSub = new Mat(allowMask,
                    new Rect(0, 0, resultW, resultH));
                if (Cv2.CountNonZero(allowSub) == 0) continue;

                using var resultMat = new Mat();
                Cv2.MatchTemplate(searchMat, entry.Image, resultMat,
                    TemplateMatchModes.CCoeffNormed, entry.Mask);

                SanitizeResultMat(resultMat);

                Cv2.MinMaxLoc(resultMat, out _, out double maxVal,
                    out _, out Point maxLoc, allowSub);

                if (Math.Abs(maxVal) < 0.01f) continue;

                double score = NormalizeScore((float)maxVal);

                if (score > minScore && (best == null || score > best.Score))
                {
                    best = new Candidate
                    {
                        TopLeft = maxLoc,
                        Center = new Point2d(
                            maxLoc.X + entry.CenterOffset.X,
                            maxLoc.Y + entry.CenterOffset.Y),
                        Angle = angle,
                        Score = score,
                    };
                }
            }

            return best;
        }

        private static bool FindBestInMask(
            Mat resultMat, Mat? allowMask, TemplateEntry entry,
            out double bestScore, out Point bestTopLeft, out Point2d bestCenter)
        {
            SanitizeResultMat(resultMat);

            // Prepare allow-mask sub-region (same size as resultMat).  null = no restriction.
            Mat? allowSub = null;
            if (allowMask != null)
            {
                allowSub = new Mat(allowMask,
                    new Rect(0, 0, resultMat.Cols, resultMat.Rows));
            }

            try
            {
                // === Use native MinMaxLoc (much faster than C# double loop) ===
                Cv2.MinMaxLoc(resultMat, out _, out double maxVal,
                    out _, out Point maxLoc, allowSub!);

                // Validate: reject near-zero NCC (degenerate flat-region matches).
                if (Math.Abs(maxVal) < 0.01f)
                {
                    bestScore = 0;
                    bestTopLeft = default;
                    bestCenter = default;
                    return false;
                }

                bestScore = maxVal;
                bestTopLeft = maxLoc;
                bestCenter = new Point2d(
                    bestTopLeft.X + entry.CenterOffset.X,
                    bestTopLeft.Y + entry.CenterOffset.Y);

                return true;
            }
            finally
            {
                allowSub?.Dispose();
            }
        }

        #endregion

        #region Layer Refinement

        private List<Candidate> RefineCandidates(
            Mat searchMat, int level,
            List<Candidate> previousCandidates,
            double angleStart, double angleExtent,
            double minScore, double maxOverlap, int maxCandidates)
        {
            double[] angles = GetAnglesInRange(LayerAngles![level], angleStart, angleExtent);

            var refinedCandidates = new List<(Candidate Candidate, double Score)>();

            foreach (var prev in previousCandidates)
            {
                // 投影 top-left（不是 center！）到当前层。
                // 注意：搜索窗口 Rect 定义的是模板左上角的放置范围，
                // 必须基于 topLeft 偏移，不是 center。
                double projectedTLX = prev.TopLeft.X * 2.0;
                double projectedTLY = prev.TopLeft.Y * 2.0;

                // 搜索窗口：上层投影有 1-2 像素偏差（金字塔缩放累积），需充足余量。
                int searchMargin = level >= 1 ? 6 : 4;
                int searchStartX = Math.Max(0, (int)projectedTLX - searchMargin);
                int searchStartY = Math.Max(0, (int)projectedTLY - searchMargin);

                // 每个候选在局部角度范围内搜索
                double angleStep = LayerAngles![level].Length > 1
                    ? LayerAngles[level][1] - LayerAngles[level][0] : 0;
                // Wider angle search at coarser levels where angle estimate is less precise.
                int angleHalfSpan = level >= 1 ? 3 : 2;
                double localAngleStart = prev.Angle - angleStep * angleHalfSpan;
                double localAngleExtent = angleStep * angleHalfSpan * 2;
                double localAngleEnd = Math.Min(localAngleStart + localAngleExtent + angleStep,
                    angleStart + angleExtent);

                foreach (double angle in angles)
                {
                    if (angle < localAngleStart || angle > localAngleEnd) continue;

                    var entry = GetAngleEntry(level, angle);
                    if (entry == null) continue;

                    // 裁剪局部搜索区域
                    int pad = searchMargin * 2;
                    int subW = Math.Min(searchMat.Cols - searchStartX, entry.Image.Width + pad);
                    int subH = Math.Min(searchMat.Rows - searchStartY, entry.Image.Height + pad);

                    if (subW <= entry.Image.Width || subH <= entry.Image.Height)
                        continue;

                    var subSearch = new Mat(searchMat,
                        new Rect(searchStartX, searchStartY, subW, subH));

                    using var resultMat = new Mat();
                    Cv2.MatchTemplate(subSearch, entry.Image, resultMat,
                        TemplateMatchModes.CCoeffNormed, entry.Mask);

                    // Sanitize non-finite pixels then use native MinMaxLoc.
                    SanitizeResultMat(resultMat);

                    Cv2.MinMaxLoc(resultMat, out _, out double bestVal,
                        out _, out Point bestLoc);

                    // Reject near-zero NCC (degenerate flat-region matches).
                    if (Math.Abs(bestVal) < 0.01f)
                        continue;

                    double score = NormalizeScore((float)bestVal);
                    if (score >= minScore)
                    {
                        int globalX = searchStartX + bestLoc.X;
                        int globalY = searchStartY + bestLoc.Y;
                        refinedCandidates.Add((new Candidate
                        {
                            TopLeft = new Point(globalX, globalY),
                            Center = new Point2d(
                                globalX + entry.CenterOffset.X,
                                globalY + entry.CenterOffset.Y),
                            Angle = angle,
                            Score = score,
                        }, score));
                    }
                }
            }

            // NMS + 合并重叠
            var withBoxes = refinedCandidates
                .Select(c =>
                {
                    var entry = GetAngleEntry(level, c.Candidate.Angle);
                    int w = entry?.Image.Width ?? 1;
                    int h = entry?.Image.Height ?? 1;
                    var box = new CvRect(c.Candidate.TopLeft.X, c.Candidate.TopLeft.Y, w, h);
                    return (Box: box, c.Score, c.Candidate);
                }).ToList();

            // NMS with index tracking for reliable matching
            var nmsInput = withBoxes
                .Select((item, idx) => (item.Box, item.Score, Index: idx))
                .ToList();
            var nmsBoxesOnly = nmsInput
                .Select(t => (t.Box, t.Score))
                .ToList();
            var keptBoxes = ApplyNMS(nmsBoxesOnly, maxOverlap);

            // Reliable matching: use index from original list
            var keptIndices = new HashSet<int>();
            foreach (var kept in keptBoxes)
            {
                var match = nmsInput.FirstOrDefault(t =>
                    Math.Abs(t.Score - kept.Score) < 0.001
                    && t.Box.X == kept.Box.X && t.Box.Y == kept.Box.Y);
                if (match != default)
                    keptIndices.Add(match.Index);
            }

            return keptIndices
                .Select(i => withBoxes[i].Candidate)
                .OrderByDescending(c => c.Score)
                .Take(maxCandidates)
                .ToList();
        }

        #endregion

        #region Helpers

        /// <summary>
        /// 校验模板掩码：null 合法（等价全白）；非 null 时必须非空、CV_8UC1、与模板同尺寸。
        /// </summary>
        private static void ValidateTemplateMask(Mat template, Mat? templateMask)
        {
            if (templateMask is null)
                return;

            if (templateMask.Empty())
                throw new ArgumentException("Template mask is empty.");

            if (templateMask.Type() != MatType.CV_8UC1)
                throw new ArgumentException(
                    $"Template mask must be CV_8UC1, got {templateMask.Type()}.");

            if (templateMask.Width != template.Width || templateMask.Height != template.Height)
                throw new ArgumentException(
                    $"Template mask size ({templateMask.Width}×{templateMask.Height}) "
                    + $"must match template ({template.Width}×{template.Height}).");
        }

        /// <summary>
        /// In-place sanitization: replaces NaN, +Inf, and -Inf with 0.
        /// MatchTemplate with mask can produce non-finite values on zero-variance
        /// regions (e.g., black borders from rotated images).  Clamping is NOT used
        /// because +Inf clamped to 1.0 would create spurious "perfect" matches.
        /// </summary>
        private static void SanitizeResultMat(Mat resultMat)
        {
            // 1) Mask pixels that are NaN (NaN != NaN, so EQ produces 0 for them).
            using var nanMask = new Mat();
            Cv2.Compare(resultMat, resultMat, nanMask, CmpType.EQ);
            // nanMask: 0 = NaN, 255 = finite or Inf
            Cv2.BitwiseNot(nanMask, nanMask);
            // nanMask: 255 = NaN, 0 = everything else

            // 2) Mask pixels that are > 1.0 (i.e., +Infinity — valid NCC is in [-1,1]).
            using var infMask = new Mat();
            Cv2.Compare(resultMat, new Scalar(1.0), infMask, CmpType.GT);
            // infMask: 255 = >1.0 (+Inf), 0 = others (NaN also yields 0)

            // 3) Mask pixels that are < -1.0 (i.e., -Infinity).
            using var negInfMask = new Mat();
            Cv2.Compare(resultMat, new Scalar(-1.0), negInfMask, CmpType.LT);
            // negInfMask: 255 = <-1.0 (-Inf), 0 = others

            // 4) Combine: non-finite = NaN OR +Inf OR -Inf → set all to 0.
            Cv2.BitwiseOr(nanMask, infMask, nanMask);
            Cv2.BitwiseOr(nanMask, negInfMask, nanMask);
            resultMat.SetTo(new Scalar(0), nanMask);
        }

        private double[] GetAnglesInRange(double[] allAngles, double start, double extent)
        {
            double end = start + extent;
            return allAngles
                .Where(a => a >= start - 0.01 && a <= end + 0.01)
                .ToArray();
        }

        private TemplateEntry? GetAngleEntry(int level, double angle)
        {
            var angles = LayerAngles![level];
            var entries = Pyramid![level];
            for (int i = 0; i < angles.Length; i++)
            {
                if (Math.Abs(angles[i] - angle) < 0.001)
                    return entries[i];
            }
            return null;
        }

        private TemplateEntry? GetBestAngleEntry(int level, double angle)
        {
            var angles = LayerAngles![level];
            var entries = Pyramid![level];
            TemplateEntry? best = null;
            double bestDiff = double.MaxValue;
            for (int i = 0; i < angles.Length; i++)
            {
                double diff = Math.Abs(angles[i] - angle);
                if (diff < bestDiff) { bestDiff = diff; best = entries[i]; }
            }
            return best;
        }

        private Mat MatchAtPosition(Mat searchMat, TemplateEntry entry, Point topLeft, int pad = 8)
        {
            int subW = Math.Min(searchMat.Cols - topLeft.X, entry.Image.Width + pad);
            int subH = Math.Min(searchMat.Rows - topLeft.Y, entry.Image.Height + pad);

            // 确保子图不小于模板
            if (subW < entry.Image.Width || subH < entry.Image.Height)
                subW = subH = 0; // 触发下方返回空 Mat

            if (subW <= 0 || subH <= 0)
            {
                // 无法创建有效的子搜索区域，返回整个搜索图上的匹配结果
                if (searchMat.Cols >= entry.Image.Width && searchMat.Rows >= entry.Image.Height)
                {
                    var fallback = new Mat();
                    Cv2.MatchTemplate(searchMat, entry.Image, fallback,
                        TemplateMatchModes.CCoeffNormed, entry.Mask);
                    return fallback;
                }
                // 搜索图比模板还小，返回 1×1 空结果
                return new Mat(1, 1, MatType.CV_32FC1, new Scalar(0));
            }

            var subSearch = new Mat(searchMat,
                new Rect(topLeft.X, topLeft.Y, subW, subH));

            var result = new Mat();
            Cv2.MatchTemplate(subSearch, entry.Image, result,
                TemplateMatchModes.CCoeffNormed, entry.Mask);

            return result;
        }

        private double RefineAngleAtLevel(
            Mat searchMat, Point topLeft, double bestAngle, int level)
        {
            var angles = LayerAngles![level];
            int bestIdx = -1;
            double bestDiff = double.MaxValue;
            for (int i = 0; i < angles.Length; i++)
            {
                double diff = Math.Abs(angles[i] - bestAngle);
                if (diff < bestDiff) { bestDiff = diff; bestIdx = i; }
            }

            if (bestIdx <= 0 || bestIdx >= angles.Length - 1)
                return bestAngle;

            var entry0 = Pyramid![level][bestIdx - 1];
            var entry1 = Pyramid![level][bestIdx];
            var entry2 = Pyramid![level][bestIdx + 1];

            using var r0 = MatchAtPosition(searchMat, entry0, topLeft);
            using var r1 = MatchAtPosition(searchMat, entry1, topLeft);
            using var r2 = MatchAtPosition(searchMat, entry2, topLeft);

            int px = Math.Min(3, r1.Cols - 1);
            int py = Math.Min(3, r1.Rows - 1);

            return RefineAngle(new[] { r0, r1, r2 },
                new[] { angles[bestIdx - 1], angles[bestIdx], angles[bestIdx + 1] },
                new Point(px, py));
        }

        #endregion

        #region Candidate Helper Struct

        private sealed class Candidate
        {
            public Point TopLeft;
            public Point2d Center;
            public double Angle;
            public double Score;
        }

        #endregion

        #region Image Utilities

        private static Mat CropFromSource(Mat src, int x, int y, int w, int h)
        {
            return new Mat(src, new Rect(x, y, w, h)).Clone();
        }

        private static Mat CropTemplateFromSource(Mat src, CvRect rect)
        {
            return CropFromSource(src,
                (int)rect.X, (int)rect.Y, (int)rect.Width, (int)rect.Height);
        }

        #endregion
    }
}
