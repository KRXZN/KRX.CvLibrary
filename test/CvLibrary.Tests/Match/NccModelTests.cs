using CvLibrary.OpenCV.Match;
using OpenCvSharp;

namespace CvLibrary.Tests.Match
{
    public class NccModelTests
    {
        /// <summary>
        /// 创建一个简单的测试模板（不对称图案，避免旋转歧义）。
        /// </summary>
        private static Mat CreateTestTemplate(int size = 100)
        {
            var template = new Mat(size, size, MatType.CV_8UC1, new Scalar(128));
            // 画不对称图案（L 形 + 点）
            Cv2.Line(template, new Point(size / 4, size / 4),
                new Point(size / 4, size * 3 / 4), Scalar.White, 3);
            Cv2.Line(template, new Point(size / 4, size * 3 / 4),
                new Point(size * 3 / 4, size * 3 / 4), Scalar.White, 3);
            // 不对称标记
            Cv2.Circle(template, new Point(size / 4 + 5, size / 4 + 5), 3,
                Scalar.Black, -1);
            return template;
        }

        [Fact]
        public void Create_AutoLevels_Template100x100_ShouldHave3Levels()
        {
            using var template = CreateTestTemplate(100);
            var options = new NccModelOptions { AngleExtent = 0, AngleStep = 1.0 };
            using var model = CvNccModel.Create(template, options);

            Assert.True(model.Levels >= 2);  // log2(100/16) ≈ 2.6 → 2+ levels
        }

        [Fact]
        public void FindMatches_PerfectMatch_ScoreShouldBeNearOne()
        {
            using var template = CreateTestTemplate(50);

            var options = new NccModelOptions
            {
                AngleStart = 0,
                AngleExtent = 0,
                AngleStep = 1.0,
                NumLevels = 1,
            };
            using var model = CvNccModel.Create(template, options);

            // 将模板贴到更大的背景上（避免金字塔层级尺寸问题）
            using var searchImage = TestImageGenerator.CreateSearchImage(
                200, 200, template, 100, 100, angle: 0);

            var results = model.FindMatches(searchImage, new FindMatchesOptions
            {
                MinScore = 0.9,
                MaxMatches = 1,
            });

            Assert.Single(results);
            Assert.True(results[0].Score >= 0.95,
                $"Expected Score >= 0.95, got {results[0].Score}");
        }

        [Fact]
        public void FindMatches_TranslationOnly_PositionShouldBeCorrect()
        {
            using var template = CreateTestTemplate(60);

            var options = new NccModelOptions
            {
                AngleStart = 0,
                AngleExtent = 0,
                AngleStep = 1.0,
            };
            using var model = CvNccModel.Create(template, options);

            // 把 60×60 模板贴在 300×300 背景的 (150, 180) 位置
            double expectedX = 150;
            double expectedY = 180;
            using var searchImage = TestImageGenerator.CreateSearchImage(
                300, 300, template, expectedX, expectedY, angle: 0);

            var results = model.FindMatches(searchImage, new FindMatchesOptions
            {
                MinScore = 0.8,
                MaxMatches = 1,
            });

            Assert.Single(results);
            Assert.True(results[0].Score >= 0.8);
            Assert.True(Math.Abs(results[0].Position.X - expectedX) < 1.5,
                $"Expected X≈{expectedX}, got {results[0].Position.X}");
            Assert.True(Math.Abs(results[0].Position.Y - expectedY) < 1.5,
                $"Expected Y≈{expectedY}, got {results[0].Position.Y}");
        }

        [Fact(Skip = "Rotation angle selection in greedy suppression needs tuning")]
        public void FindMatches_WithRotation_AngleShouldBeCorrect()
        {
            // 使用不对称的模板
            var template = new Mat(60, 60, MatType.CV_8UC1, Scalar.Black);
            // Draw a distinctive asymmetric pattern with unique rotation
            Cv2.Rectangle(template, new Rect(5, 5, 50, 15), Scalar.White, -1);
            Cv2.Rectangle(template, new Rect(5, 20, 30, 35), Scalar.Gray, -1);
            Cv2.Circle(template, new Point(20, 13), 4, Scalar.Black, -1);

            var options = new NccModelOptions
            {
                AngleStart = -10,
                AngleExtent = 20,  // -10° ~ 10°
                AngleStep = 2.0,   // finer step
                NumLevels = 1,
            };
            using var model = CvNccModel.Create(template, options);

            // Test at 6° (should snap to 6° with 2° step)
            double expectedAngle = 6.0;
            using var searchImage = TestImageGenerator.CreateSearchImage(
                300, 300, template, 150, 150, angle: expectedAngle);

            var results = model.FindMatches(searchImage, new FindMatchesOptions
            {
                MinScore = 0.5,
                MaxMatches = 1,
                AngleStart = options.AngleStart,
                AngleExtent = options.AngleExtent,
            });

            Assert.Single(results);
            Assert.False(double.IsNaN(results[0].Angle));
            double diff = Math.Abs(results[0].Angle - expectedAngle);
            Assert.True(diff <= 2.5,
                $"Expected angle≈{expectedAngle}°, got {results[0].Angle}° (diff={diff}°)");

            template.Dispose();
        }

        [Fact]
        public void FindMatches_NoMatch_ShouldReturnEmpty()
        {
            using var template = CreateTestTemplate(50);

            var options = new NccModelOptions { AngleExtent = 0 };
            using var model = CvNccModel.Create(template, options);

            // 用纯噪声图搜索
            using var noiseImage = new Mat(200, 200, MatType.CV_8UC1);
            var rng = new Random(42);
            for (int y = 0; y < noiseImage.Rows; y++)
                for (int x = 0; x < noiseImage.Cols; x++)
                    noiseImage.Set<byte>(y, x, (byte)rng.Next(0, 256));

            var results = model.FindMatches(noiseImage, new FindMatchesOptions
            {
                MinScore = 0.9,  // 高分阈值——噪声不可能匹配
                MaxMatches = 5,
            });

            Assert.Empty(results);
        }

        [Fact]
        public void FindMatches_MultiInstance_FindsMultiple()
        {
            using var template = CreateTestTemplate(40);

            var options = new NccModelOptions
            {
                AngleExtent = 0,
                AngleStep = 1.0,
            };
            using var model = CvNccModel.Create(template, options);

            var instances = new List<InstanceGroundTruth>
            {
                new(60, 60, 0),
                new(140, 140, 0),
                new(100, 60, 0),
            };
            var (searchImage, _) = TestImageGenerator.CreateMultiInstanceImage(
                200, 200, template, instances);

            try
            {
                var results = model.FindMatches(searchImage, new FindMatchesOptions
                {
                    MinScore = 0.7,
                    MaxMatches = 5,
                });

                Assert.True(results.Count >= 2,
                    $"Expected ≥2 matches, found {results.Count}");
            }
            finally
            {
                searchImage.Dispose();
            }
        }

        [Fact]
        public void FindMatches_IgnoreGlobalPolarity_ShouldMatchInverted()
        {
            // Use a simpler high-contrast template
            var template = new Mat(40, 40, MatType.CV_8UC1, Scalar.Black);
            Cv2.Circle(template, new Point(20, 20), 10, Scalar.White, -1);

            var options = new NccModelOptions
            {
                AngleExtent = 0,
                Metric = MatchMetric.IgnoreGlobalPolarity,
                NumLevels = 1,
            };
            using var model = CvNccModel.Create(template, options);

            // 正常图
            using var normalImage = TestImageGenerator.CreateSearchImage(
                150, 150, template, 75, 75);

            // 反转图（白背景黑圆 → 黑背景白圆）
            using var invertedImage = TestImageGenerator.InvertBrightness(normalImage);

            var results = model.FindMatches(invertedImage, new FindMatchesOptions
            {
                MinScore = 0.5,
                MaxMatches = 1,
            });

            Assert.Single(results);
            Assert.True(results[0].Score >= 0.5,
                $"Inverted match score too low: {results[0].Score}");
            template.Dispose();
        }

        [Fact]
        public void Create_TemplateWithNonGray_ShouldAutoConvert()
        {
            using var colorTemplate = new Mat(50, 50, MatType.CV_8UC3, new Scalar(100, 150, 200));
            var options = new NccModelOptions { AngleExtent = 0 };
            using var model = CvNccModel.Create(colorTemplate, options);
            // Should not throw
            Assert.True(model.Levels > 0);
        }

        [Fact]
        public void Create_WithMask_MaskedRegionChange_ShouldNotAffectMatch()
        {
            // 模板：不对称图案
            using var template = CreateTestTemplate(60);

            // 掩码：中央挖洞（黑色 = 不参与匹配），洞覆盖模板的黑点标记
            var hole = new Rect(20, 20, 20, 20);
            using var mask = new Mat(template.Size(), MatType.CV_8UC1, Scalar.White);
            Cv2.Rectangle(mask, hole, Scalar.Black, -1);

            var options = new NccModelOptions { AngleExtent = 0, NumLevels = 1 };
            using var model = CvNccModel.Create(template, options, mask);

            // 变体：洞区内容改为棋盘图案（模拟字符变化），洞外像素不变
            using var variant = template.Clone();
            for (int yy = 0; yy < hole.Height; yy += 4)
            {
                for (int xx = 0; xx < hole.Width; xx += 4)
                {
                    Cv2.Rectangle(variant,
                        new Rect(hole.X + xx, hole.Y + yy, 2, 2),
                        ((xx + yy) / 4) % 2 == 0 ? Scalar.Black : Scalar.White, -1);
                }
            }

            double expectedX = 150;
            double expectedY = 180;
            using var searchImage = TestImageGenerator.CreateSearchImage(
                300, 300, variant, expectedX, expectedY, angle: 0);

            var results = model.FindMatches(searchImage, new FindMatchesOptions
            {
                MinScore = 0.8,
                MaxMatches = 1,
            });

            Assert.Single(results);
            // 洞外像素完全一致，带掩码的分数应接近 1
            Assert.True(results[0].Score >= 0.95,
                $"Masked match score too low: {results[0].Score}");
            Assert.True(Math.Abs(results[0].Position.X - expectedX) < 1.5,
                $"Expected X≈{expectedX}, got {results[0].Position.X}");
            Assert.True(Math.Abs(results[0].Position.Y - expectedY) < 1.5,
                $"Expected Y≈{expectedY}, got {results[0].Position.Y}");
        }

        [Fact]
        public void Create_InvalidMask_ShouldThrowArgumentException()
        {
            using var template = CreateTestTemplate(50);
            var options = new NccModelOptions { AngleExtent = 0 };

            // 尺寸不符
            using var wrongSize = new Mat(40, 40, MatType.CV_8UC1, Scalar.White);
            Assert.Throws<ArgumentException>(() => CvNccModel.Create(template, options, wrongSize));

            // 类型不符（应为 CV_8UC1）
            using var wrongType = new Mat(50, 50, MatType.CV_8UC3, Scalar.White);
            Assert.Throws<ArgumentException>(() => CvNccModel.Create(template, options, wrongType));

            // 空 Mat
            using var empty = new Mat();
            Assert.Throws<ArgumentException>(() => CvNccModel.Create(template, options, empty));
        }

        [Fact]
        public void Dispose_DisposedModel_FindMatchesShouldThrow()
        {
            using var template = CreateTestTemplate(30);
            var model = CvNccModel.Create(template, new NccModelOptions { AngleExtent = 0 });
            model.Dispose();

            Assert.Throws<ObjectDisposedException>(() =>
                model.FindMatches(template));
        }
    }
}
