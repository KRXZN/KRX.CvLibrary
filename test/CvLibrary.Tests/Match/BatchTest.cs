using System.Diagnostics;
using CvCommon;
using CvLibrary.OpenCV;
using CvLibrary.OpenCV.Match;
using OpenCvSharp;

namespace CvLibrary.Tests.Match;

/// <summary>
/// 批量测试：所有 PCB 数据集 + 大角度旋转扩展。
/// 验证自动金字塔 + 灰色填充方案的精度和性能，输出可视化结果图。
/// </summary>
public class BatchTest
{
    private static readonly Scalar DetectedColor = new(0, 255, 0);   // 绿色
    private static readonly Scalar TrueColor = new(255, 0, 0);       // 蓝色
    private static readonly Scalar InfoBg = new(40, 40, 40);         // 深灰背景
    private static readonly Scalar InfoText = new(200, 200, 200);    // 浅灰文字

    // ================================================================
    // 测试 1：小角度精度 — 输出每张图的检测结果
    // ================================================================
    [Fact]
    public void Batch_SmallAngles_AllBoards()
    {
        var baseDir = FindAnglePcbDir();
        if (baseDir == null) { Console.WriteLine("SKIP: dataset not found"); return; }

        var boardDirs = Directory.GetDirectories(baseDir).OrderBy(d => d).ToList();
        var outRoot = Path.Combine(
            Path.GetDirectoryName(typeof(BatchTest).Assembly.Location)!, "BatchResults_SmallAngles");
        Directory.CreateDirectory(outRoot);

        int totalImages = 0, totalOk = 0;
        double totalMs = 0;

        Console.WriteLine($"=== Batch Small-Angle Test — Visual Output ===");
        Console.WriteLine($"Output: {outRoot}");
        Console.WriteLine();

        foreach (var boardDir in boardDirs)
        {
            var boardName = Path.GetFileName(boardDir);
            var templatePath = Directory.GetFiles(boardDir, "R_T*.png").FirstOrDefault();
            if (templatePath == null) continue;

            using var template = new Mat(templatePath, ImreadModes.Color);
            using var model = CvNccModel.Create(template, new NccModelOptions
            {
                AngleStart = -2.0,
                AngleExtent = 4.0,
                AngleStep = 0.2,
                NumLevels = 1,
            });

            var testFiles = Directory.GetFiles(boardDir, "PCBR*_*.png")
                .Where(f => !Path.GetFileName(f).StartsWith("R_T", StringComparison.OrdinalIgnoreCase))
                .OrderBy(f => f)
                .ToList();

            var boardOutDir = Path.Combine(outRoot, boardName);
            Directory.CreateDirectory(boardOutDir);

            int boardOk = 0;
            foreach (var testPath in testFiles)
            {
                double trueAngle = ExtractAngle(Path.GetFileName(testPath));
                using var searchMat = new Mat(testPath, ImreadModes.Color);
                using var display = searchMat.Clone();

                var sw = Stopwatch.StartNew();
                var results = model.FindMatches(searchMat, new FindMatchesOptions
                {
                    MinScore = 0.5, MaxMatches = 1, SubPixelRefinement = false,
                });
                sw.Stop();
                totalMs += sw.ElapsedMilliseconds;
                totalImages++;

                string outName = $"{boardName}_{trueAngle:F1}deg".Replace(".", "p") + ".png";
                string outPath = Path.Combine(boardOutDir, outName);

                if (results.Count > 0)
                {
                    var r = results[0];
                    double angErr = Math.Abs(r.Angle - trueAngle);
                    bool ok = angErr <= 0.5;
                    if (ok) boardOk++;

                    // 绘制检测结果
                    DrawRotatedRect(display, r.Position, r.Angle, template.Width, template.Height, DetectedColor);
                    DrawCross(display, (int)r.Position.X, (int)r.Position.Y, 8, DetectedColor);
                    DrawInfoPanel(display, trueAngle, r, angErr, sw.ElapsedMilliseconds, ok);
                }
                else
                {
                    DrawNoMatch(display, trueAngle, sw.ElapsedMilliseconds);
                }

                Cv2.ImWrite(outPath, display);
            }

            totalOk += boardOk;
            Console.WriteLine($"{boardName,-8}: {boardOk}/{testFiles.Count} OK  |  "
                + $"{boardOutDir}");
        }

        Console.WriteLine();
        Console.WriteLine($"TOTAL: {totalOk}/{totalImages} strict-OK ({totalOk*100.0/totalImages:F1}%)");
        Console.WriteLine($"Time: {totalMs/totalImages:F0}ms avg per image");
        Console.WriteLine($"All result images: {outRoot}");

        Assert.True(totalOk >= totalImages * 0.50);
    }

    // ================================================================
    // 测试 2：大角度鲁棒性 — 输出对比图
    // ================================================================
    [Fact]
    public void Batch_LargeAngles_SelectedBoards()
    {
        var baseDir = FindAnglePcbDir();
        if (baseDir == null) { Console.WriteLine("SKIP: dataset not found"); return; }

        var selectedBoards = new[] { "PCBR01", "PCBR04", "PCBR10" };
        double[] testAngles = { 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 };

        var outRoot = Path.Combine(
            Path.GetDirectoryName(typeof(BatchTest).Assembly.Location)!, "BatchResults_LargeAngles");
        Directory.CreateDirectory(outRoot);

        int total = 0, ok = 0;
        double totalMs = 0;
        var summaryLines = new List<string>();

        Console.WriteLine($"=== Batch Large-Angle Test — Visual Output ===");
        Console.WriteLine($"Output: {outRoot}");
        Console.WriteLine();

        foreach (var boardName in selectedBoards)
        {
            var boardDir = Path.Combine(baseDir, boardName);
            var templatePath = Directory.GetFiles(boardDir, "R_T*.png").First();
            var basePath = Directory.GetFiles(boardDir, "PCBR*_0.000000.png").First();

            using var template = new Mat(templatePath, ImreadModes.Color);
            using var baseImage = new Mat(basePath, ImreadModes.Color);

            // Reference position
            using var refModel = CvNccModel.Create(template,
                new NccModelOptions { AngleExtent = 0, NumLevels = 1 });
            var refR = refModel.FindMatches(baseImage,
                new FindMatchesOptions { MinScore = 0.7 });
            double refCX = refR[0].Position.X, refCY = refR[0].Position.Y;
            refModel.Dispose();

            // Main model: auto pyramid + gray fill
            using var model = CvNccModel.Create(template, new NccModelOptions
            {
                AngleStart = 0, AngleExtent = 50, AngleStep = 1.0,
            });

            var boardOutDir = Path.Combine(outRoot, boardName);
            Directory.CreateDirectory(boardOutDir);

            int boardOk = 0;

            foreach (double trueAngle in testAngles)
            {
                total++;
                using var rotated = CvTool.RotateImage(baseImage, trueAngle);
                using var display = rotated.Clone();

                // Ground truth position
                double cx = baseImage.Width / 2.0, cy = baseImage.Height / 2.0;
                double rad = trueAngle * Math.PI / 180.0;
                double dx = refCX - cx, dy = refCY - cy;
                double gtCX = dx * Math.Cos(rad) - dy * Math.Sin(rad) + rotated.Width / 2.0;
                double gtCY = dx * Math.Sin(rad) + dy * Math.Cos(rad) + rotated.Height / 2.0;

                var sw = Stopwatch.StartNew();
                var results = model.FindMatches(rotated, new FindMatchesOptions
                {
                    MinScore = 0.6, MaxMatches = 1, SubPixelRefinement = false,  // Disabled: unstable with pyramid cascade
                });
                sw.Stop();
                totalMs += sw.ElapsedMilliseconds;

                string outName = $"{boardName}_{trueAngle:F0}deg.png";
                string outPath = Path.Combine(boardOutDir, outName);

                if (results.Count > 0)
                {
                    var r = results[0];
                    double pe = Math.Sqrt(Math.Pow(r.Position.X - gtCX, 2)
                                        + Math.Pow(r.Position.Y - gtCY, 2));
                    double ae = Math.Abs(r.Angle - trueAngle);

                    if (pe < 5 && ae < 2.0) { ok++; boardOk++; }

                    // 绘制：绿色=检测，蓝色=真值
                    DrawRotatedRect(display, r.Position, r.Angle, template.Width, template.Height, DetectedColor);
                    DrawCross(display, (int)r.Position.X, (int)r.Position.Y, 12, DetectedColor);
                    DrawRotatedRect(display, new CvPoint(gtCX, gtCY), trueAngle, template.Width, template.Height, TrueColor);
                    DrawCross(display, (int)gtCX, (int)gtCY, 12, TrueColor);
                    DrawPanelLarge(display, r, trueAngle, pe, ae, sw.ElapsedMilliseconds);

                    summaryLines.Add(
                        $"{boardName,-8} | {trueAngle,5:F0}° | {r.Score,8:F4} | {r.Angle,7:F2}° | {pe,6:F1}px | {ae,6:F2}° | {sw.ElapsedMilliseconds,4}ms");
                }
                else
                {
                    DrawNoMatch(display, trueAngle, sw.ElapsedMilliseconds);
                    summaryLines.Add(
                        $"{boardName,-8} | {trueAngle,5:F0}° | {"NO MATCH",8} | {"-",7} | {"-",6} | {"-",6} | {sw.ElapsedMilliseconds,4}ms");
                }

                Cv2.ImWrite(outPath, display);
            }

            Console.WriteLine($"{boardName,-8}: {boardOk}/{testAngles.Length} OK  |  {boardOutDir}");
        }

        Console.WriteLine();
        Console.WriteLine($"{"Board",-8} | {"Angle",5} | {"Score",8} | {"DetAng",7} | {"ErrPos",6} | {"ErrAng",6} | {"Time",4}");
        Console.WriteLine(new string('-', 60));
        foreach (var line in summaryLines) Console.WriteLine(line);
        Console.WriteLine(new string('-', 60));
        Console.WriteLine($"TOTAL: {ok}/{total} passed ({ok*100.0/total:F1}%), avg {totalMs/total:F0}ms/img");
        Console.WriteLine($"All result images: {outRoot}");

        Assert.True(ok >= total * 0.7);
    }

    // ================================================================
    // Drawing helpers
    // ================================================================
    private static void DrawRotatedRect(Mat m, CvPoint center, double angle,
        int w, int h, Scalar color)
    {
        var r = new RotatedRect(
            new Point2f((float)center.X, (float)center.Y),
            new Size2f(w, h), -(float)angle);
        var pts = r.Points();
        for (int i = 0; i < 4; i++)
            Cv2.Line(m,
                new Point((int)pts[i].X, (int)pts[i].Y),
                new Point((int)pts[(i+1)%4].X, (int)pts[(i+1)%4].Y),
                color, 2, LineTypes.AntiAlias);
    }

    private static void DrawCross(Mat m, int x, int y, int len, Scalar color)
    {
        Cv2.Line(m, new Point(x - len, y), new Point(x + len, y), color, 2, LineTypes.AntiAlias);
        Cv2.Line(m, new Point(x, y - len), new Point(x, y + len), color, 2, LineTypes.AntiAlias);
    }

    private static void DrawInfoPanel(Mat image, double trueAngle,
        CvMatchResult r, double angErr, long ms, bool ok)
    {
        int panelW = 240, panelH = 110;
        int panelX = Math.Min(image.Width - panelW - 10, image.Width - panelW - 5);
        int panelY = Math.Min(image.Height - panelH - 10, image.Height - panelH - 5);
        if (panelX < 0) panelX = 5;
        if (panelY < 0) panelY = 5;

        Cv2.Rectangle(image, new Rect(panelX, panelY, panelW, panelH), InfoBg, -1);
        Cv2.Rectangle(image, new Rect(panelX, panelY, panelW, panelH), new Scalar(100, 100, 100), 1);

        string status = ok ? "OK" : "ANGLE-OFF";
        var statusColor = ok ? DetectedColor : new Scalar(0, 165, 255);

        string[] lines =
        {
            $"True Angle: {trueAngle:F1} deg",
            $"Det  Angle: {r.Angle:F2} deg",
            $"Angle Err: {angErr:F2} deg",
            $"Score:     {r.Score:F4}",
            $"Time:      {ms} ms",
            $">> {status} <<",
        };

        for (int i = 0; i < lines.Length; i++)
        {
            var col = i == lines.Length - 1 ? statusColor : InfoText;
            Cv2.PutText(image, lines[i],
                new Point(panelX + 8, panelY + 16 + i * 16),
                HersheyFonts.HersheySimplex, 0.4, col, 1);
        }

        // Draw detected rect
        Cv2.Rectangle(image,
            new Rect(panelX + 5, panelY + 5, panelW - 10, panelH - 10),
            ok ? DetectedColor : new Scalar(0, 165, 255), 1);
    }

    private static void DrawPanelLarge(Mat image, CvMatchResult r,
        double trueAngle, double posErr, double angErr, long ms)
    {
        int panelW = 230, panelH = 140;
        int panelX = 10, panelY = 10;

        Cv2.Rectangle(image, new Rect(panelX, panelY, panelW, panelH), InfoBg, -1);
        Cv2.Rectangle(image, new Rect(panelX, panelY, panelW, panelH), new Scalar(100, 100, 100), 1);

        string[] lines =
        {
            $"True:  {trueAngle,5:F1} deg",
            $"Det:   ({r.Position.X:F0},{r.Position.Y:F0})",
            $"Angle: {r.Angle,5:F2} deg",
            $"Score: {r.Score,5:F4}",
            $"PosErr:{posErr,5:F1} px",
            $"AngErr:{angErr,5:F2} deg",
            $"Time:  {ms} ms",
        };

        for (int i = 0; i < lines.Length; i++)
            Cv2.PutText(image, lines[i],
                new Point(panelX + 8, panelY + 18 + i * 16),
                HersheyFonts.HersheySimplex, 0.4, InfoText, 1);

        // Legend
        int ly = panelY + panelH + 15;
        Cv2.Line(image, new Point(panelX, ly), new Point(panelX + 30, ly), TrueColor, 2);
        Cv2.PutText(image, "True", new Point(panelX + 35, ly + 4),
            HersheyFonts.HersheySimplex, 0.35, TrueColor, 1);
        Cv2.Line(image, new Point(panelX + 80, ly), new Point(panelX + 110, ly), DetectedColor, 2);
        Cv2.PutText(image, "Detected", new Point(panelX + 115, ly + 4),
            HersheyFonts.HersheySimplex, 0.35, DetectedColor, 1);
    }

    private static void DrawNoMatch(Mat image, double trueAngle, long ms)
    {
        Cv2.PutText(image, "NO MATCH", new Point(20, 50),
            HersheyFonts.HersheySimplex, 1.0, new Scalar(0, 0, 255), 2);
        Cv2.PutText(image, $"True: {trueAngle:F1} deg  Time: {ms}ms",
            new Point(20, 80),
            HersheyFonts.HersheySimplex, 0.5, new Scalar(0, 0, 255), 1);
    }

    // ================================================================
    // Helpers
    // ================================================================
    private static double ExtractAngle(string filename)
    {
        var parts = filename.Split('_');
        if (parts.Length >= 3)
        {
            string numStr = parts[^1].Replace(".png", "");
            if (double.TryParse(numStr,
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture,
                out double val))
                return val;
        }
        return 0;
    }

    private static string? FindAnglePcbDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory,
                "..", "..", "..", "..", "..", "test", "TestData", "Real",
                "PCB_Alignment_Datasets", "02_1_Angle0_1Testing", "AnglePCB", "AnglePCB"),
            Path.Combine(
                Path.GetDirectoryName(typeof(BatchTest).Assembly.Location)!,
                "..", "..", "..", "..", "test", "TestData", "Real",
                "PCB_Alignment_Datasets", "02_1_Angle0_1Testing", "AnglePCB", "AnglePCB"),
        };
        foreach (var d in candidates)
        {
            var resolved = Path.GetFullPath(d);
            if (Directory.Exists(resolved)) return resolved;
        }
        return null;
    }
}
