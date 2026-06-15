using System.Diagnostics;
using CvCommon;
using CvLibrary.OpenCV;
using CvLibrary.OpenCV.Match;
using OpenCvSharp;

namespace CvLibrary.Tests.Match;

public class LargeAngleDemo
{
    private static readonly Scalar DetectedColor = new(0, 255, 0);
    private static readonly Scalar TrueColor = new(255, 0, 0);
    private static readonly Scalar PanelBg = new(255, 255, 255);
    private static readonly Scalar PanelBorder = new(0, 0, 0);

    [Fact]
    public void Demo_LargeAngles()
    {
        var baseDir = FindAnglePcbDir();
        if (baseDir == null) { Console.WriteLine("ERROR: dataset not found"); return; }

        var pcbDir = Path.Combine(baseDir, "PCBR01");
        using var template = new Mat(Path.Combine(pcbDir, "R_T01.png"), ImreadModes.Color);
        using var baseImage = new Mat(Path.Combine(pcbDir, "PCBR01_501_0.000000.png"), ImreadModes.Color);

        // 参考位置
        var refModel = CvNccModel.Create(template, new NccModelOptions { AngleExtent = 0, NumLevels = 1 });
        var refR = refModel.FindMatches(baseImage, new FindMatchesOptions { MinScore = 0.7 });
        double refCX = refR[0].Position.X, refCY = refR[0].Position.Y;
        refModel.Dispose();

        // 模型：自动金字塔 + 灰色填充（避免黑边零方差）
        var model = CvNccModel.Create(template, new NccModelOptions
        {
            AngleStart = 0, AngleExtent = 50, AngleStep = 1.0,
        });

        double[] testAngles = { 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 };

        var outDir = Path.Combine(
            Path.GetDirectoryName(typeof(LargeAngleDemo).Assembly.Location)!, "LargeAngleResults");
        Directory.CreateDirectory(outDir);

        Console.WriteLine($"Template: {template.Width}x{template.Height}  Base: {baseImage.Width}x{baseImage.Height}");
        Console.WriteLine($"Ref: ({refCX:F0},{refCY:F0})  Model: auto pyramid (L={model.Levels}), 0-50 deg, step=1.0, gray fill");
        Console.WriteLine(new string('-', 70));

        foreach (double trueAngle in testAngles)
        {
            // 旋转整图
            using var rotatedImage = CvTool.RotateImage(baseImage, trueAngle);

            // GT 位置
            double imgCX = baseImage.Width / 2.0, imgCY = baseImage.Height / 2.0;
            double rad = trueAngle * Math.PI / 180.0;
            double dx = refCX - imgCX, dy = refCY - imgCY;
            double gtCX = dx * Math.Cos(rad) - dy * Math.Sin(rad) + rotatedImage.Width / 2.0;
            double gtCY = dx * Math.Sin(rad) + dy * Math.Cos(rad) + rotatedImage.Height / 2.0;

            // 计时匹配
            var sw = Stopwatch.StartNew();
            var results = model.FindMatches(rotatedImage, new FindMatchesOptions
            {
                MinScore = 0.6,
                MaxMatches = 1,
                SubPixelRefinement = false,
            });
            sw.Stop();

            // 画结果图
            using var display = rotatedImage.Clone();

            if (results.Count > 0)
            {
                var r = results[0];
                double ae = r.Angle - trueAngle;
                double pe = Math.Sqrt(Math.Pow(r.Position.X - gtCX, 2) + Math.Pow(r.Position.Y - gtCY, 2));
                string s = ae >= 0 ? "+" : "";

                Console.WriteLine(
                    $"  True={trueAngle,3:F0} deg  |  "
                    + $"Score={r.Score:F4}  Det=({r.Position.X:F0},{r.Position.Y:F0})  "
                    + $"Ang={r.Angle:F2} deg  Err(pos)={pe:F1}px  Err(ang)={s}{ae:F2} deg  |  "
                    + $"{sw.ElapsedMilliseconds}ms");

                DrawRot(display, r.Position, r.Angle, template.Width, template.Height, DetectedColor);
                DrawCross(display, (int)r.Position.X, (int)r.Position.Y, 12, DetectedColor);
                DrawRot(display, new CvPoint(gtCX, gtCY), trueAngle, template.Width, template.Height, TrueColor);
                DrawCross(display, (int)gtCX, (int)gtCY, 12, TrueColor);
                DrawPanel(display, r, trueAngle, pe, ae, sw.ElapsedMilliseconds);
            }
            else
            {
                Console.WriteLine(
                    $"  True={trueAngle,3:F0} deg  |  NO MATCH (minScore=0.45)  |  {sw.ElapsedMilliseconds}ms");
                Cv2.PutText(display, "NO MATCH", new Point(20, 40),
                    HersheyFonts.HersheySimplex, 0.8, new Scalar(0, 0, 255), 2);
            }

            Cv2.ImWrite(Path.Combine(outDir, $"imgrot_{trueAngle:F0}deg.png"), display);
        }

        model.Dispose();
        Console.WriteLine(new string('-', 70));
        Console.WriteLine($"Results: {outDir}");
    }

    static void DrawRot(Mat m, CvPoint c, double a, int w, int h, Scalar col)
    {
        var r = new RotatedRect(new Point2f((float)c.X, (float)c.Y), new Size2f(w, h), -(float)a);
        var p = r.Points();
        for (int i = 0; i < 4; i++)
            Cv2.Line(m, new Point((int)p[i].X, (int)p[i].Y), new Point((int)p[(i + 1) % 4].X, (int)p[(i + 1) % 4].Y), col, 1, LineTypes.AntiAlias);
    }
    static void DrawCross(Mat m, int x, int y, int l, Scalar c)
    {
        Cv2.Line(m, new Point(x - l, y), new Point(x + l, y), c, 1, LineTypes.AntiAlias);
        Cv2.Line(m, new Point(x, y - l), new Point(x, y + l), c, 1, LineTypes.AntiAlias);
    }
    static void DrawPanel(Mat m, CvMatchResult r, double ta, double pe, double ae, long ms)
    {
        string s = ae >= 0 ? "+" : "";
        string[] ll = { $"True:   {ta,5:F1} deg", $"Det:    ({r.Position.X:F0},{r.Position.Y:F0})",
            $"Angle:  {r.Angle,5:F2} deg", $"Score:  {r.Score,5:F4}",
            $"PosErr: {pe,5:F1} px", $"AngErr: {s}{ae,5:F2} deg", $"Time:   {ms} ms" };
        int pw = 200, ph = 144;
        Cv2.Rectangle(m, new Rect(10, 10, pw, ph), PanelBg, -1);
        Cv2.Rectangle(m, new Rect(10, 10, pw, ph), PanelBorder, 1);
        for (int i = 0; i < ll.Length; i++)
            Cv2.PutText(m, ll[i], new Point(18, 26 + i * 18), HersheyFonts.HersheySimplex, 0.4, new Scalar(0, 0, 0), 1);
        int ly = 168;
        Cv2.Line(m, new Point(15, ly), new Point(45, ly), TrueColor, 2);
        Cv2.PutText(m, "True", new Point(50, ly + 4), HersheyFonts.HersheySimplex, 0.35, TrueColor, 1);
        Cv2.Line(m, new Point(90, ly), new Point(120, ly), DetectedColor, 2);
        Cv2.PutText(m, "Detected", new Point(125, ly + 4), HersheyFonts.HersheySimplex, 0.35, DetectedColor, 1);
    }

    static string? FindAnglePcbDir()
    {
        foreach (var d in new[] {
            Path.Combine(AppContext.BaseDirectory,"..","..","..","..","..","test","TestData","Real","PCB_Alignment_Datasets","02_1_Angle0_1Testing","AnglePCB","AnglePCB"),
            Path.Combine(Path.GetDirectoryName(typeof(LargeAngleDemo).Assembly.Location)!,"..","..","..","..","test","TestData","Real","PCB_Alignment_Datasets","02_1_Angle0_1Testing","AnglePCB","AnglePCB"),
        }) { var r = Path.GetFullPath(d); if (Directory.Exists(r)) return r; }
        return null;
    }
}
