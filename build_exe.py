#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Công cụ Tạo Tệp Thực Thi Khởi Chạy Gọn Nhẹ (`TraCuuPhapLuat.exe` ~24 KB)
và Dọn Dẹp Các Thư Mục Rác Phục Vụ Nộp Sáng Kiến & Đẩy Code Lên GitHub.

Giải quyết triệt để 02 vấn đề của PyInstaller truyền thống đối với dự án AI/PyTorch lớn:
1. Không nhân bản 2.5 GB thư viện PyTorch CUDA sang các thư mục `build/`, `dist/`, `BO_HO_SO_NOP_SANG_KIEN/`.
2. Chỉ tạo DUY NHẤT 01 tệp `TraCuuPhapLuat.exe` (gắn sẵn biểu tượng Logo Học viện ANND) ngay tại thư mục gốc,
   tự động cấu hình chuẩn UTF-8 Tiếng Việt và kích hoạt trực tiếp nhân Python + GPU CUDA trong `.venv`.

Cách sử dụng:
    python build_exe.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


CSHARP_LAUNCHER_SOURCE = r"""
using System;
using System.Diagnostics;
using System.IO;
using System.Text;

class Program
{
    static int Main(string[] args)
    {
        try
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.Title = "HỆ THỐNG AI TRA CỨU VĂN BẢN PHÁP LUẬT VÀ QUY CHẾ ĐÀO TẠO THÔNG MINH - HỌC VIỆN ANND";
        }
        catch { }

        string baseDir = AppDomain.CurrentDomain.BaseDirectory;
        string appScript = Path.Combine(baseDir, "app.py");

        if (!File.Exists(appScript))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("[!] LỖI: Không tìm thấy tệp 'app.py' trong thư mục: " + baseDir);
            Console.WriteLine("    Vui lòng đặt tệp TraCuuPhapLuat.exe cùng thư mục với app.py!");
            Console.ResetColor();
            Console.WriteLine("\nNhấn phím Enter để thoát...");
            Console.ReadLine();
            return 1;
        }

        // Ưu tiên tìm python.exe trong môi trường ảo .venv của dự án
        string[] candidatePythons = new string[]
        {
            Path.Combine(baseDir, ".venv", "Scripts", "python.exe"),
            Path.Combine(baseDir, "venv", "Scripts", "python.exe"),
            "python.exe",
            "python"
        };

        string selectedPython = null;
        foreach (string cand in candidatePythons)
        {
            if (cand.Contains(Path.DirectorySeparatorChar.ToString()))
            {
                if (File.Exists(cand))
                {
                    selectedPython = cand;
                    break;
                }
            }
            else
            {
                selectedPython = cand;
                break;
            }
        }

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("================================================================================");
        Console.WriteLine("  HỌC VIỆN AN NINH NHÂN DÂN (T01)");
        Console.WriteLine("  HỆ THỐNG AI TRA CỨU VĂN BẢN PHÁP LUẬT VÀ QUY CHẾ ĐÀO TẠO THÔNG MINH");
        Console.WriteLine("================================================================================");
        Console.ResetColor();
        Console.WriteLine("[*] Thư mục làm việc : " + baseDir);
        Console.WriteLine("[*] Trình thực thi   : " + selectedPython);
        Console.WriteLine("[*] Đang khởi chạy máy chủ AI và mở trình duyệt tại http://127.0.0.1:8000 ...\n");

        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = selectedPython;
        string extraArgs = args.Length > 0 ? " " + string.Join(" ", args) : "";
        psi.Arguments = "\"" + appScript + "\"" + extraArgs;
        psi.WorkingDirectory = baseDir;
        psi.UseShellExecute = false;
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        psi.EnvironmentVariables["PYTHONUTF8"] = "1";

        try
        {
            using (Process proc = Process.Start(psi))
            {
                proc.WaitForExit();
                if (proc.ExitCode != 0)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine("\n[!] Tiến trình dừng với mã trạng thái: " + proc.ExitCode);
                    Console.ResetColor();
                    Console.WriteLine("Nhấn phím Enter để đóng cửa sổ...");
                    Console.ReadLine();
                }
                return proc.ExitCode;
            }
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("\n[!] Không thể khởi chạy Python: " + ex.Message);
            Console.WriteLine("    Nếu chạy trên máy tính mới, vui lòng cài Python 3.10+ và chạy:");
            Console.WriteLine("    python -m venv .venv && .\\.venv\\Scripts\\activate && pip install -r requirements.txt");
            Console.ResetColor();
            Console.WriteLine("\nNhấn phím Enter để đóng cửa sổ...");
            Console.ReadLine();
            return 1;
        }
    }
}
"""


def cleanup_heavy_build_artifacts(base_dir: Path) -> None:
    """Xóa các thư mục build/dist/BO_HO_SO_NOP_SANG_KIEN nặng ~2GB do PyInstaller tạo ra trước đó."""
    heavy_dirs = [
        base_dir / "build",
        base_dir / "dist",
        base_dir / "BO_HO_SO_NOP_SANG_KIEN",
    ]
    for d in heavy_dirs:
        if d.exists():
            print(f"[*] Đang dọn dẹp thư mục trùng lặp/nặng: {d.name} ...")
            shutil.rmtree(d, ignore_errors=True)

    spec_file = base_dir / "TraCuuPhapLuat.spec"
    if spec_file.exists():
        try:
            spec_file.unlink()
        except OSError:
            pass


def create_academy_ico(base_dir: Path) -> Path | None:
    """Tạo tệp biểu tượng .ico từ `Logo học viện.jpg` để gắn lên tệp `TraCuuPhapLuat.exe`."""
    logo_jpg = base_dir / "Logo học viện.jpg"
    ico_path = base_dir / "logo_t01.ico"
    if not logo_jpg.exists():
        return None
    try:
        from PIL import Image, ImageDraw
        img = Image.open(logo_jpg).convert("RGBA")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize(
            (256, 256), Image.Resampling.LANCZOS
        )
        mask = Image.new("L", (256, 256), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((4, 4, 252, 252), fill=255)
        out = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        out.paste(img, (0, 0), mask=mask)
        out.save(
            ico_path,
            format="ICO",
            sizes=[(256, 256), (64, 64), (48, 48), (32, 32), (16, 16)],
        )
        return ico_path
    except Exception as exc:
        print(f"[!] Không thể tạo biểu tượng .ico ({exc}), sẽ tạo .exe với icon mặc định.")
        return None


def find_windows_csc_compiler() -> Path | None:
    """Tìm trình biên dịch C# nguyên bản (`csc.exe`) có sẵn trong mọi hệ điều hành Windows 10/11."""
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = [
        windir / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe",
        windir / "Microsoft.NET" / "Framework" / "v4.0.30319" / "csc.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_lightweight_exe() -> None:
    base_dir = Path(__file__).resolve().parent
    print("=====================================================================")
    print(" CÔNG CỤ TẠO TỆP THỰC THI GỌN NHẸ TraCuuPhapLuat.exe (CHUẨN NỘP CODE)")
    print("=====================================================================")

    # 1. Dọn dẹp toàn bộ các thư mục nặng 2GB & các file .exe trùng lặp cũ
    cleanup_heavy_build_artifacts(base_dir)

    # 2. Tạo icon Học viện ANND (.ico)
    ico_path = create_academy_ico(base_dir)

    # 3. Biên dịch tệp thực thi duy nhất `TraCuuPhapLuat.exe` ngay tại thư mục gốc
    exe_path = base_dir / "TraCuuPhapLuat.exe"
    cs_path = base_dir / "_launcher_stub.cs"
    csc_exe = find_windows_csc_compiler()

    if csc_exe is not None:
        print(f"[*] Đang biên dịch tệp thực thi gốc bằng {csc_exe.name}...")
        cs_path.write_text(CSHARP_LAUNCHER_SOURCE.strip(), encoding="utf-8-sig")
        cmd = [
            str(csc_exe),
            "/nologo",
            "/target:exe",
            "/optimize+",
            f"/out:{exe_path}",
        ]
        if ico_path and ico_path.exists():
            cmd.append(f"/win32icon:{ico_path}")
        cmd.append(str(cs_path))

        try:
            subprocess.run(cmd, check=True)
            print(f"[+] Đã tạo thành công DUY NHẤT tệp thực thi: {exe_path}")
            print(f"[+] Dung lượng tệp TraCuuPhapLuat.exe: {exe_path.stat().st_size / 1024:.1f} KB (Siêu nhẹ!)")
        finally:
            if cs_path.exists():
                cs_path.unlink()
            if ico_path and ico_path.exists():
                ico_path.unlink()
    else:
        print("[!] Đang chạy trên môi trường không phải Windows (.NET csc.exe).")

    # 4. Tạo kèm tệp cài đặt môi trường 1 chạm cho máy tính mới (`CaiDat_MoiTruong_MayMoi.bat`)
    setup_bat = base_dir / "CaiDat_MoiTruong_MayMoi.bat"
    setup_bat.write_text(
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        "cd /d \"%~dp0\"\r\n"
        "echo ===================================================================\r\n"
        "echo  CAI DAT MOI TRUONG TU DONG TREN MAY TINH MOI (CHI CHAY 1 LAN DAU)\r\n"
        "echo ===================================================================\r\n"
        "if not exist \".venv\\Scripts\\python.exe\" (\r\n"
        "    echo [*] Dang khoi tao moi truong ao .venv...\r\n"
        "    python -m venv .venv\r\n"
        ")\r\n"
        "call .venv\\Scripts\\activate\r\n"
        "echo [*] Dang cai dat cac thu vien trong requirements.txt...\r\n"
        "pip install -r requirements.txt\r\n"
        "echo [*] Dang kiem tra va kich hoat GPU NVIDIA (neu may co card NVIDIA)...\r\n"
        "python hybrid_search.py --fix-gpu\r\n"
        "echo.\r\n"
        "echo [+] HOAN TAT! Bay gio ban co the nhap dup vao TraCuuPhapLuat.exe de su dung.\r\n"
        "pause\r\n",
        encoding="utf-8",
    )

    print("\n=====================================================================")
    print(" [HOÀN TẤT!]")
    print(f" 1. Để chạy ứng dụng: Nhấp đúp vào `{exe_path.name}` ngay trong thư mục này.")
    print(" 2. Không còn các thư mục rác nặng 2 GB (build/, dist/, BO_HO_SO_NOP_SANG_KIEN/).")
    print(" 3. Khi sang máy tính mới: Chạy `CaiDat_MoiTruong_MayMoi.bat` (1 lần đầu),")
    print("    sau đó bấm `TraCuuPhapLuat.exe` để sử dụng bình thường.")
    print("=====================================================================\n")


if __name__ == "__main__":
    build_lightweight_exe()