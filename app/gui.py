from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pic2Mesh")
        self.geometry("700x340")
        self.resizable(False, False)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.blender_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="single")
        self.preset_var = tk.StringVar(value="balanced")
        self.export_var = tk.StringVar(value="glb")

        self._build()

    def _build(self):
        pad = {"padx": 8, "pady": 6}
        row = 0

        ttk.Label(self, text="Input image/folder").grid(column=0, row=row, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.input_var, width=68).grid(column=1, row=row, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self.pick_input).grid(column=2, row=row, **pad)
        row += 1

        ttk.Label(self, text="Output folder").grid(column=0, row=row, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.output_var, width=68).grid(column=1, row=row, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self.pick_output).grid(column=2, row=row, **pad)
        row += 1

        ttk.Label(self, text="Blender executable").grid(column=0, row=row, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.blender_var, width=68).grid(column=1, row=row, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self.pick_blender).grid(column=2, row=row, **pad)
        row += 1

        ttk.Label(self, text="Mode").grid(column=0, row=row, sticky="w", **pad)
        ttk.Combobox(self, textvariable=self.mode_var, values=["single", "multi"], state="readonly", width=20).grid(column=1, row=row, sticky="w", **pad)
        row += 1

        ttk.Label(self, text="Preset").grid(column=0, row=row, sticky="w", **pad)
        ttk.Combobox(self, textvariable=self.preset_var, values=["fast", "balanced", "high"], state="readonly", width=20).grid(column=1, row=row, sticky="w", **pad)
        row += 1

        ttk.Label(self, text="Export format").grid(column=0, row=row, sticky="w", **pad)
        ttk.Combobox(self, textvariable=self.export_var, values=["glb", "fbx", "obj"], state="readonly", width=20).grid(column=1, row=row, sticky="w", **pad)
        row += 1

        ttk.Button(self, text="Run", command=self.run_pipeline).grid(column=1, row=row, sticky="w", **pad)

    def pick_input(self):
        path = filedialog.askopenfilename(title="Pick image")
        if path:
            self.input_var.set(path)

    def pick_output(self):
        path = filedialog.askdirectory(title="Pick output folder")
        if path:
            self.output_var.set(path)

    def pick_blender(self):
        path = filedialog.askopenfilename(title="Pick blender.exe", filetypes=[("Blender", "blender.exe"), ("All", "*")])
        if path:
            self.blender_var.set(path)

    def run_pipeline(self):
        root = Path(__file__).resolve().parent.parent
        cmd = [
            sys.executable,
            "-m",
            "app.main",
            "--input",
            self.input_var.get(),
            "--output",
            self.output_var.get(),
            "--blender",
            self.blender_var.get(),
            "--mode",
            self.mode_var.get(),
            "--preset",
            self.preset_var.get(),
            "--export",
            self.export_var.get(),
        ]
        try:
            proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            if proc.returncode == 0:
                messagebox.showinfo("Success", proc.stdout.strip() or "Pipeline complete")
            else:
                messagebox.showerror("Error", proc.stderr or proc.stdout)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    App().mainloop()
