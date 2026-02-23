import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu
import os
import glob
import threading
import time
import math
import sys
import queue
import gc
import logging
import matplotlib.pyplot as plt

# --- Custom Modules ---
from audio_processor import AudioProcessor
from transcription_engine import TranscriptionEngine
from metadata_handler import MetadataHandler
from mood_analyzer import MoodAnalyzer
from logging_utils import setup_logging, log_context

# --- Dependency Checks ---
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    messagebox.showerror("Dependency Error", "Please install TkinterDnD2:\npip install tkinterdnd2-universal")
    sys.exit(1)

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DOWNLOAD_DIR = os.path.join(SCRIPT_DIR, "Audio Models")
AUDIO_EXTENSIONS = ["*.wav", "*.mp3", "*.flac", "*.aac", "*.m4a", "*.ogg"]
VIDEO_EXTENSIONS = ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.wmv", "*.flv"]
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS
MODELS = {
    "deep-learning-analytics/whisper-large-v3-turbo": "Large v3 Turbo. Optimized for speed/accuracy balance.",
    "large-v3": "Large v3 (~1.55B). Best accuracy.",
    "large-v2": "Large v2 (~1.55B). Improved.",
    "medium": "Medium (~769M). High accuracy.", "medium.en": "Medium English-only (~769M).",
    "small": "Small (~244M). Good accuracy.", "small.en": "Small English-only (~244M).",
    "base": "Base (~74M). Balanced.", "base.en": "Base English-only (~74M).",
    "tiny": "Tiny (~39M). Fast, low accuracy.", "tiny.en": "Tiny English-only (~39M).",
    "distil-large-v3": "Distilled L-v3. Faster large-v3.",
    "distil-large-v2": "Distilled L-v2 (~750M). Faster large.",
    "distil-medium.en": "Distilled M-en (~450M). Faster medium.en.",
    "distil-small.en": "Distilled S-en (~150M). Faster small.en."
}
MODEL_SIZES = list(MODELS.keys())
QUANTIZED_MODEL_SUFFIXES = ["int8", "int8_float16", "int8_bfloat16", "int16", "float16"]
COMPUTE_TYPES_CPU = ["default", "auto", "int8", "int16", "float32"]
COMPUTE_TYPES_CUDA = ["default", "auto", "float16", "int8_float16", "int8", "bfloat16", "int8_bfloat16"]
COMPUTE_TYPES_ROCM = ["default", "auto", "float16", "int8_float16", "int8"]

# --- Theme Colors ---
DARK_BG = "#1e1e1e"; DARK_FG = "#e0e0e0"; DARK_WIDGET_BG = "#2d2d2d"
DARK_SELECT_BG = "#3a3d41"; DARK_BUTTON = "#007acc"; DARK_BUTTON_FG = "#ffffff"
DARK_BUTTON_ACTIVE = "#005f9e"; STOP_BUTTON_BG = "#c74e4e"; STOP_BUTTON_ACTIVE = "#a13e3e"
DARK_TEXT_AREA = "#252526"; DARK_PROGRESS_BAR = "#9b59b6"; DARK_PROGRESS_BG = "#3a3d41"
DARK_BORDER = "#4a4d51"; ACCENT_COLOR = "#9b59b6"; PLOT_BG = "#2d2d2d"
LISTBOX_BG = "#252526"; LISTBOX_FG = "#e0e0e0"; LISTBOX_SELECT_BG = ACCENT_COLOR; LISTBOX_SELECT_FG = DARK_BG

# Status indicators
STATUS_PENDING = "⏳"
STATUS_PROCESSING = "⚙️"
STATUS_COMPLETED = "✅"
STATUS_SKIPPED = "⚠️"
STATUS_ERROR = "❌"

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

# --- Helper Functions ---
def segments_to_srt(segments):
    from faster_whisper import format_timestamp
    srt_content = ""
    for i, segment in enumerate(segments):
        start_time = format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')
        end_time = format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')
        text = segment.text.strip().replace('-->', '->')
        srt_content += f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_content

def segments_to_lrc(segments):
    lrc_content = ""
    for segment in segments:
        total_seconds = segment.start; minutes = int(total_seconds // 60); seconds = int(total_seconds % 60)
        centiseconds = int((total_seconds - int(total_seconds)) * 100)
        lrc_timestamp = f"[{minutes:02d}:{seconds:02d}.{centiseconds:02d}]"
        text = segment.text.strip()
        if text: lrc_content += f"{lrc_timestamp}{text}\n"
    return lrc_content

def segments_to_txt(segments): return "\n".join(segment.text.strip() for segment in segments)

def format_eta(seconds):
    if seconds < 0 or not math.isfinite(seconds): return "--:--"
    seconds = int(seconds); hours = seconds // 3600; minutes = (seconds % 3600) // 60; secs = seconds % 60
    if hours > 0: return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0: return f"{minutes}m {secs:02d}s"
    else: return f"{secs}s"

# --- Tooltip Class ---
class ToolTip:
    def __init__(self, widget, status_label, text='widget info'):
        self.widget = widget; self.status_label = status_label; self.text = text
        self.widget.bind("<Enter>", self.enter); self.widget.bind("<Leave>", self.leave); self.widget.bind("<ButtonPress>", self.leave)
    def enter(self, event=None): self.status_label.config(text=self.text)
    def leave(self, event=None): self.status_label.config(text="")

# --- GUI Class ---
class WhisperGUI:
    def __init__(self, master):
        self.master = master
        master.title("Codex Audio Transcriber V2")
        master.geometry("1000x800")

        # --- Logging Setup ---
        self.logger, self.log_listener = setup_logging()
        self.log_listener.start()

        # --- Variables ---
        self.input_mode = tk.StringVar(value="single"); self.input_path = tk.StringVar()
        self.input_label_text = tk.StringVar(value="Select File:")
        self.model_size = tk.StringVar(value="large-v2")
        self.quantization = tk.StringVar(value="") # Default to empty (let model decide or user select valid)
        self.device = tk.StringVar()
        self.compute_type = tk.StringVar(value="auto") # Default auto
        self.vad_filter = tk.BooleanVar(value=False); self.beam_size = tk.IntVar(value=5)
        self.overwrite_output = tk.BooleanVar(value=False)
        self.translate_output = tk.BooleanVar(value=False) # New Translation Toggle
        self.model_description = tk.StringVar(value=MODELS.get(self.model_size.get(), "..."))

        # --- Processing State ---
        self.processing_active = False; self.processing_thread = None; self.stop_requested = 0
        self.file_start_time = None; self.batch_start_time = None
        self.completed_file_times = []; self.total_batch_files = 0; self.processed_batch_files = 0
        self.file_data = {}

        # --- Visualizer Variables ---
        self.fig, self.ax = None, None; self.canvas, self.canvas_widget = None, None
        self.visualizer_frame = None
        self.spectrogram_line_ref = [None] # List to hold reference mutably

        # --- Detect Devices ---
        self.available_devices = ["cpu"]
        if CUDA_AVAILABLE:
            self.available_devices.append("cuda")
            default_device = "cuda"; default_compute = "float16"
        else:
            default_device = "cpu"; default_compute = "int8"
            self.logger.info("CUDA GPU NOT DETECTED - Defaulting to CPU")

        self.device.set(default_device); self.compute_type.set(default_compute)

        # --- Tooltips ---
        self.tooltips = {
            "input_mode_single": "Process a single audio/video file.",
            "input_mode_batch": "Process all supported files in a directory.",
            "input_path": "Path to file/directory.",
            "input_browse": "Browse for input.",
            "model_size": "Select Whisper model.",
            "model_description_area": "Model description.",
            "quantization": "Model quantization (int8, float16).",
            "device": "Processing device (CPU/CUDA).",
            "compute_type": "Computation data type.",
            "beam_size": "Decoding beams (Default: 5).",
            "vad_filter": "Voice Activity Detection.",
            "overwrite_output": "Overwrite existing output files.",
            "translate_output": "Translate to English (generate subtitles).",
            "output_info": "Output: SRT (video) / LRC (audio).",
            "start_stop_button": "Start/Stop processing.",
            "file_progress_label": "Current file progress.",
            "batch_progress_label": "Batch progress.",
            "file_progress_bar": "Current file progress bar.",
            "batch_progress_bar": "Batch progress bar.",
            "visualizer_area": "Audio spectrogram.",
            "file_status_list": "File queue status.",
            "status_bar": "Tooltips."
        }

        # --- Layout ---
        self.style = ttk.Style(master); self.setup_dark_theme()
        master.configure(bg=DARK_BG)
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        master.columnconfigure(0, weight=1); master.rowconfigure(0, weight=1)

        # Ensure Model Dir
        os.makedirs(MODEL_DOWNLOAD_DIR, exist_ok=True)

        # Build UI
        self.create_widgets()
        self.init_plot()
        self.assign_tooltips()
        self.update_input_label()

        # Connect Drag & Drop
        self.main_frame.drop_target_register('DND_Files')
        self.main_frame.dnd_bind('<<Drop>>', self.handle_drop)

    def create_widgets(self):
        # Menu Bar
        self.menu_bar = Menu(self.master)
        self.master.config(menu=self.menu_bar)
        tools_menu = Menu(self.menu_bar, tearoff=0, background=DARK_WIDGET_BG, foreground=DARK_FG)
        self.menu_bar.add_cascade(label="Tools", menu=tools_menu)
        # tools_menu.add_command(label="Install Core Dependencies", command=self.run_install_core_deps_thread)
        # tools_menu.add_command(label="Check/Install PyTorch", command=self.check_pytorch_install)
        tools_menu.add_separator()
        tools_menu.add_command(label="Open Model Folder", command=self.open_model_folder)

        # Main Frame
        self.main_frame = ttk.Frame(self.master, padding="10", style="Futuristic.TFrame")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.columnconfigure(0, weight=3); self.main_frame.columnconfigure(1, weight=2)
        self.main_frame.rowconfigure(4, weight=1)

        # Left Panel
        left_panel = ttk.Frame(self.main_frame, style="Futuristic.TFrame")
        left_panel.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)

        # Input Frame
        input_frame = ttk.LabelFrame(left_panel, text="📁 Input Source", padding=(10, 5), style="Futuristic.TLabelframe")
        input_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=(5, 10)); input_frame.columnconfigure(1, weight=1)

        ttk.Radiobutton(input_frame, text="Single File", variable=self.input_mode, value="single", command=self.update_input_label, style="Futuristic.TRadiobutton").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Radiobutton(input_frame, text="Batch Directory", variable=self.input_mode, value="batch", command=self.update_input_label, style="Futuristic.TRadiobutton").grid(row=0, column=1, columnspan=2, padx=5, pady=2, sticky="w")

        self.lbl_input = ttk.Label(input_frame, textvariable=self.input_label_text, style="Futuristic.TLabel")
        self.lbl_input.grid(row=1, column=0, padx=5, pady=2, sticky="w")

        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_path, width=40, style="Futuristic.TEntry")
        self.input_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        self.btn_browse = ttk.Button(input_frame, text="Browse...", command=self.browse_input, style="Accent.TButton")
        self.btn_browse.grid(row=1, column=2, padx=5, pady=2)

        # Model Frame
        model_frame = ttk.LabelFrame(left_panel, text="🧠 Model Configuration", padding=(10, 5), style="Futuristic.TLabelframe")
        model_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        model_frame.columnconfigure(1, weight=1); model_frame.columnconfigure(3, weight=1)

        ttk.Label(model_frame, text="Model Size:", style="Futuristic.TLabel").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_size, values=MODEL_SIZES, width=18, style="Futuristic.TCombobox")
        self.model_combo.grid(row=0, column=1, columnspan=3, padx=5, pady=2, sticky="ew")
        self.model_combo.bind("<<ComboboxSelected>>", self.update_model_description)

        ttk.Label(model_frame, text="Device:", style="Futuristic.TLabel").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.device_combo = ttk.Combobox(model_frame, textvariable=self.device, values=self.available_devices, state="readonly", width=8, style="Futuristic.TCombobox")
        self.device_combo.grid(row=1, column=1, columnspan=3, padx=5, pady=2, sticky="ew")
        # removed update_compute_types binding as it is now auto

        self.model_desc_label = ttk.Label(model_frame, textvariable=self.model_description, wraplength=450, justify=tk.LEFT, style="Desc.TLabel")
        self.model_desc_label.grid(row=2, column=0, columnspan=4, padx=5, pady=(5,2), sticky="ew")

        # Processing Frame
        proc_frame = ttk.LabelFrame(left_panel, text="⚙️ Processing Options", padding=(10, 5), style="Futuristic.TLabelframe")
        proc_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        ttk.Label(proc_frame, text="Beam Size:", style="Futuristic.TLabel").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.beam_spinbox = ttk.Spinbox(proc_frame, from_=1, to=100, textvariable=self.beam_size, width=5, style="Futuristic.TSpinbox")
        self.beam_spinbox.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        self.vad_check = ttk.Checkbutton(proc_frame, text="VAD Filter", variable=self.vad_filter, style="Futuristic.TCheckbutton")
        self.vad_check.grid(row=0, column=2, padx=15, pady=2, sticky="w")

        # Overwrite and Translation
        self.overwrite_check = ttk.Checkbutton(proc_frame, text="Overwrite Existing", variable=self.overwrite_output, style="Futuristic.TCheckbutton")
        self.overwrite_check.grid(row=1, column=0, columnspan=2, padx=5, pady=2, sticky="w")

        self.translate_check = ttk.Checkbutton(proc_frame, text="Translate to English", variable=self.translate_output, style="Futuristic.TCheckbutton")
        self.translate_check.grid(row=1, column=2, padx=5, pady=2, sticky="w")

        # Output Frame
        output_frame = ttk.LabelFrame(left_panel, text="💾 Output", padding=(10, 5), style="Futuristic.TLabelframe")
        output_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.output_info_label = ttk.Label(output_frame, text="Output: SRT (Video) / LRC (Audio)", style="Desc.TLabel")
        self.output_info_label.pack(anchor="w", padx=5, pady=5)

        # Action & Progress
        action_frame = ttk.Frame(self.main_frame, padding=(10, 5), style="Futuristic.TFrame")
        action_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        action_frame.columnconfigure(1, weight=1)

        self.file_progress_label = ttk.Label(action_frame, text="File Progress:", style="Futuristic.TLabel")
        self.file_progress_label.grid(row=0, column=0, sticky="w")

        self.batch_progress_label = ttk.Label(action_frame, text="Batch Progress:", style="Futuristic.TLabel")

        self.file_progress_bar = ttk.Progressbar(action_frame, orient=tk.HORIZONTAL, length=150, mode='determinate', style="Futuristic.Horizontal.TProgressbar")
        self.file_progress_bar.grid(row=0, column=1, sticky="ew", padx=5)

        self.batch_progress_bar = ttk.Progressbar(action_frame, orient=tk.HORIZONTAL, length=300, mode='determinate', style="Futuristic.Horizontal.TProgressbar")

        self.start_stop_button = ttk.Button(action_frame, text="🚀 Start Transcription", command=self.toggle_processing, style="Accent.TButton", width=20)
        self.start_stop_button.grid(row=0, column=2, rowspan=2, padx=10, sticky="e")

        # File List
        filelist_frame = ttk.LabelFrame(self.main_frame, text="📊 File Queue", padding=(10, 5), style="Futuristic.TLabelframe")
        filelist_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        filelist_frame.columnconfigure(0, weight=1); filelist_frame.rowconfigure(0, weight=1)

        self.file_tree = ttk.Treeview(filelist_frame, columns=("filename", "type", "status", "progress"), show="headings", style="Futuristic.Treeview")
        self.file_tree.heading("filename", text="Filename"); self.file_tree.heading("type", text="Type")
        self.file_tree.heading("status", text="Status"); self.file_tree.heading("progress", text="Progress")
        self.file_tree.column("filename", width=400); self.file_tree.column("type", width=60, anchor="center")
        self.file_tree.column("status", width=100, anchor="center"); self.file_tree.column("progress", width=80, anchor="e")

        scrolly = ttk.Scrollbar(filelist_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        scrollx = ttk.Scrollbar(filelist_frame, orient=tk.HORIZONTAL, command=self.file_tree.xview)
        self.file_tree.configure(yscrollcommand=scrolly.set, xscrollcommand=scrollx.set)

        self.file_tree.grid(row=0, column=0, sticky="nsew")
        scrolly.grid(row=0, column=1, sticky="ns"); scrollx.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Visualizer
        self.visualizer_outer_frame = ttk.LabelFrame(self.main_frame, text="🔊 Audio Spectrogram", padding=(10, 5), style="Futuristic.TLabelframe")
        self.visualizer_outer_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=(10, 5), pady=5)
        self.visualizer_outer_frame.columnconfigure(0, weight=1); self.visualizer_outer_frame.rowconfigure(0, weight=1)

        self.visualizer_frame = ttk.Frame(self.visualizer_outer_frame, style="Futuristic.TFrame", relief=tk.SUNKEN, borderwidth=1)
        self.visualizer_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Status Bar
        self.status_bar = ttk.Label(self.main_frame, text="Hover for details.", style="Status.TLabel", padding=5)
        self.status_bar.grid(row=5, column=0, columnspan=2, sticky="ew")

    def setup_dark_theme(self):
        """Configures ttk styles."""
        self.style.theme_use('clam')
        self.style.configure('.', background=DARK_BG, foreground=DARK_FG, fieldbackground=DARK_WIDGET_BG, bordercolor=DARK_BORDER)
        self.style.map('.', background=[('active', DARK_SELECT_BG)], foreground=[('disabled', '#777777')])

        self.style.configure("Futuristic.TFrame", background=DARK_BG)
        self.style.configure("Futuristic.TLabel", background=DARK_BG, foreground=DARK_FG)
        self.style.configure("Desc.TLabel", background=DARK_BG, foreground="#a0a0a0", font=('Segoe UI', 8, 'italic'))
        self.style.configure("Futuristic.TLabelframe", background=DARK_BG, foreground=DARK_FG, bordercolor=DARK_BORDER)
        self.style.configure("Futuristic.TLabelframe.Label", background=DARK_BG, foreground=ACCENT_COLOR, font=('Segoe UI', 10, 'bold'))
        self.style.configure("Accent.TButton", background=ACCENT_COLOR, foreground=DARK_BG, bordercolor=DARK_BORDER, font=('Segoe UI', 9, 'bold'))
        self.style.map("Accent.TButton", background=[('active', '#af7ac5')], foreground=[('disabled', '#777777')])
        self.style.configure("Stop.TButton", background=STOP_BUTTON_BG, foreground=DARK_BUTTON_FG, font=('Segoe UI', 9, 'bold'))
        self.style.map("Stop.TButton", background=[('active', STOP_BUTTON_ACTIVE)])
        self.style.configure("Futuristic.TEntry", fieldbackground=DARK_WIDGET_BG, foreground=DARK_FG, insertcolor=DARK_FG, bordercolor=DARK_BORDER)
        self.style.configure("Futuristic.TCombobox", fieldbackground=DARK_WIDGET_BG, background=DARK_WIDGET_BG, foreground=DARK_FG, arrowcolor=DARK_FG, bordercolor=DARK_BORDER)
        self.style.map("Futuristic.TCombobox", fieldbackground=[('readonly', DARK_WIDGET_BG)], selectbackground=[('focus', DARK_SELECT_BG)])
        self.style.configure("Futuristic.TSpinbox", fieldbackground=DARK_WIDGET_BG, background=DARK_WIDGET_BG, foreground=DARK_FG, arrowcolor=DARK_FG, bordercolor=DARK_BORDER)
        self.style.configure("Futuristic.TCheckbutton", background=DARK_BG, foreground=DARK_FG, indicatorcolor=DARK_WIDGET_BG)
        self.style.map("Futuristic.TCheckbutton", indicatorcolor=[('selected', ACCENT_COLOR), ('active', DARK_SELECT_BG)], background=[('active', DARK_BG)])
        self.style.configure("Futuristic.TRadiobutton", background=DARK_BG, foreground=DARK_FG, indicatorcolor=DARK_WIDGET_BG)
        self.style.map("Futuristic.TRadiobutton", indicatorcolor=[('selected', ACCENT_COLOR), ('active', DARK_SELECT_BG)], background=[('active', DARK_BG)])
        self.style.configure("Futuristic.Horizontal.TProgressbar", troughcolor=DARK_PROGRESS_BG, background=DARK_PROGRESS_BAR, bordercolor=DARK_BORDER, thickness=10)
        self.style.configure("Status.TLabel", background=DARK_BG, foreground="#a0a0a0", font=('Segoe UI', 8))
        self.style.configure("Futuristic.Treeview", background=LISTBOX_BG, foreground=LISTBOX_FG, fieldbackground=LISTBOX_BG, rowheight=25)
        self.style.map("Futuristic.Treeview", background=[('selected', LISTBOX_SELECT_BG)], foreground=[('selected', LISTBOX_SELECT_FG)])
        self.style.configure("Futuristic.Treeview.Heading", background=DARK_WIDGET_BG, foreground=DARK_FG, font=('Segoe UI', 9, 'bold'))

    def assign_tooltips(self):
        ToolTip(self.btn_browse, self.status_bar, self.tooltips["input_browse"])
        ToolTip(self.start_stop_button, self.status_bar, self.tooltips["start_stop_button"])
        # ... (Add other tooltips if needed)

    def log(self, message):
        self.logger.info(message)

    def init_plot(self):
        try:
            self.fig, self.ax = plt.subplots(facecolor=PLOT_BG, constrained_layout=True)
            self.ax.set_facecolor(PLOT_BG)
            self.ax.tick_params(colors=DARK_FG, labelsize=8)
            for spine in self.ax.spines.values(): spine.set_color(DARK_BORDER)
            self.ax.text(0.5, 0.5, 'Spectrogram', ha='center', va='center', transform=self.ax.transAxes, color=DARK_FG)
            self.ax.set_xticks([]); self.ax.set_yticks([])
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualizer_frame)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            self.log(f"Plot Error: {e}")

    def clear_plot(self):
        if self.ax:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'Cleared', ha='center', va='center', transform=self.ax.transAxes, color=DARK_FG)
            self.ax.set_xticks([]); self.ax.set_yticks([])
            self.canvas.draw_idle()

    def update_model_description(self, event=None):
        self.model_description.set(MODELS.get(self.model_size.get(), ""))

    def update_input_label(self):
        mode = self.input_mode.get()
        if mode == "single":
            self.input_label_text.set("Select File:")
            self.batch_progress_label.grid_remove()
            self.batch_progress_bar.grid_remove()
        else:
            self.input_label_text.set("Select Dir:")
            self.batch_progress_label.grid(row=1, column=0, sticky="w")
            self.batch_progress_bar.grid(row=1, column=1, sticky="ew")

        if not self.processing_active:
            self.input_path.set("")
            self.file_tree.delete(*self.file_tree.get_children())
            self.file_data.clear()

    def update_compute_types(self, event=None):
        dev = self.device.get()
        vals = []
        if dev == "cuda":
            vals = COMPUTE_TYPES_CUDA
        elif dev == "rocm":
            vals = COMPUTE_TYPES_ROCM
        else:
            # CPU: Filter out float16 as it's not natively supported/efficient
            vals = [ct for ct in COMPUTE_TYPES_CPU if ct != "float16"]

        self.compute_combo['values'] = vals

        # Reset to auto if current selection is invalid for new device
        current = self.compute_type.get()
        if current not in vals:
            self.compute_type.set("auto" if "auto" in vals else (vals[0] if vals else "default"))

    def browse_input(self):
        if self.processing_active: return
        mode = self.input_mode.get()
        if mode == "single":
            fp = filedialog.askopenfilename(filetypes=[("Media", " ".join([e[1:] for e in SUPPORTED_EXTENSIONS]))])
            if fp:
                self.input_path.set(fp)
                self.populate_file_list()
                AudioProcessor.plot_spectrogram_thread(fp, self.canvas, self.ax, self.log, self.clear_plot, self.spectrogram_line_ref)
        else:
            dp = filedialog.askdirectory()
            if dp:
                self.input_path.set(dp)
                self.populate_file_list()

    def populate_file_list(self):
        self.file_tree.delete(*self.file_tree.get_children())
        self.file_data.clear()
        path = self.input_path.get()
        if not path or not os.path.exists(path): return

        files = []
        if os.path.isfile(path): files = [path]
        else:
            for ext in SUPPORTED_EXTENSIONS:
                files.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))

        files = sorted(list(set(files)))
        self.total_batch_files = len(files)

        for f in files:
            fname = os.path.basename(f)
            ftype = "Video" if any(fname.lower().endswith(e[1:]) for e in VIDEO_EXTENSIONS) else "Audio"
            iid = self.file_tree.insert("", "end", values=(fname, ftype, STATUS_PENDING, "0%"))
            self.file_data[f] = {"id": iid, "status": STATUS_PENDING}

    def handle_drop(self, event):
        if self.processing_active: return
        path = event.data.strip('{}')
        if os.path.isdir(path):
            self.input_mode.set("batch")
            self.input_path.set(path)
        else:
            self.input_mode.set("single")
            self.input_path.set(path)
        self.populate_file_list()
        if os.path.isfile(path):
             AudioProcessor.plot_spectrogram_thread(path, self.canvas, self.ax, self.log, self.clear_plot, self.spectrogram_line_ref)

    def toggle_processing(self):
        if not self.processing_active:
            self.start_processing()
        else:
            self.request_stop()

    def start_processing(self):
        files = list(self.file_data.keys())
        if not files: return

        self.processing_active = True
        self.stop_requested = 0
        self.processed_batch_files = 0
        self.start_stop_button.config(text="🛑 Stop", style="Stop.TButton")
        self.processing_thread = threading.Thread(target=self.run_processing_loop, args=(files,), daemon=True)
        self.processing_thread.start()

    def request_stop(self):
        if self.stop_requested == 0:
            if messagebox.askyesno("Stop", "Stop after current file?"):
                self.stop_requested = 1
        elif self.stop_requested == 1:
            if messagebox.askyesno("Stop Immediate", "Force stop now?"):
                self.stop_requested = 2

    def run_processing_loop(self, files):
        model_size = self.model_size.get()
        device = self.device.get()

        # Auto-detect compute type
        compute = "int8"
        if device == "cuda":
            compute = "float16" # or 'auto' if faster-whisper supports it robustly, but float16 is standard for GPU

        # Quantization is now implied by model choice or handled internally if model name has suffix
        # But we removed manual selection. Standard models don't have suffix in name usually unless 'distil-large-v2' etc.
        # We just use model_size directly.
        full_model = model_size

        engine = TranscriptionEngine(full_model, device, compute, MODEL_DOWNLOAD_DIR)

        try:
            self.log(f"Loading Model: {full_model}...")
            engine.load_model()

            with log_context(batch_id=str(int(time.time()))):
                for i, fp in enumerate(files):
                    if self.stop_requested == 2: break
                    if self.stop_requested == 1 and i > 0: break

                    if i > 0 and i % 5 == 0:
                        gc.collect()
                        if device == "cuda": torch.cuda.empty_cache()

                    self.processed_batch_files = i + 1
                    self.process_single_file(engine, fp)

                    # Update Batch Progress
                    prog = (i + 1) / len(files) * 100
                    self.master.after(0, lambda v=prog: self.batch_progress_bar.configure(value=v))

            self.log("Batch processing complete.")

        except Exception as e:
            self.log(f"Critical Error: {e}")
        finally:
            engine.close()
            self.processing_active = False
            self.master.after(0, lambda: self.start_stop_button.config(text="🚀 Start Transcription", style="Accent.TButton"))

    def process_single_file(self, engine, filepath):
        fname = os.path.basename(filepath)
        iid = self.file_data[filepath]["id"]

        # Update UI
        self.master.after(0, lambda: self.file_tree.set(iid, "status", STATUS_PROCESSING))
        self.master.after(0, lambda: self.file_progress_label.config(text=f"Processing: {fname}"))

        try:
            is_video = any(fname.lower().endswith(e[1:]) for e in VIDEO_EXTENSIONS)
            is_audio = not is_video

            # Transcription Task
            task = "transcribe"
            if self.translate_output.get():
                task = "translate" # Force translation to English

            # Transcribe
            self.log(f"Transcribing {fname} (Task: {task})...")
            start_t = time.time()
            segments, info = engine.transcribe(filepath, task=task, beam_size=self.beam_size.get(), vad_filter=self.vad_filter.get())

            # Collect segments
            segment_list = []
            dur = info.duration
            for seg in segments:
                if self.stop_requested == 2: raise InterruptedError("Stopped")
                segment_list.append(seg)
                # Update progress bar
                if dur > 0:
                    pct = (seg.end / dur) * 100
                    self.master.after(0, lambda v=pct: self.file_progress_bar.configure(value=v))
                    self.master.after(0, lambda v=pct: self.file_tree.set(iid, "progress", f"{v:.1f}%"))

            # Save Output
            base_path = os.path.splitext(filepath)[0]
            if is_video:
                srt = segments_to_srt(segment_list)
                with open(base_path + ".srt", "w", encoding="utf-8") as f: f.write(srt)
                self.log(f"Saved SRT: {base_path}.srt")
            else:
                lrc = segments_to_lrc(segment_list)
                with open(base_path + ".lrc", "w", encoding="utf-8") as f: f.write(lrc)
                self.log(f"Saved LRC: {base_path}.lrc")

            # Metadata (Audio Only)
            if is_audio:
                self.log("Analyzing audio metadata...")
                bpm = AudioProcessor.calculate_bpm(filepath)
                full_text = segments_to_txt(segment_list)
                moods = MoodAnalyzer.get_moods_from_text(full_text)
                MetadataHandler.update_audio_metadata(filepath, bpm, moods)
                self.log(f"Tagged: BPM={bpm}, Moods={moods}")

            self.master.after(0, lambda: self.file_tree.set(iid, "status", STATUS_COMPLETED))

        except Exception as e:
            self.log(f"Error processing {fname}: {e}")
            self.master.after(0, lambda: self.file_tree.set(iid, "status", STATUS_ERROR))
        finally:
            self.master.after(0, lambda: self.file_progress_bar.configure(value=0))

    def on_closing(self):
        if self.processing_active:
            if messagebox.askyesno("Exit", "Stop processing and exit?"):
                self.stop_requested = 2
                self.master.destroy()
        else:
            self.master.destroy()

    def open_model_folder(self):
        try:
            if sys.platform == "win32": os.startfile(MODEL_DOWNLOAD_DIR)
            elif sys.platform == "darwin": subprocess.Popen(["open", MODEL_DOWNLOAD_DIR])
            else: subprocess.Popen(["xdg-open", MODEL_DOWNLOAD_DIR])
        except Exception as e:
            self.log(f"Error opening folder: {e}")

if __name__ == "__main__":
    try:
        root = TkinterDnD.Tk()
        app = WhisperGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Fatal Error: {e}")
