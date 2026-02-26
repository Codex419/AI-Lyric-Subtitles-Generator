import librosa
import numpy as np
import threading
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import librosa.display

class AudioProcessor:
    @staticmethod
    def calculate_bpm(audio_path):
        """Calculates BPM using librosa."""
        try:
            y, sr = librosa.load(audio_path, sr=None, duration=120) # Analyze first 2 mins max for speed
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            # tempo can be a scalar or an array depending on librosa version/method
            if np.ndim(tempo) > 0: tempo = tempo[0]
            return int(round(tempo))
        except Exception as e:
            print(f"BPM Error: {e}")
            return 0

    @staticmethod
    def plot_spectrogram_thread(audio_path, canvas, ax, log_callback, clear_plot_callback, line_holder):
        """
        Internal method to generate spectrogram in a thread and update the provided canvas/ax.
        line_holder is a mutable list/dict to store the line object reference.
        """
        try:
            log_callback(f"   Generating spectrogram for {os.path.basename(audio_path)}...")
            try: y, sr = librosa.load(audio_path, sr=None, mono=True)
            except Exception as load_err:
                log_callback(f"   ❌ Error loading audio: {load_err}")
                if "audioread" in str(load_err) or "backend" in str(load_err):
                    log_callback("      (Hint: FFmpeg might be missing or not in PATH for video files.)")
                clear_plot_callback()
                return
            if y is None or len(y) == 0:
                log_callback("   ⚠️ Audio could not be loaded or is empty.")
                clear_plot_callback()
                return
            try:
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)
            except Exception as spec_err:
                log_callback(f"   ❌ Error computing spectrogram: {spec_err}")
                clear_plot_callback()
                return

            def _do_plot_main_thread():
                if not ax or not canvas: return
                try:
                    ax.clear()
                    librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
                    ax.set_title(f'Mel Spectrogram: {os.path.basename(audio_path)}', color="#e0e0e0", fontsize=9)
                    ax.set_xlabel("Time (s)", color="#e0e0e0", fontsize=8)
                    ax.set_ylabel("Frequency (Mel)", color="#e0e0e0", fontsize=8)

                    # Add line
                    line = Line2D([0, 0], [0, 1], transform=ax.get_xaxis_transform(), color='red', linestyle='--', linewidth=1, visible=False)
                    ax.add_line(line)
                    line_holder[0] = line # Update reference

                    canvas.draw_idle()
                except Exception as plot_err:
                    log_callback(f"   ❌ Error during plotting: {plot_err}")
                    clear_plot_callback()

            # Schedule on main thread (assuming canvas.get_tk_widget().master is available)
            if canvas.get_tk_widget().winfo_exists():
                canvas.get_tk_widget().after(0, _do_plot_main_thread)

            log_callback("   ✅ Spectrogram generated.")
        except Exception as e:
            log_callback(f"   ❌ Unexpected error in spectrogram thread: {e}")
            clear_plot_callback()
