import os
import re
import mutagen
from mutagen.id3 import ID3, TMOO, TBPM, TXXX, TIT2, TPE1, TALB, ID3NoHeaderError
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

class MetadataHandler:
    @staticmethod
    def extract_metadata_from_path(filepath):
        """
        Extracts Artist, Title, and Album from filepath.
        Pattern: .../Artist/Album [Year]/Artist - Title.ext
        """
        try:
            # Normalize path separators for robust handling across OS
            filepath = filepath.replace('\\', '/')
            filename = os.path.basename(filepath)
            parent_dir = os.path.dirname(filepath)
            album_dir_name = os.path.basename(parent_dir)

            artist = None; title = None; album = None

            # 1. Artist and Title from Filename (Artist - Title.ext)
            base_name, _ = os.path.splitext(filename)
            if " - " in base_name:
                parts = base_name.split(" - ", 1)
                artist = parts[0].strip()
                title = parts[1].strip()

            # 2. Album from Parent Directory (Album [Year])
            if album_dir_name:
                # Remove [Year] suffix
                album = re.sub(r'\s*\[.*?\]\s*$', '', album_dir_name)
                # Also remove (Year) just in case
                album = re.sub(r'\s*\(.*?\)\s*$', '', album).strip()

            return artist, title, album

        except Exception as e:
            print(f"Path Parsing Error: {e}")
            return None, None, None

    @staticmethod
    def update_audio_metadata(filepath, bpm, moods):
        """Updates audio file metadata with BPM, Moods, and extracted info (Mutagen)."""
        try:
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()

            # Extract Metadata from Path
            path_artist, path_title, path_album = MetadataHandler.extract_metadata_from_path(filepath)

            if ext == ".mp3":
                try: audio = ID3(filepath)
                except ID3NoHeaderError: audio = ID3(); audio.save(filepath) # Create if missing

                # Moods (TMOO) - ID3v2.4
                existing_moods = []
                if "TMOO" in audio:
                    existing_moods = audio["TMOO"].text

                all_moods = []
                for m in existing_moods: all_moods.append(str(m))
                for m in moods:
                    if m not in all_moods: all_moods.append(str(m))

                final_moods = all_moods[:7] # Enforce Max 7
                while len(final_moods) < 3: final_moods.append("Unknown")

                audio["TMOO"] = TMOO(encoding=3, text=final_moods)
                audio["TXXX:MOOD"] = TXXX(encoding=3, desc="MOOD", text=final_moods) # Fallback

                # BPM
                if bpm > 0:
                    audio["TBPM"] = TBPM(encoding=3, text=str(bpm))

                # Path Metadata
                if path_title: audio["TIT2"] = TIT2(encoding=3, text=path_title)
                if path_artist: audio["TPE1"] = TPE1(encoding=3, text=path_artist)
                if path_album: audio["TALB"] = TALB(encoding=3, text=path_album)

                audio.save()

            elif ext == ".flac":
                audio = FLAC(filepath)

                # Moods
                current_moods = audio.get("MOOD", [])
                all_moods = list(current_moods)
                for m in moods:
                    if m not in all_moods: all_moods.append(m)

                final_moods = all_moods[:7]
                audio["MOOD"] = final_moods # List is fine for FLAC tags

                if bpm > 0: audio["BPM"] = [str(bpm)] # Wrap in list

                # Path Metadata (Vorbis)
                if path_title: audio["TITLE"] = [path_title] # Wrap in list
                if path_artist: audio["ARTIST"] = [path_artist] # Wrap in list
                if path_album: audio["ALBUM"] = [path_album] # Wrap in list

                audio.save()

            elif ext == ".ogg":
                audio = OggVorbis(filepath)
                 # Moods
                current_moods = audio.get("MOOD", [])
                all_moods = list(current_moods)
                for m in moods:
                    if m not in all_moods: all_moods.append(m)

                final_moods = all_moods[:7]
                audio["MOOD"] = final_moods # List is fine for Ogg tags

                if bpm > 0: audio["BPM"] = [str(bpm)] # Wrap in list

                # Path Metadata (Vorbis)
                if path_title: audio["TITLE"] = [path_title] # Wrap in list
                if path_artist: audio["ARTIST"] = [path_artist] # Wrap in list
                if path_album: audio["ALBUM"] = [path_album] # Wrap in list

                audio.save()

            elif ext == ".wav":
                 audio = WAVE(filepath)
                 if audio.tags is None: audio.add_tags()

                 # Use ID3 tags for WAV
                 existing_moods = []
                 if "TMOO" in audio.tags: existing_moods = audio.tags["TMOO"].text

                 all_moods = [str(m) for m in existing_moods]
                 for m in moods:
                     if m not in all_moods: all_moods.append(str(m))

                 final_moods = all_moods[:7]

                 audio.tags["TMOO"] = TMOO(encoding=3, text=final_moods)
                 audio.tags["TXXX:MOOD"] = TXXX(encoding=3, desc="MOOD", text=final_moods)
                 if bpm > 0:
                     audio.tags["TBPM"] = TBPM(encoding=3, text=str(bpm))

                 # Path Metadata
                 if path_title: audio.tags["TIT2"] = TIT2(encoding=3, text=path_title)
                 if path_artist: audio.tags["TPE1"] = TPE1(encoding=3, text=path_artist)
                 if path_album: audio.tags["TALB"] = TALB(encoding=3, text=path_album)

                 audio.save()

            return True
        except Exception as e:
            print(f"Metadata Error: {e}")
            return False
