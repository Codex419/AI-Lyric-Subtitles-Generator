import os
import re

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

        # 1. Artist and Title from Filename (Artist - Title.ext)
        base_name, _ = os.path.splitext(filename)
        if " - " in base_name:
            parts = base_name.split(" - ", 1)
            artist = parts[0].strip()
            title = parts[1].strip()
        else:
            return None, None, None

        # 2. Album from Parent Directory (Album [Year])
        # Remove [Year] suffix
        album = re.sub(r'\s*\[.*?\]\s*$', '', album_dir_name)
        # Also remove (Year) just in case
        album = re.sub(r'\s*\(.*?\)\s*$', '', album).strip()

        return artist, title, album

    except Exception as e:
        print(f"Error parsing path: {e}")
        return None, None, None

# Test Cases
test_paths = [
    "/home/user/Music/The Beatles/Abbey Road [1969]/The Beatles - Come Together.mp3",
    "C:\\Users\\Music\\Daft Punk\\Discovery [2001]\\Daft Punk - One More Time.flac",
    "/Music/Artist Name/Album Name/Artist Name - Song Title.wav",
    "/Music/No Year/Just Album/Artist - Title.mp3",
    "/Music/Complex/Album [2022] [Deluxe]/Artist - Title with - Hyphen.mp3"
]

print("--- Testing Metadata Extraction ---")
for path in test_paths:
    artist, title, album = extract_metadata_from_path(path)
    print(f"Path: {path}")
    print(f"  Artist: '{artist}'")
    print(f"  Title:  '{title}'")
    print(f"  Album:  '{album}'")
    print("-" * 20)
