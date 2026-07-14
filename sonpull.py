import os
import io
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Read-only access is enough since we're only searching/downloading
SCOPES = [""]

# MIME types we treat as "songs"
AUDIO_MIME_TYPES = [
    "audio/mpeg",  # .mp3
    "audio/wav",
    "audio/x-wav",
    "audio/flac",
    "audio/mp4",  # .m4a
    "audio/ogg"
    "audio/wav",
]


def _get_drive_service():
    """Authenticate with Google (cached after first run) and return a Drive API client."""
    creds = None
    token_path = "token.pickle"

    if os.path.exists(token_path):
        with open(token_path, "rb") as token_file:
            creds = pickle.load(token_file)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "wb") as token_file:
            pickle.dump(creds, token_file)

    return build("drive", "v3", credentials=creds)


def load_song(keyword, download_dir="downloaded_songs", max_results=5):
    """
    Search Google Drive for songs matching a keyword and download them locally.

    Args:
        keyword (str): Text to match against file names, e.g. "bohemian rhapsody".
        download_dir (str): Local folder to save matches into.
        max_results (int): Max number of matching files to download.

    Returns:
        list[str]: Local file paths of the downloaded songs.
    """
    service = _get_drive_service()

    mime_filter = " or ".join(f"mimeType='{m}'" for m in AUDIO_MIME_TYPES)
    query = f"name contains '{keyword}' and ({mime_filter}) and trashed=false"

    results = service.files().list(
        q=query,
        pageSize=max_results,
        fields="files(id, name, mimeType, size)",
    ).execute()

    files = results.get("files", [])

    if not files:
        print(f"No songs found matching '{keyword}'.")
        return []

    os.makedirs(download_dir, exist_ok=True)
    downloaded_paths = []

    for file in files:
        file_id = file["id"]
        file_name = file["name"]
        local_path = os.path.join(download_dir, file_name)

        print(f"Downloading '{file_name}'...")
        request = service.files().get_media(fileId=file_id)

        with io.FileIO(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        downloaded_paths.append(local_path)
        print(f"Saved to {local_path}")

    return downloaded_paths


if __name__ == "__main__":
    song_keyword = input("Enter a song name or keyword to search for: ")
    paths = load_song(song_keyword)

    if paths:
        print("\nDownloaded songs:")
        for p in paths:
            print(f" - {p}")
