import subprocess
from pydub import AudioSegment

mix_youtube_link = input("What is the YouTube link for the mix?\n > ")
subprocess.Popen(["yt-dlp", mix_youtube_link, "-x", "--audio-format", "mp3", "-o", "mix.mp3"]).wait()

original_mp3 = AudioSegment.from_mp3("mix.mp3")

five_minutes = 5*60*1000
slice_count = len(original_mp3) // five_minutes
slice_length = len(original_mp3) // slice_count

mix_title = input("Title > ")
mix_artist = input("Artist > ")
mix_year = input("Year > ")

for i in range(slice_count):
    start = i*slice_length
    end = (i+1)*slice_length - 1
    
    slice_title = "Part-" + str(i) + ".mp3"
    original_mp3[start:end].export(slice_title,
                                   format="mp3",
                                   tags={"TITLE": slice_title,
                                         "ALBUM": mix_title,
                                         "ALBUMARTIST": mix_artist,
                                         "YEAR": mix_year,
                                         "GENRE": "DJ Mix",
                                         "TRACK": str(i+1),
                                         "COMMENT": "NOTE: This is YouTube audio and part of a larger mix."})
