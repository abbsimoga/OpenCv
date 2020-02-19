vidID = "iRYZjOuUnlU"
from youtube_transcript_api import YouTubeTranscriptApi
transcript = YouTubeTranscriptApi.get_transcript(vidID)
writenscript = ""
for script in transcript:
    writenscript += script["text"] + "\n"

f=open("opinionfromYT.txt", "a+")
f.write(writenscript)
f.close()
# with open("opinionfromYT.txt","w") as f:
#     f.write(writenscript)
#     f.close()