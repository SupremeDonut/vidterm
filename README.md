# a video viewer in the terminal

## requirements
- Rust
- FFmpeg

## how to use
1. compile with `cargo build`
2. command line arguments: `vidterm file [<fps=15>]`
3. options:
 - Ctrl-C or q to exit
 - Space to pause/unpause
 - 1-5 to change resampling algorithm
 - [ and ] to decrease/increase strength (if applicable)

## to use live streams
to get a live stream running an HLS URL is required
for Twitch:
- copy the twitch stream link
- use https://pwn.sh/tools/getstream.html to get the link

for Youtube:
- install yt-dlp
- copy the youtube stream link
- get the formats of the video: `yt-dlp --list-formats link`
- find the format that you want
- get the url from that format: `yt-dlp -f format -g link`

