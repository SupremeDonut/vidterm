use crossterm::event::{Event, KeyCode, poll, read};
use crossterm::style::{Color, Colors, Print, SetColors};
use crossterm::{
    cursor, execute, queue,
    terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use ctrlc;
use std::env;
use std::io::{self, Read, Write};
use std::process::{Command, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::{
    thread,
    time::{Duration, Instant},
};
mod filters;
use filters::*;

fn get_video_input() -> (Option<String>, u16) {
    let default_frame_rate = 15;
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} input video [<frame rate = 15>]", args[0]);
        return (None, default_frame_rate);
    }
    let input = &args[1];
    let frame_rate = {
        if args.len() >= 3 {
            str::parse(&args[2]).unwrap_or(default_frame_rate)
        } else {
            default_frame_rate
        }
    };
    (Some(input.to_string()), frame_rate)
}

fn get_video_size(video: &str) -> (usize, usize) {
    let out = Command::new("ffprobe")
        .args(&[
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            video,
        ])
        .output()
        .unwrap();

    let dims = String::from_utf8(out.stdout).unwrap();
    let parts: Vec<&str> = dims.lines().next().unwrap().trim().split(',').collect();
    (parts[0].parse().unwrap(), parts[1].parse().unwrap())
}

fn get_new_size(size: (usize, usize), term_size: (u16, u16)) -> (usize, usize) {
    let width_scale = (term_size.0 as f32) / (size.0 as f32);
    let height_scale = (2.0 * term_size.1 as f32) / (size.1 as f32);
    let scale_factor = width_scale.min(height_scale);
    let new_width = (size.0 as f32 * scale_factor) as usize;
    let new_height = (size.1 as f32 * scale_factor) as usize;
    (new_width, new_height)
}

fn render_frame(frame: &Vec<u8>, size: (usize, usize)) -> io::Result<()> {
    let mut stdout = io::stdout();
    queue!(stdout, cursor::MoveTo(0, 0))?;
    for y in (0..size.1).step_by(2) {
        queue!(stdout, cursor::MoveTo(0, (y / 2) as u16))?;
        for x in 0..size.0 {
            let top_pixel = get_color(frame, size, (x, y));
            let bottom_pixel = get_color(frame, size, (x, y + 1));
            queue!(
                stdout,
                SetColors(Colors::new(top_pixel, bottom_pixel)),
                Print("▀".to_string())
            )?;
        }
        queue!(stdout, crossterm::style::ResetColor)?;
    }
    stdout.flush()
}

fn get_color(frame: &Vec<u8>, size: (usize, usize), pos: (usize, usize)) -> Color {
    if pos.1 >= size.1 {
        return Color::Reset;
    }

    let idx = (pos.1 * size.0 + pos.0) * 4;
    return Color::Rgb {
        r: frame[idx],
        g: frame[idx + 1],
        b: frame[idx + 2],
    };
}

fn main() -> io::Result<()> {
    let input = get_video_input();
    if input.0.is_none() {
        return Ok(());
    }
    let video = &input.0.unwrap();
    let frame_rate = &input.1;
    let size = get_video_size(video);
    let frame_size = size.0 * size.1 * 4;
    let mut filter = FilterType::Gaussian;
    let mut strength = 2f32;

    let mut child = Command::new("ffmpeg")
        .args(&[
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video,
            "-r",
            &frame_rate.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    let mut out = child.stdout.take().unwrap();
    let mut chunk = vec![0; frame_size];
    let mut paused = false;
    let finished = Arc::new(AtomicBool::new(false));
    let f = finished.clone();
    let mut skipped = true;

    ctrlc::set_handler(move || {
        // make sure sigint is as graceful as possible
        f.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    execute!(io::stdout(), EnterAlternateScreen, cursor::Hide)?;
    let mut start = Instant::now();
    let frame_delay = Duration::from_millis((1000 / frame_rate) as u64);
    let mut term_size = terminal::size().unwrap();
    let mut frame = 0;

    let compute = Compute::new();

    while !finished.load(Ordering::SeqCst) {
        let mut redraw = !paused;
        if poll(Duration::from_millis(1))? {
            match read()? {
                Event::Key(event) => {
                    if event.is_press() {
                        match event.code {
                            KeyCode::Char(' ') => {
                                paused = !paused;
                                if !paused {
                                    frame = 0;
                                    start = Instant::now();
                                }
                            }
                            KeyCode::Char('q') => finished.store(true, Ordering::SeqCst),
                            KeyCode::Char('1') => filter = FilterType::Nearest,
                            KeyCode::Char('2') => filter = FilterType::Bilinear,
                            KeyCode::Char('3') => filter = FilterType::Gaussian,
                            KeyCode::Char('4') => filter = FilterType::Lanczos,
                            KeyCode::Char('5') => filter = FilterType::Box,
                            KeyCode::Char('[') => strength = (strength - 0.5).max(1.0),
                            KeyCode::Char(']') => strength += 0.5,
                            _ => {}
                        }
                    }
                }
                Event::Resize(width, height) => {
                    term_size = (width, height);
                    execute!(io::stdout(), Clear(ClearType::All))?;
                    redraw = true;
                }
                _ => {}
            }
        }

        if !paused && !out.read_exact(&mut chunk).is_ok() {
            paused = true;
            continue;
        }

        let now = Instant::now();
        let expected_frame = (now - start).as_millis() / frame_delay.as_millis();
        if frame + 1 < expected_frame {
            frame += 1;
            skipped = true;
            continue;
        }

        if !redraw && !skipped {
            continue;
        }

        let new_size = get_new_size(size, term_size);
        let resized = compute.resample_gpu(&chunk, size, new_size, filter, strength);
        render_frame(&resized, new_size)?;

        let now = Instant::now();
        let expected_frame = (now - start).as_millis() / frame_delay.as_millis();
        if frame > expected_frame {
            thread::sleep(frame_delay * (frame - expected_frame) as u32);
        }
        frame += 1;
        skipped = false;
    }
    execute!(io::stdout(), LeaveAlternateScreen)?;
    child.kill().ok();
    child.wait().ok();
    Ok(())
}
