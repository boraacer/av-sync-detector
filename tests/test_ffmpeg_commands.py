from avsync_detector.live import LiveOptions, build_av_ffmpeg_cmd


def test_combined_ffmpeg_command_is_realtime_paced_before_input():
    options = LiveOptions()

    cmd = build_av_ffmpeg_cmd("http://example.test/source", options, audio_fd=3, video_fd=4)

    assert cmd[cmd.index("-i") - 1] == "-re"


def test_combined_ffmpeg_command_normalizes_quoted_shell_escapes_in_urls():
    options = LiveOptions()
    bad_url = "https://example.test/live/index.m3u8?token\\=abc\\&edge\\=fl1004"

    cmd = build_av_ffmpeg_cmd(bad_url, options, audio_fd=3, video_fd=4)

    assert cmd[cmd.index("-i") + 1] == "https://example.test/live/index.m3u8?token=abc&edge=fl1004"


def test_combined_ffmpeg_command_uses_one_input_for_audio_and_video_outputs():
    options = LiveOptions()

    cmd = build_av_ffmpeg_cmd("https://example.test/live/index.m3u8?token\\=abc", options, audio_fd=3, video_fd=4)

    assert cmd.count("-i") == 1
    assert cmd[cmd.index("-i") + 1] == "https://example.test/live/index.m3u8?token=abc"
    assert "pipe:3" in cmd
    assert "pipe:4" in cmd
    assert cmd.count("-map") == 2
