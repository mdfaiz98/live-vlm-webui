"""Real unit tests for GPU monitor - matches actual implementation."""


class TestGPUMonitorFactory:
    """Test the GPU monitor factory function."""

    def test_create_monitor_returns_monitor(self):
        """Test that create_monitor returns a GPUMonitor instance."""
        from live_vlm_webui.gpu_monitor import create_monitor, GPUMonitor

        # This will auto-detect or fall back to NVMLMonitor
        monitor = create_monitor()

        assert monitor is not None
        assert isinstance(monitor, GPUMonitor)
        print(f"✅ Got monitor: {type(monitor).__name__}")

    def test_create_monitor_apple_silicon(self):
        """Test creating Apple Silicon monitor."""
        from live_vlm_webui.gpu_monitor import create_monitor, AppleSiliconMonitor

        monitor = create_monitor(platform="apple")

        assert monitor is not None
        assert isinstance(monitor, AppleSiliconMonitor)
        print("✅ Created AppleSiliconMonitor")

    def test_get_stats_returns_dict(self):
        """Test that get_stats returns a dictionary."""
        from live_vlm_webui.gpu_monitor import create_monitor

        monitor = create_monitor()
        stats = monitor.get_stats()

        assert isinstance(stats, dict)
        assert "platform" in stats
        assert "cpu_percent" in stats
        assert "ram_used_gb" in stats
        print(f"✅ Stats keys: {list(stats.keys())}")


class TestGetCPUModel:
    """Test the get_cpu_model utility function."""

    def test_get_cpu_model_returns_string(self):
        """Test that get_cpu_model returns a string."""
        from live_vlm_webui.gpu_monitor import get_cpu_model

        cpu_model = get_cpu_model()

        assert isinstance(cpu_model, str)
        assert len(cpu_model) > 0
        print(f"✅ CPU Model: {cpu_model}")


class TestQualcommMonitor:
    """QualcommMonitor busy-% math (GPU from DRM fdinfo, NPU from app-side busy time)."""

    def test_busy_percentages(self, tmp_path, monkeypatch):
        from live_vlm_webui import gpu_monitor

        (tmp_path / "cur_freq").write_text("405000000\n")
        (tmp_path / "max_freq").write_text("800000000\n")
        mon = gpu_monitor.QualcommMonitor(str(tmp_path))

        clock = {"t": 100.0}
        monkeypatch.setattr(gpu_monitor.time, "monotonic", lambda: clock["t"])
        gpu = {"a": 0, "b": 0}
        monkeypatch.setattr(mon, "_gpu_busy_ns", lambda: dict(gpu))
        npu = {"s": 0.0}
        mon.npu_busy_source = lambda: npu["s"]

        mon.get_stats()  # first sample only sets the baseline
        clock["t"] += 0.5
        gpu["a"] += 150_000_000  # two clients busy 0.15 s + 0.1 s in 0.5 s -> 50%
        gpu["b"] += 100_000_000
        gpu["c"] = 9_000_000_000  # new client: its lifetime total must not count
        npu["s"] += 0.2  # 0.2 s of NPU inference in 0.5 s -> 40%
        stats = mon.get_stats()

        assert abs(stats["gpu_percent"] - 50.0) < 1e-6
        assert abs(stats["npu_percent"] - 40.0) < 1e-6

        # Detection + a VLM request together can exceed the interval: capped at 100%
        clock["t"] += 0.5
        npu["s"] += 0.9
        assert mon.get_stats()["npu_percent"] == 100.0
        assert stats["gpu_freq_mhz"] == 405.0
        assert stats["gpu_max_freq_mhz"] == 800.0
        mon.update_history(stats)
        history = mon.get_history()
        assert history["npu_util"] == [stats["npu_percent"]]

    def test_no_busy_sources_reports_none(self, tmp_path):
        from live_vlm_webui import gpu_monitor

        mon = gpu_monitor.QualcommMonitor(str(tmp_path))  # no devfreq files -> clock None
        stats = mon.get_stats()
        assert stats["npu_percent"] is None
        assert stats["gpu_freq_mhz"] is None

    def test_board_name_regex(self):
        import re

        pat = r"\b(AIR|ASR|AFE)\s*-?\s*([A-Z]?\d+)"
        for model, expected in [
            ("Advantech Technologies, Inc. Addons AFEA503 A1", "AFE-A503"),
            ("Advantech Technologies, Inc. Addons AIR055 A1", "AIR-055"),
            ("Advantech ASRD501-A1 Platform", "ASR-D501"),
        ]:
            m = re.search(pat, model, re.IGNORECASE)
            assert f"{m.group(1).upper()}-{m.group(2).upper()}" == expected


class TestVLMBusySeconds:
    def test_counts_inflight_and_finished_requests(self, monkeypatch):
        from live_vlm_webui import vlm_service
        from live_vlm_webui.vlm_service import VLMService

        clock = {"t": 10.0}
        monkeypatch.setattr(vlm_service.time, "perf_counter", lambda: clock["t"])
        monkeypatch.setattr(VLMService, "_busy_done_seconds", 2.0)
        monkeypatch.setattr(VLMService, "_inflight_since", {1: 9.0, 2: 9.5})
        # 2 s finished + 1 s and 0.5 s still in flight
        assert abs(VLMService.busy_seconds() - 3.5) < 1e-9
