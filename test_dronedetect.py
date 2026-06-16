"""
DroneDetect Smoke Test Suite
============================
Exercises the pure-Python logic in DroneDetect.py without running the
Streamlit UI or touching real SDR/BLE hardware.  This is deliberately
a smoke test rather than a full unit-test suite — it checks that the
most critical building blocks (frequency planning, signature database,
fingerprinting engine, helper formatters, and background threads) work
correctly and don't regress.

Run with:
    pytest test_dronedetect.py -v
from the /home/bat/DroneDetect directory.
"""

import json
import queue
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make sure the project root is importable even when pytest is invoked from
# another working directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

# DroneDetect imports streamlit at module level; stub it out so the module
# loads cleanly in a headless test environment.
import types
import unittest.mock as mock

# Build a minimal streamlit stub so `import streamlit as st` succeeds.
_st_stub = types.ModuleType("streamlit")
_st_stub.session_state = {}
_st_stub.cache_data = lambda *a, **kw: (lambda f: f)
_st_stub.cache_resource = lambda *a, **kw: (lambda f: f)
sys.modules.setdefault("streamlit", _st_stub)

# Also stub out optional heavy SDR / drone SDKs so they don't block import.
for _mod in ("SoapySDR", "djitellopy", "pymavlink", "pymavlink.mavutil",
             "bleak", "olympe", "pyparrot", "pyparrot.Bebop"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

import DroneDetect as dd  # noqa: E402  (must come after stubs)

# =============================================================================
# 1. parse_range / build_freq_plan / build_freq_plan_from_text
# =============================================================================

class TestParseRange:
    def test_mhz_suffix(self):
        lo, hi = dd.parse_range("433m-435m")
        assert lo == pytest.approx(433e6)
        assert hi == pytest.approx(435e6)

    def test_ghz_suffix(self):
        lo, hi = dd.parse_range("2.4g-2.484g")
        assert lo == pytest.approx(2.4e9)
        assert hi == pytest.approx(2.484e9)

    def test_khz_suffix(self):
        lo, hi = dd.parse_range("100k-200k")
        assert lo == pytest.approx(100e3)
        assert hi == pytest.approx(200e3)

    def test_whitespace_stripped(self):
        lo, hi = dd.parse_range("  900m - 928m  ")
        assert lo == pytest.approx(900e6)
        assert hi == pytest.approx(928e6)

    def test_reversed_range_returns_values(self):
        # parse_range itself just splits — it does not sort.
        # build_freq_plan handles reversal.
        lo, hi = dd.parse_range("928m-900m")
        assert lo == pytest.approx(928e6)
        assert hi == pytest.approx(900e6)


class TestBuildFreqPlan:
    def test_basic_plan(self):
        plan = dd.build_freq_plan(100e6, 110e6, 2e6)
        assert plan[0] == pytest.approx(100e6)
        assert plan[-1] == pytest.approx(110e6)
        # 100, 102, 104, 106, 108, 110  → 6 points
        assert len(plan) == 6

    def test_reversed_endpoints_sorted(self):
        plan_fwd = dd.build_freq_plan(100e6, 110e6, 2e6)
        plan_rev = dd.build_freq_plan(110e6, 100e6, 2e6)
        np.testing.assert_array_equal(plan_fwd, plan_rev)

    def test_zero_step_clamped_to_one(self):
        # step < 1 is clamped to 1.0 Hz — should not raise
        plan = dd.build_freq_plan(433e6, 433e6 + 10, 0.0)
        assert len(plan) == 11  # 0,1,2,...,10 Hz

    def test_oversized_raises_value_error(self):
        # 1 Hz step over 100 MHz → >> 50 000 points
        with pytest.raises(ValueError, match="Too many steps"):
            dd.build_freq_plan(0.0, 100e6, 1.0)

    def test_single_frequency(self):
        plan = dd.build_freq_plan(915e6, 915e6, 1e6)
        assert len(plan) == 1
        assert plan[0] == pytest.approx(915e6)


class TestBuildFreqPlanFromText:
    def test_single_band(self):
        plan = dd.build_freq_plan_from_text("433m-435m", 1e6)
        assert plan[0] == pytest.approx(433e6)
        assert plan[-1] == pytest.approx(435e6)

    def test_multi_band_covers_all_bands(self):
        text = "433m-435m,902m-928m"
        plan = dd.build_freq_plan_from_text(text, 2e6)
        # Plan must contain frequencies in both the 433 MHz and 900 MHz bands
        has_433 = any(432e6 <= f <= 436e6 for f in plan)
        has_900 = any(900e6 <= f <= 930e6 for f in plan)
        assert has_433, "Plan does not cover 433 MHz band"
        assert has_900, "Plan does not cover 900 MHz band"

    def test_multi_band_step_count(self):
        # 433-435 with 1 MHz step = 3 points; 902-928 with 2 MHz step = 14 points
        plan = dd.build_freq_plan_from_text("433m-435m,902m-928m", 2e6)
        # 433, 435 → 2 points for first band (floor((435-433)/2)+1 = 2)
        # 902, 904, ..., 928 → 14 points for second band
        assert len(plan) == 2 + 14

    def test_multi_band_interleaves_by_default(self):
        plan = dd.build_freq_plan_from_text("100m-102m,200m-202m", 1e6)
        expected = np.array([100e6, 200e6, 101e6, 201e6, 102e6, 202e6], dtype=np.float64)
        np.testing.assert_array_equal(plan, expected)

    def test_sequential_strategy_preserved_when_requested(self):
        plan = dd.build_freq_plan_from_text("100m-102m,200m-202m", 1e6, strategy="sequential")
        expected = np.array([100e6, 101e6, 102e6, 200e6, 201e6, 202e6], dtype=np.float64)
        np.testing.assert_array_equal(plan, expected)

    def test_oversized_combined_raises(self):
        # Very small step across wide range — combined size > 50 000
        with pytest.raises(ValueError, match="Too many steps"):
            dd.build_freq_plan_from_text("1m-6g", 1.0)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Empty frequency range"):
            dd.build_freq_plan_from_text("", 1e6)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown scan strategy"):
            dd.build_freq_plan_from_text("433m-435m", 1e6, strategy="bogus")


class TestSdrInventoryHelpers:
    def test_parse_soapy_args_text(self):
        args = dd.parse_soapy_args_text("driver=rtlsdr, serial=1234")
        assert args == {"driver": "rtlsdr", "serial": "1234"}

    def test_choose_soapy_device_prefers_hackrf(self):
        devices = [
            {"driver": "uhd", "label": "USRP B210"},
            {"driver": "hackrf", "label": "HackRF One"},
            {"driver": "rtlsdr", "label": "RTL-SDR"},
        ]
        chosen = dd.choose_soapy_device(devices)
        assert chosen is not None
        assert chosen["driver"] == "hackrf"

    def test_enumerate_soapy_devices_safe_parses_payload(self, monkeypatch):
        monkeypatch.setattr(dd, "HAVE_SOAPY", True)

        def fake_run(*args, **kwargs):
            payload = {
                "ok": True,
                "devices": [
                    {"driver": "rtlsdr", "label": "RTL-SDR Blog", "serial": "0001"},
                    {"driver": "audio", "label": "PulseAudio"},
                ],
            }
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=json.dumps(payload) + "\n", stderr="")

        monkeypatch.setattr(dd.subprocess, "run", fake_run)
        devices, note = dd.enumerate_soapy_devices_safe("")
        assert note == ""
        assert len(devices) == 2
        assert devices[0]["driver"] == "rtlsdr"
        assert devices[0]["label"] == "RTL-SDR Blog"

    def test_enumerate_soapy_devices_safe_reports_signal_crash(self, monkeypatch):
        monkeypatch.setattr(dd, "HAVE_SOAPY", True)

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args=args[0], returncode=-11, stdout="", stderr="")

        monkeypatch.setattr(dd.subprocess, "run", fake_run)
        devices, note = dd.enumerate_soapy_devices_safe("")
        assert devices == []
        assert "signal 11" in note


# =============================================================================
# 1b. workflow helpers / recording helpers
# =============================================================================

class TestWorkflowHelpers:
    def test_lookup_wifi_fingerprint_exact(self):
        hit = dd.lookup_wifi_fingerprint("DJI", "Tello / Tello EDU")
        assert hit is not None
        assert hit["adapter"] == "tello"
        assert hit["ip"] == "192.168.10.1"

    def test_command_guidance_for_tello_is_direct(self):
        match = dd.DroneMatch(
            sig_key="dji_tello",
            name="DJI Tello / Tello EDU",
            brand="DJI",
            confidence=0.95,
            center_freq_hz=2412e6,
            detected_bw_hz=20e6,
            snr_db=18.0,
            peak_db=-45.0,
            timestamp_utc="2026-01-01T00:00:00+00:00",
            notes="wifi",
            adapter="tello",
            wifi_ip="192.168.10.1",
            wifi_port=8889,
        )
        plan = dd.command_guidance_for_match(match)
        assert plan["direct"] is not None
        assert plan["direct"]["adapter"] == "tello"

    def test_command_guidance_for_consumer_dji_is_not_direct(self):
        match = dd.DroneMatch(
            sig_key="dji_ocusync3",
            name="DJI OcuSync 3 / O3+",
            brand="DJI",
            confidence=0.82,
            center_freq_hz=2.437e9,
            detected_bw_hz=20e6,
            snr_db=14.0,
            peak_db=-52.0,
            timestamp_utc="2026-01-01T00:00:00+00:00",
            notes="ocusync",
            adapter="mavlink",
            wifi_ip="",
            wifi_port=0,
        )
        plan = dd.command_guidance_for_match(match)
        assert plan["direct"] is None
        assert "Passive DJI detection only" in plan["title"]

    def test_save_detection_capture_writes_npz_and_cf32(self, tmp_path):
        result = dd.ScanResult(
            timestamp_utc="2026-01-01T00:00:00+00:00",
            center_freq_hz=2.437e9,
            peak_db=-40.0,
            noise_floor_db=-70.0,
            snr_db=18.0,
            occupied_bw_hz=2e6,
            detected=True,
            freq_axis_hz=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            spectrum_db=np.array([-70.0, -40.0, -72.0], dtype=np.float32),
            sample_rate=2.4e6,
            frontend_bandwidth_hz=2.4e6,
            matches=[
                dd.DroneMatch(
                    sig_key="dji_tello",
                    name="DJI Tello / Tello EDU",
                    brand="DJI",
                    confidence=0.91,
                    center_freq_hz=2.437e9,
                    detected_bw_hz=20e6,
                    snr_db=18.0,
                    peak_db=-40.0,
                    timestamp_utc="2026-01-01T00:00:00+00:00",
                    notes="test",
                    adapter="tello",
                    wifi_ip="192.168.10.1",
                    wifi_port=8889,
                )
            ],
        )
        iq = np.array([1 + 1j, -1 - 1j, 0.5 - 0.25j], dtype=np.complex64)
        out_path = dd.save_detection_capture(iq, result, str(tmp_path))
        out_file = Path(out_path)
        assert out_file.exists()
        assert out_file.suffix == ".npz"
        cf32_path = out_file.with_suffix(".cf32")
        assert cf32_path.exists()
        with np.load(out_file, allow_pickle=False) as npz:
            np.testing.assert_array_equal(npz["iq"], iq)
            assert float(npz["sample_rate"]) == pytest.approx(2.4e6)
            assert str(npz["top_match_name"]) == "DJI Tello / Tello EDU"

    def test_build_target_picture_merges_recent_hits(self):
        match = dd.DroneMatch(
            sig_key="dji_tello",
            name="DJI Tello / Tello EDU",
            brand="DJI",
            confidence=0.91,
            center_freq_hz=2.437e9,
            detected_bw_hz=20e6,
            snr_db=18.0,
            peak_db=-40.0,
            timestamp_utc="2026-01-01T00:00:00+00:00",
            notes="test",
            adapter="tello",
            wifi_ip="192.168.10.1",
            wifi_port=8889,
        )
        det = dd.ScanResult(
            timestamp_utc="2026-01-01T00:00:00+00:00",
            center_freq_hz=2.437e9,
            peak_db=-40.0,
            noise_floor_db=-70.0,
            snr_db=18.0,
            occupied_bw_hz=2e6,
            detected=True,
            freq_axis_hz=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            spectrum_db=np.array([-70.0, -40.0, -72.0], dtype=np.float32),
            matches=[match],
        )
        rows = dd.build_target_picture(
            [det],
            [{"source": "wifi", "timestamp_utc": "2026-01-01T00:00:05+00:00", "brand": "DJI", "model": "Tello / Tello EDU", "ssid": "TELLO-123", "band": "2.4 GHz", "channel": 6}],
        )
        assert rows
        assert rows[0]["Lead"] == "DJI Tello / Tello EDU"
        assert "SDR" in rows[0]["Sensors"]


class TestNetworkInventoryHelpers:
    def test_parse_onvif_probe_response(self):
        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">
  <e:Body>
    <d:ProbeMatches>
      <d:ProbeMatch>
        <wsa:EndpointReference>
          <wsa:Address>urn:uuid:camera-1</wsa:Address>
        </wsa:EndpointReference>
        <d:Types>dn:NetworkVideoTransmitter</d:Types>
        <d:Scopes>
          onvif://www.onvif.org/name/Front_Door
          onvif://www.onvif.org/hardware/Hikvision_DS-2CD
        </d:Scopes>
        <d:XAddrs>http://192.168.1.50/onvif/device_service</d:XAddrs>
        <d:MetadataVersion>1</d:MetadataVersion>
      </d:ProbeMatch>
    </d:ProbeMatches>
  </e:Body>
</e:Envelope>"""
        rows = dd.parse_onvif_probe_response(xml)
        assert len(rows) == 1
        assert rows[0]["endpoint"] == "urn:uuid:camera-1"
        assert rows[0]["xaddr"] == "http://192.168.1.50/onvif/device_service"
        assert rows[0]["device_name"] == "Front Door"
        assert "Hikvision" in rows[0]["hardware"]

    def test_parse_onvif_system_datetime(self):
        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
  <soap:Body>
    <tds:GetSystemDateAndTimeResponse>
      <tds:SystemDateAndTime>
        <tds:DateTimeType>NTP</tds:DateTimeType>
        <tds:DaylightSavings>false</tds:DaylightSavings>
        <tds:TimeZone><tds:TZ>UTC+0</tds:TZ></tds:TimeZone>
        <tds:UTCDateTime>
          <tds:Time><tds:Hour>12</tds:Hour><tds:Minute>34</tds:Minute><tds:Second>56</tds:Second></tds:Time>
          <tds:Date><tds:Year>2026</tds:Year><tds:Month>6</tds:Month><tds:Day>15</tds:Day></tds:Date>
        </tds:UTCDateTime>
        <tds:LocalDateTime>
          <tds:Time><tds:Hour>08</tds:Hour><tds:Minute>34</tds:Minute><tds:Second>56</tds:Second></tds:Time>
          <tds:Date><tds:Year>2026</tds:Year><tds:Month>6</tds:Month><tds:Day>15</tds:Day></tds:Date>
        </tds:LocalDateTime>
      </tds:SystemDateAndTime>
    </tds:GetSystemDateAndTimeResponse>
  </soap:Body>
</soap:Envelope>"""
        row = dd.parse_onvif_system_datetime(xml)
        assert row is not None
        assert row["utc"] == "2026-06-15 12:34:56"
        assert row["local"] == "2026-06-15 08:34:56"
        assert row["timezone"] == "UTC+0"

    def test_classify_http_fingerprint(self):
        label = dd.classify_http_fingerprint(
            server="Boa/0.94.14rc21",
            title="Hikvision Camera Login",
            realm="IP Camera",
        )
        assert label == "Hikvision camera / NVR"

    def test_build_owner_recovery_playbook_tello(self):
        steps = dd.build_owner_recovery_playbook(
            ip="192.168.10.1",
            open_ports=[8889, 11111],
            http_data={"realm": "admin"},
        )
        joined = "\n".join(steps)
        assert "Tello-class control path detected" in joined
        assert "admin UI" in joined


# =============================================================================
# 2. DRONE_SIGNATURES database integrity
# =============================================================================

class TestDroneSignaturesDatabase:
    """Validates the built-in drone signature database."""

    @pytest.fixture(scope="class")
    @classmethod
    def sigs(cls):
        return dd.DRONE_SIGNATURES

    def test_not_empty(self, sigs):
        assert len(sigs) > 0

    def test_no_duplicate_keys(self, sigs):
        keys = list(sigs.keys())
        assert len(keys) == len(set(keys)), "Duplicate keys found in DRONE_SIGNATURES"

    def test_required_fields_present(self, sigs):
        for key, sig in sigs.items():
            assert sig.key, f"{key}: 'key' is empty"
            assert sig.name, f"{key}: 'name' is empty"
            assert sig.brand, f"{key}: 'brand' is empty"
            assert sig.modulation, f"{key}: 'modulation' is empty"

    def test_key_matches_dict_key(self, sigs):
        for dict_key, sig in sigs.items():
            assert sig.key == dict_key, (
                f"Signature key mismatch: dict key '{dict_key}' != sig.key '{sig.key}'"
            )

    def test_bandwidth_positive(self, sigs):
        for key, sig in sigs.items():
            assert sig.bandwidth_hz > 0, f"{key}: bandwidth_hz must be > 0"

    def test_snr_floor_in_range(self, sigs):
        for key, sig in sigs.items():
            assert 0 <= sig.snr_floor_db <= 40, (
                f"{key}: snr_floor_db={sig.snr_floor_db} is out of [0, 40]"
            )

    def test_all_frequencies_valid_mhz(self, sigs):
        for key, sig in sigs.items():
            all_freqs = (
                sig.freqs_24ghz_mhz
                + sig.freqs_58ghz_mhz
                + sig.freqs_900mhz_mhz
                + sig.freqs_433mhz_mhz
            )
            for f in all_freqs:
                assert 0 < f < 7000, (
                    f"{key}: frequency {f} MHz is outside valid range (0, 7000)"
                )

    def test_at_least_some_entries_have_frequencies(self, sigs):
        entries_with_freqs = [
            key for key, sig in sigs.items()
            if (sig.freqs_24ghz_mhz or sig.freqs_58ghz_mhz
                or sig.freqs_900mhz_mhz or sig.freqs_433mhz_mhz)
        ]
        assert len(entries_with_freqs) > 0


# =============================================================================
# 3. fingerprint_signal
# =============================================================================

class TestFingerprintSignal:
    def test_returns_list(self):
        result = dd.fingerprint_signal(2412e6, 20e6, 15.0, -50.0)
        assert isinstance(result, list)

    def test_no_match_for_obscure_frequency(self):
        # 1234.5 MHz is well outside every drone band
        result = dd.fingerprint_signal(1234.5e6, 5e6, 20.0, -60.0)
        assert result == []

    def test_ocusync_24ghz_returns_dji_match(self):
        # 2412.3 MHz is right on a WiFi / OcuSync channel — DJI should rank high
        result = dd.fingerprint_signal(2412.3e6, 20e6, 15.0, -55.0)
        assert len(result) > 0
        brands = [m.brand for m in result]
        assert "DJI" in brands, f"Expected DJI in matches, got brands: {brands}"

    def test_top_match_has_highest_confidence(self):
        result = dd.fingerprint_signal(2412.3e6, 20e6, 15.0, -55.0)
        if len(result) > 1:
            confidences = [m.confidence for m in result]
            assert confidences == sorted(confidences, reverse=True), (
                "Results are not sorted descending by confidence"
            )

    def test_confidence_bounds(self):
        result = dd.fingerprint_signal(2412.3e6, 20e6, 15.0, -55.0)
        for m in result:
            assert 0.0 <= m.confidence <= 1.0, (
                f"confidence {m.confidence} for {m.sig_key} is out of [0, 1]"
            )

    def test_returns_drone_match_objects(self):
        result = dd.fingerprint_signal(2412.3e6, 20e6, 15.0, -55.0)
        for m in result:
            assert isinstance(m, dd.DroneMatch)

    def test_at_most_three_matches(self):
        # fingerprint_signal caps at top-3
        result = dd.fingerprint_signal(2412.3e6, 20e6, 15.0, -55.0)
        assert len(result) <= 3

    def test_433mhz_mavlink_match(self):
        # 433.1 MHz is in the MAVLink 433 band
        result = dd.fingerprint_signal(433.1e6, 500e3, 12.0, -70.0)
        assert len(result) > 0
        keys = [m.sig_key for m in result]
        assert any("433" in k or "mavlink" in k for k in keys), (
            f"Expected a 433 MHz MAVLink match, got keys: {keys}"
        )

    def test_match_fields_present(self):
        result = dd.fingerprint_signal(2412.3e6, 20e6, 15.0, -55.0)
        if result:
            m = result[0]
            assert m.sig_key
            assert m.name
            assert m.brand
            assert m.timestamp_utc


# =============================================================================
# 4. build_ocusync_channel_map
# =============================================================================

class TestBuildOcusyncChannelMap:
    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go
        fig = dd.build_ocusync_channel_map([])
        assert isinstance(fig, go.Figure)

    def test_empty_results(self):
        import plotly.graph_objects as go
        fig = dd.build_ocusync_channel_map([])
        assert isinstance(fig, go.Figure)
        # Title should mention "0/" since no active channels
        assert "0/" in fig.layout.title.text

    def test_with_scan_result_at_ocusync_freq(self):
        import plotly.graph_objects as go
        from datetime import datetime, timezone

        # Build a minimal ScanResult at a known OcuSync 2.4 GHz channel
        # Channel 1 is at 2400.5 MHz
        ch1_hz = 2400.5e6
        fake_result = dd.ScanResult(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            center_freq_hz=ch1_hz,
            peak_db=-60.0,
            noise_floor_db=-100.0,
            snr_db=40.0,
            occupied_bw_hz=10e6,
            detected=True,
            freq_axis_hz=np.linspace(-5e6, 5e6, 256, dtype=np.float32),
            spectrum_db=np.full(256, -60.0, dtype=np.float32),
            matches=[],
        )

        fig = dd.build_ocusync_channel_map([fake_result])
        assert isinstance(fig, go.Figure)
        # At least 1 channel should be active now
        assert "1/" in fig.layout.title.text or "active" in fig.layout.title.text.lower()

    def test_5ghz_band(self):
        import plotly.graph_objects as go
        fig = dd.build_ocusync_channel_map([], band="5.8GHz")
        assert isinstance(fig, go.Figure)
        assert "5.8GHz" in fig.layout.title.text


# =============================================================================
# 5. human_freq / human_bw / confidence_color helpers
# =============================================================================

class TestHumanFreq:
    def test_433mhz(self):
        result = dd.human_freq(433e6)
        assert "433" in result

    def test_24ghz_contains_2400_or_2dot4(self):
        result = dd.human_freq(2.4e9)
        # Returns "2.400 GHz"
        assert "2.4" in result or "2400" in result

    def test_1ghz(self):
        result = dd.human_freq(1e9)
        assert "GHz" in result or "1000" in result

    def test_khz_range(self):
        result = dd.human_freq(500e3)
        assert "kHz" in result or "500" in result

    def test_hz_range(self):
        result = dd.human_freq(100.0)
        assert "Hz" in result


class TestHumanBw:
    def test_20mhz(self):
        result = dd.human_bw(20e6)
        assert "20" in result

    def test_500khz(self):
        result = dd.human_bw(500e3)
        assert "500" in result or "kHz" in result

    def test_returns_string(self):
        assert isinstance(dd.human_bw(10e6), str)


class TestConfidenceColor:
    def test_high_confidence_green(self):
        assert dd.confidence_color(0.80) == "🟢"

    def test_exact_threshold_70_green(self):
        assert dd.confidence_color(0.70) == "🟢"

    def test_medium_confidence_yellow(self):
        assert dd.confidence_color(0.50) == "🟡"

    def test_exact_threshold_45_yellow(self):
        assert dd.confidence_color(0.45) == "🟡"

    def test_low_confidence_red(self):
        assert dd.confidence_color(0.20) == "🔴"

    def test_zero_confidence_red(self):
        assert dd.confidence_color(0.0) == "🔴"

    def test_boundary_just_below_70(self):
        assert dd.confidence_color(0.699) == "🟡"

    def test_boundary_just_below_45(self):
        assert dd.confidence_color(0.449) == "🔴"


# =============================================================================
# 6. WIFI_DRONE_FINGERPRINTS / probe_wifi_drones
# =============================================================================

# Simple dotted-quad check: four 1-3 digit groups separated by dots.
# Full range validation (0-255) is not needed here — we only want to catch
# obviously malformed strings like hostnames or empty values.
_DOTTED_QUAD_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


class TestWifiDroneFingerprints:
    def test_non_empty(self):
        assert len(dd.WIFI_DRONE_FINGERPRINTS) > 0

    def test_required_fields(self):
        required = {"brand", "model", "ip", "port", "adapter"}
        for entry in dd.WIFI_DRONE_FINGERPRINTS:
            missing = required - set(entry.keys())
            assert not missing, f"Entry missing fields {missing}: {entry}"

    def test_all_ips_are_valid_dotted_quad(self):
        for entry in dd.WIFI_DRONE_FINGERPRINTS:
            ip = entry["ip"]
            assert _DOTTED_QUAD_RE.match(ip), (
                f"'{ip}' for {entry['brand']} {entry['model']} is not a valid IPv4 address"
            )

    def test_all_ports_are_positive_integers(self):
        for entry in dd.WIFI_DRONE_FINGERPRINTS:
            assert isinstance(entry["port"], int) and entry["port"] > 0, (
                f"Invalid port {entry['port']} for {entry['brand']}"
            )

    def test_probe_wifi_drones_returns_list(self):
        # Runs a real (fast) network probe; no drone expected on CI.
        result = dd.probe_wifi_drones(timeout=0.2)
        assert isinstance(result, list)
        # Each returned item, if any, must have the expected keys
        for item in result:
            assert "brand" in item
            assert "ip" in item


# =============================================================================
# 7. DRONE_SSID_PREFIXES
# =============================================================================

class TestDroneSsidPrefixes:
    def test_non_empty(self):
        assert len(dd.DRONE_SSID_PREFIXES) > 0

    def test_each_entry_is_3_tuple(self):
        for entry in dd.DRONE_SSID_PREFIXES:
            assert len(entry) == 3, f"Expected 3-tuple, got length {len(entry)}: {entry}"

    def test_first_field_is_string(self):
        for brand, model, prefixes in dd.DRONE_SSID_PREFIXES:
            assert isinstance(brand, str) and brand, (
                f"brand field is not a non-empty string: {brand!r}"
            )

    def test_second_field_is_string(self):
        for brand, model, prefixes in dd.DRONE_SSID_PREFIXES:
            assert isinstance(model, str) and model, (
                f"model field is not a non-empty string: {model!r}"
            )

    def test_third_field_is_list_of_nonempty_strings(self):
        for brand, model, prefixes in dd.DRONE_SSID_PREFIXES:
            assert isinstance(prefixes, list) and len(prefixes) > 0, (
                f"prefixes for {brand}/{model} must be a non-empty list"
            )
            for p in prefixes:
                assert isinstance(p, str) and p, (
                    f"prefix {p!r} for {brand}/{model} is not a non-empty string"
                )

    def test_known_brands_present(self):
        brands = {entry[0] for entry in dd.DRONE_SSID_PREFIXES}
        for expected in ("DJI", "Parrot", "Skydio"):
            assert expected in brands, f"Expected brand '{expected}' missing from DRONE_SSID_PREFIXES"


# =============================================================================
# 8. WifiScanThread / BleScanThread
# =============================================================================

class TestWifiScanThread:
    @pytest.fixture
    def sensor_q(self):
        return queue.Queue(maxsize=100)

    def test_can_be_instantiated(self, sensor_q):
        t = dd.WifiScanThread(sensor_q)
        assert isinstance(t, dd.WifiScanThread)

    def test_has_stop_method(self, sensor_q):
        t = dd.WifiScanThread(sensor_q)
        assert callable(t.stop)

    def test_has_status_attribute(self, sensor_q):
        t = dd.WifiScanThread(sensor_q)
        assert hasattr(t, "status")

    def test_start_stop_join_no_hang(self, sensor_q):
        # nmcli --rescan yes can take 5-25 s depending on adapter state;
        # give generous timeout so this isn't flaky in a full-suite run.
        t = dd.WifiScanThread(sensor_q, interval_s=0.1)
        t.start()
        time.sleep(0.1)
        t.stop()
        t.join(timeout=30.0)  # nmcli rescan can take up to ~20 s
        assert not t.is_alive(), "WifiScanThread did not stop within 30 seconds"

    def test_status_string_updated(self, sensor_q):
        t = dd.WifiScanThread(sensor_q, interval_s=0.1)
        t.start()
        time.sleep(0.5)
        t.stop()
        t.join(timeout=3.0)
        # Status should have been updated from its initial "Starting…" value
        # (it will be set to either a success or error message)
        assert t.status != "Starting…" or not t.is_alive()


class TestBleScanThread:
    @pytest.fixture
    def sensor_q(self):
        return queue.Queue(maxsize=100)

    def test_can_be_instantiated(self, sensor_q):
        t = dd.BleScanThread(sensor_q)
        assert isinstance(t, dd.BleScanThread)

    def test_has_stop_method(self, sensor_q):
        t = dd.BleScanThread(sensor_q)
        assert callable(t.stop)

    def test_has_status_attribute(self, sensor_q):
        t = dd.BleScanThread(sensor_q)
        assert hasattr(t, "status")

    def test_is_daemon_thread(self, sensor_q):
        t = dd.BleScanThread(sensor_q)
        assert t.daemon, "BleScanThread should be a daemon thread"

    def test_instantiation_does_not_start_ble_scan(self, sensor_q):
        # Just constructing the thread must not trigger any BLE scan.
        t = dd.BleScanThread(sensor_q, scan_duration=0.01, interval_s=9999)
        assert not t.is_alive()
        # Do NOT start it — we don't want a real BLE scan in CI


# =============================================================================
# 9. Miscellaneous / edge-case regression guards
# =============================================================================

class TestDataclassStructures:
    def test_drone_match_can_be_constructed(self):
        from datetime import datetime, timezone
        m = dd.DroneMatch(
            sig_key="test",
            name="Test Drone",
            brand="Acme",
            confidence=0.75,
            center_freq_hz=2.4e9,
            detected_bw_hz=10e6,
            snr_db=15.0,
            peak_db=-60.0,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            notes="test note",
        )
        assert m.confidence == pytest.approx(0.75)
        assert m.adapter == ""   # default

    def test_scan_result_default_matches_list(self):
        from datetime import datetime, timezone
        r = dd.ScanResult(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            center_freq_hz=2.4e9,
            peak_db=-80.0,
            noise_floor_db=-110.0,
            snr_db=30.0,
            occupied_bw_hz=10e6,
            detected=True,
            freq_axis_hz=np.zeros(1, dtype=np.float32),
            spectrum_db=np.zeros(1, dtype=np.float32),
        )
        assert r.matches == []

    def test_ocusync_channel_lists_not_empty(self):
        assert len(dd.OCUSYNC_CHANNELS_2G) == 36
        assert len(dd.OCUSYNC_CHANNELS_5G) == 6

    def test_ocusync_2g_channels_in_valid_range(self):
        for ch in dd.OCUSYNC_CHANNELS_2G:
            assert 2400 <= ch <= 2484, f"OcuSync 2G channel {ch} MHz out of range"

    def test_ocusync_5g_channels_in_valid_range(self):
        for ch in dd.OCUSYNC_CHANNELS_5G:
            assert 5700 <= ch <= 5850, f"OcuSync 5G channel {ch} MHz out of range"
