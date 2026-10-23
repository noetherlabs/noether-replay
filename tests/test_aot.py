"""Real AOT round trips and bounded-input admission regressions."""

import gzip
import hashlib
import io
import struct

import numpy as np
import pytest

from noether_replay.aot import read_aot
from noether_replay.replay import events
from noether_replay.report import build_report

aotpy = pytest.importorskip("aotpy")
fits = pytest.importorskip("astropy.io.fits")


@pytest.fixture
def aot_system():
    system = aotpy.AOSystem(ao_mode="SCAO", name="synthetic-aot")
    telescope = aotpy.MainTelescope("telescope")
    system.main_telescope = telescope
    source = aotpy.NaturalGuideStar("source")
    system.sources = [source]
    time = aotpy.Time("time", timestamps=[1.0, 2.0, 3.0], frame_numbers=[10.0, 11.0, 13.0])
    sensor = aotpy.ShackHartmann("wfs", source=source, n_valid_subapertures=1)
    sensor.measurements = aotpy.Image(
        "measurements",
        np.arange(6, dtype=">f4").reshape(3, 2, 1),
        unit="rad",
        time=time,
    )
    sensor.detector = aotpy.Detector(
        "detector",
        frame_rate=1000.0,
        integration_time=0.0009,
        pixel_intensities=aotpy.Image(
            "pixels",
            np.arange(12, dtype=">f4").reshape(3, 2, 2),
            unit="ADU",
            time=time,
        ),
    )
    system.wavefront_sensors = [sensor]
    dm = aotpy.DeformableMirror("dm", telescope=telescope, n_valid_actuators=2)
    system.wavefront_correctors = [dm]
    system.loops = [
        aotpy.ControlLoop(
            "loop",
            commanded_corrector=dm,
            input_sensor=sensor,
            framerate=1000.0,
            commands=aotpy.Image("commands", np.zeros((3, 2), dtype=">f8"), unit="m", time=time),
            time_filter_num=aotpy.Image("numerator", np.array([[0.3]])),
        )
    ]
    return system


@pytest.fixture
def aot_file(tmp_path, aot_system):
    path = tmp_path / "fixture.fits"
    aot_system.write_to_file(path)
    return path


def test_actual_aot_round_trip_preserves_arrays_and_unknown_event_semantics(aot_file, aot_system):
    recording = read_aot(aot_file)
    streams = {stream.id: stream for stream in recording.streams}
    source = aot_system.wavefront_sensors[0].measurements.data
    measured = streams["wfs:wfs:measurements"]
    np.testing.assert_array_equal(measured.values, source)
    assert measured.values.dtype.str == source.dtype.str
    assert measured.units == "rad"
    assert measured.frame_ids == (10, 11, 13)
    assert measured.metadata["aot_time"]["timestamps_seconds"] == [1.0, 2.0, 3.0]
    assert measured.times_ns == {}
    assert measured.clock_id is None
    assert measured.uncertainty_ns is None
    assert events(recording) == []
    commands = streams["loop:loop:commands"]
    assert commands.kind == "command_recorded"
    assert commands.metadata["collection_controller"]["frame_rate_hz"] == 1000.0
    assert commands.metadata["collection_controller"]["time_filter_numerator"]["values"] == [[0.3]]
    assert recording.calibrations == ()
    assert (
        recording.provenance["source_sha256"] == hashlib.sha256(aot_file.read_bytes()).hexdigest()
    )
    assert recording.provenance["aotpy_version"] == "3.2.1"
    assert read_aot(aot_file).digest() == recording.digest()
    report = build_report(recording)
    assert report["status"] == "evidence_gaps"
    assert {item["code"] for item in report["findings"]} >= {
        "missing_availability",
        "unknown_command_semantics",
        "unqualified_calibration_interval",
    }


def test_missing_aot_time_does_not_invent_frame_ids_or_frequency_timestamps(tmp_path, aot_system):
    aot_system.wavefront_sensors[0].measurements.time = None
    path = tmp_path / "missing.fits"
    aot_system.write_to_file(path)
    stream = read_aot(path).streams[0]
    assert stream.frame_ids is None
    assert stream.times_ns == {}
    assert "aot_time" not in stream.metadata


@pytest.mark.parametrize(
    "frames", [[1.0, 2.5, 3.0], [float(2**54), float(2**54 + 4), float(2**54 + 8)]]
)
def test_unqualified_source_frame_numbers_remain_metadata(tmp_path, aot_system, frames):
    aot_system.wavefront_sensors[0].measurements.time.frame_numbers = frames
    path = tmp_path / "ambiguous-frames.fits"
    aot_system.write_to_file(path)
    stream = read_aot(path).streams[0]
    assert stream.frame_ids is None
    assert stream.metadata["aot_time"]["frame_numbers"] == frames
    assert "frame_identity_issue" in stream.metadata


def test_nonfinite_source_arrays_are_retained(tmp_path, aot_system):
    aot_system.wavefront_sensors[0].measurements.data[0, 0, 0] = np.nan
    path = tmp_path / "nonfinite.fits"
    aot_system.write_to_file(path)
    recording = read_aot(path)
    assert np.isnan(recording.streams[0].values[0, 0, 0])
    assert build_report(recording)["streams"][1]["nonfinite_values"] == 1


@pytest.mark.parametrize("reference", ["FILEREF<x>", "URLREF<http://a>"])
def test_external_references_are_rejected_without_external_access(aot_file, monkeypatch, reference):
    with fits.open(aot_file, mode="update", memmap=False) as hdus:
        hdus["AOT_WAVEFRONT_SENSORS"].data["MEASUREMENTS"][0] = reference
    original = fits.open
    opened = []

    def guarded_open(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO), (
            "Adapter attempted to resolve an external path or URL"
        )
        opened.append(source)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(fits, "open", guarded_open)
    with pytest.raises(ValueError, match="External"):
        read_aot(aot_file)
    assert len(opened) == 2


def forbidden_parser(*args, **kwargs):
    raise AssertionError("aotpy was called before unsafe FITS input was rejected")


def test_gzip_disguised_as_fits_is_rejected_before_parser(aot_file, monkeypatch):
    aot_file.write_bytes(gzip.compress(aot_file.read_bytes()))
    monkeypatch.setattr(aotpy.AOSystem, "read_from_file", forbidden_parser)
    with pytest.raises(ValueError, match="uncompressed"):
        read_aot(aot_file)


def test_unused_compressed_image_is_rejected_before_parser(aot_file, monkeypatch):
    with fits.open(aot_file, mode="append", memmap=False) as hdus:
        hdus.append(fits.CompImageHDU(np.zeros((2, 2)), name="UNUSED"))
    monkeypatch.setattr(aotpy.AOSystem, "read_from_file", forbidden_parser)
    with pytest.raises(ValueError, match="Compressed"):
        read_aot(aot_file)


def test_huge_unused_image_header_is_rejected_without_allocation(aot_file, monkeypatch):
    with fits.open(aot_file, mode="append", memmap=False) as hdus:
        hdus.append(fits.ImageHDU(np.zeros(1, dtype=np.uint8), name="UNUSED"))
    with fits.open(aot_file, memmap=False) as hdus:
        hdu = hdus["UNUSED"]
        location = hdu.fileinfo()["hdrLoc"] + hdu.header.index("NAXIS1") * 80
    payload = bytearray(aot_file.read_bytes())
    payload[location : location + 80] = fits.Card("NAXIS1", 2**35).image.encode("ascii")
    aot_file.write_bytes(payload)
    monkeypatch.setattr(aotpy.AOSystem, "read_from_file", forbidden_parser)
    with pytest.raises(ValueError, match="extent"):
        read_aot(aot_file)


def test_variable_array_descriptor_cannot_escape_its_heap(tmp_path, monkeypatch):
    column = fits.Column(
        name="DATA", format="PD()", array=np.array([np.array([1.0])], dtype=object)
    )
    path = tmp_path / "bad-heap.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([column])]).writeto(path)
    with fits.open(path, memmap=False) as hdus:
        location = hdus[1].fileinfo()["datLoc"]
    payload = bytearray(path.read_bytes())
    struct.pack_into(">ii", payload, location, 1000, 0)
    path.write_bytes(payload)
    monkeypatch.setattr(aotpy.AOSystem, "read_from_file", forbidden_parser)
    with pytest.raises(ValueError, match="heap"):
        read_aot(path)


def test_file_size_gate_precedes_parser(aot_file, monkeypatch):
    import noether_replay.aot as adapter

    monkeypatch.setattr(adapter, "MAX_FILE_BYTES", 1)
    monkeypatch.setattr(aotpy.AOSystem, "read_from_file", forbidden_parser)
    with pytest.raises(ValueError, match="regular file"):
        read_aot(aot_file)


def test_source_hash_binds_the_parsed_snapshot(aot_file, monkeypatch):
    original_bytes = aot_file.read_bytes()
    original = aotpy.AOSystem.read_from_file

    def replace_source_then_parse(snapshot, **kwargs):
        aot_file.write_bytes(b"source changed after snapshot")
        return original(snapshot, **kwargs)

    monkeypatch.setattr(aotpy.AOSystem, "read_from_file", replace_source_then_parse)
    recording = read_aot(aot_file)
    assert recording.provenance["source_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert recording.streams[0].values.shape == (3, 2, 1)


def test_heap_alias_expansion_counts_each_descriptor(tmp_path, monkeypatch):
    import noether_replay.aot as adapter

    column = fits.Column(
        name="DATA",
        format="PD()",
        array=np.array([np.arange(4, dtype=float), np.arange(4, dtype=float)], dtype=object),
    )
    path = tmp_path / "aliased-heap.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([column])]).writeto(path)
    with fits.open(path, memmap=False) as hdus:
        location = hdus[1].fileinfo()["datLoc"]
    payload = bytearray(path.read_bytes())
    struct.pack_into(">ii", payload, location + 8, 4, 0)
    path.write_bytes(payload)
    monkeypatch.setattr(adapter, "MAX_TABLE_ELEMENTS", 6)
    monkeypatch.setattr(aotpy.AOSystem, "read_from_file", forbidden_parser)
    with pytest.raises(ValueError, match="expanded table"):
        read_aot(path)


def test_symlink_is_not_followed(aot_file, tmp_path, monkeypatch):
    link = tmp_path / "link.fits"
    link.symlink_to(aot_file)
    monkeypatch.setattr(aotpy.AOSystem, "read_from_file", forbidden_parser)
    with pytest.raises((ValueError, OSError)):
        read_aot(link)


def test_decoded_image_budget_applies_to_unused_images(aot_file, monkeypatch):
    import noether_replay.aot as adapter

    with fits.open(aot_file, mode="append", memmap=False) as hdus:
        hdus.append(fits.ImageHDU(np.zeros(20, dtype=np.uint8), name="UNUSED"))
    # The disk array is 20 bytes, but FITS scaling can require 160 decoded bytes.
    monkeypatch.setattr(adapter, "MAX_ARRAY_BYTES", 100)
    monkeypatch.setattr(aotpy.AOSystem, "read_from_file", forbidden_parser)
    with pytest.raises(ValueError, match="allocation"):
        read_aot(aot_file)


def test_repeated_image_references_are_bounded_before_any_stream_copy(tmp_path, monkeypatch):
    import noether_replay.aot as adapter

    system = aotpy.AOSystem(ao_mode="SCAO", name="shared-image")
    system.main_telescope = aotpy.MainTelescope("telescope")
    source = aotpy.NaturalGuideStar("source")
    system.sources = [source]
    image = aotpy.Image("shared", np.zeros((4, 2, 1), dtype=np.float64))
    system.wavefront_sensors = [
        aotpy.ShackHartmann(
            f"wfs-{index}",
            source=source,
            n_valid_subapertures=1,
            measurements=image,
        )
        for index in range(5)
    ]
    path = tmp_path / "shared-image.fits"
    system.write_to_file(path)
    monkeypatch.setattr(adapter, "MAX_ARRAY_BYTES", 64)

    def forbidden_copy(*args, **kwargs):
        raise AssertionError("Stream copying began before aggregate admission")

    monkeypatch.setattr(adapter, "_stream", forbidden_copy)
    with pytest.raises(ValueError, match="aggregate"):
        adapter.read_aot(path)


def test_repeated_filter_references_have_a_cumulative_metadata_bound(
    tmp_path, aot_system, monkeypatch
):
    import noether_replay.aot as adapter

    shared_filter = aotpy.Image("shared-filter", np.arange(64, dtype=np.float64).reshape(1, 64))
    template = aot_system.loops[0]
    aot_system.loops = [
        aotpy.ControlLoop(
            f"loop-{index}",
            commanded_corrector=template.commanded_corrector,
            input_sensor=template.input_sensor,
            time_filter_num=shared_filter,
        )
        for index in range(10)
    ]
    path = tmp_path / "shared-filter.fits"
    aot_system.write_to_file(path)
    monkeypatch.setattr(adapter, "MAX_JSON_BYTES", 3000)
    calls = []
    original = adapter._image_reference

    def counted_reference(image):
        calls.append(image)
        return original(image)

    monkeypatch.setattr(adapter, "_image_reference", counted_reference)
    with pytest.raises(ValueError, match="metadata"):
        adapter.read_aot(path)
    assert len(calls) < 20, "Metadata expansion must stop before converting every aliased filter"
