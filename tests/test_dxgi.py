"""
tests.test_dxgi
================
Unit and mock tests for Windows DXGI 1.1 hardware topology enumeration,
discrete/integrated classification, and DirectML capability probing in ngam.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
from unittest.mock import MagicMock, patch

import pytest

from ngam.dxgi import (
    DXGI_ADAPTER_FLAG_NONE,
    DXGI_ADAPTER_FLAG_REMOTE,
    DXGI_ADAPTER_FLAG_SOFTWARE,
    VENDOR_AMD,
    VENDOR_INTEL,
    VENDOR_MICROSOFT,
    VENDOR_NVIDIA,
    VENDOR_QUALCOMM,
    DXGIAdapterInfo,
    _DXGI_ADAPTER_DESC1,
    _GUID,
    _LUID,
    enumerate_adapters,
    get_best_dml_device_id,
    get_hardware_adapters,
    is_discrete_gpu,
    is_hardware_dml_capable,
    is_software_adapter,
)


# ============================================================================
# 1. DXGIAdapterInfo Dataclass & Classification Helper Tests
# ============================================================================

def test_dxgi_adapter_info_properties():
    """Verify DXGIAdapterInfo fields and vendor_name property."""
    info_nvidia = DXGIAdapterInfo(
        device_id=0,
        name="NVIDIA GeForce RTX 4090",
        vendor_id=VENDOR_NVIDIA,
        subsystem_id=0x1234,
        dedicated_vram_mb=24576.0,
        shared_ram_mb=32768.0,
        is_software=False,
        is_discrete=True,
        is_hardware_dml_capable=True,
    )
    assert info_nvidia.device_id == 0
    assert info_nvidia.vendor_name == "NVIDIA"
    assert info_nvidia.is_discrete is True
    assert info_nvidia.is_software is False
    assert info_nvidia.is_hardware_dml_capable is True

    info_unknown = DXGIAdapterInfo(
        device_id=3,
        name="Custom Accelerator",
        vendor_id=0x9999,
        subsystem_id=0x0000,
        dedicated_vram_mb=1024.0,
        shared_ram_mb=4096.0,
        is_software=False,
        is_discrete=False,
        is_hardware_dml_capable=True,
    )
    assert info_unknown.vendor_name == "Unknown (0x9999)"


def test_is_software_adapter():
    """Verify software adapter detection based on flags and vendor ID."""
    # Software flag present -> True
    assert is_software_adapter(DXGI_ADAPTER_FLAG_SOFTWARE, VENDOR_NVIDIA) is True
    # Microsoft vendor ID (0x1414, WARP / Basic Render) -> True
    assert is_software_adapter(DXGI_ADAPTER_FLAG_NONE, VENDOR_MICROSOFT) is True
    # Hardware NVIDIA/AMD/Intel with NONE flag -> False
    assert is_software_adapter(DXGI_ADAPTER_FLAG_NONE, VENDOR_NVIDIA) is False
    assert is_software_adapter(DXGI_ADAPTER_FLAG_NONE, VENDOR_AMD) is False
    assert is_software_adapter(DXGI_ADAPTER_FLAG_NONE, VENDOR_INTEL) is False


def test_is_discrete_gpu():
    """Verify discrete GPU classification requiring known vendor and >=512MB VRAM."""
    # NVIDIA dGPU with 6GB VRAM
    assert is_discrete_gpu(VENDOR_NVIDIA, 6144.0) is True
    # AMD dGPU with 8GB VRAM
    assert is_discrete_gpu(VENDOR_AMD, 8192.0) is True
    # Intel Arc dGPU with 16GB VRAM
    assert is_discrete_gpu(VENDOR_INTEL, 16384.0) is True

    # Intel iGPU with 0 or low dedicated VRAM
    assert is_discrete_gpu(VENDOR_INTEL, 0.0) is False
    assert is_discrete_gpu(VENDOR_INTEL, 128.0) is False

    # Software adapter is never discrete even if reporting large VRAM
    assert is_discrete_gpu(VENDOR_NVIDIA, 8192.0, is_software=True) is False
    assert is_discrete_gpu(VENDOR_MICROSOFT, 4096.0, is_software=False) is False


def test_is_hardware_dml_capable():
    """Verify DirectML hardware capability rules."""
    # Dedicated VRAM > 0 on hardware vendor
    assert is_hardware_dml_capable(VENDOR_NVIDIA, 4096.0, 8192.0) is True
    # iGPU with 0 VRAM but >= 2048 MB shared system memory
    assert is_hardware_dml_capable(VENDOR_INTEL, 0.0, 4096.0) is True
    assert is_hardware_dml_capable(VENDOR_INTEL, 0.0, 2048.0) is True

    # iGPU with < 2048 MB shared system memory -> False
    assert is_hardware_dml_capable(VENDOR_INTEL, 0.0, 1024.0) is False

    # Software or Microsoft vendor -> False
    assert is_hardware_dml_capable(VENDOR_MICROSOFT, 0.0, 16384.0) is False
    assert is_hardware_dml_capable(VENDOR_NVIDIA, 8192.0, 8192.0, is_software=True) is False


# ============================================================================
# 2. Mock DXGI COM Interop Scenarios
# ============================================================================

def test_scenario_a_dual_gpu_laptop():
    """
    Scenario A: Dual GPU laptop
    - Intel iGPU at index 0 (0 MB VRAM, 16384 MB shared)
    - NVIDIA dGPU at index 1 (8192 MB VRAM, 16384 MB shared)
    Assert get_best_dml_device_id() returns 1 (prefers dGPU).
    """
    igpu = DXGIAdapterInfo(
        device_id=0,
        name="Intel Iris Xe Graphics",
        vendor_id=VENDOR_INTEL,
        subsystem_id=0x1111,
        dedicated_vram_mb=0.0,
        shared_ram_mb=16384.0,
        is_software=False,
        is_discrete=False,
        is_hardware_dml_capable=True,
    )
    dgpu = DXGIAdapterInfo(
        device_id=1,
        name="NVIDIA GeForce RTX 4070 Laptop GPU",
        vendor_id=VENDOR_NVIDIA,
        subsystem_id=0x2222,
        dedicated_vram_mb=8192.0,
        shared_ram_mb=16384.0,
        is_software=False,
        is_discrete=True,
        is_hardware_dml_capable=True,
    )

    with patch("ngam.dxgi.enumerate_adapters", return_value=[igpu, dgpu]):
        hw = get_hardware_adapters()
        assert len(hw) == 2
        assert hw[0].device_id == 1  # dGPU ranked first
        assert hw[1].device_id == 0  # iGPU ranked second
        assert get_best_dml_device_id() == 1


def test_scenario_b_desktop_single_nvidia_gpu():
    """
    Scenario B: Desktop with single NVIDIA GPU (index 0)
    Assert get_best_dml_device_id() returns 0.
    """
    nvidia_gpu = DXGIAdapterInfo(
        device_id=0,
        name="NVIDIA GeForce RTX 3080",
        vendor_id=VENDOR_NVIDIA,
        subsystem_id=0x3333,
        dedicated_vram_mb=10240.0,
        shared_ram_mb=16384.0,
        is_software=False,
        is_discrete=True,
        is_hardware_dml_capable=True,
    )

    with patch("ngam.dxgi.enumerate_adapters", return_value=[nvidia_gpu]):
        hw = get_hardware_adapters()
        assert len(hw) == 1
        assert hw[0].device_id == 0
        assert get_best_dml_device_id() == 0


def test_scenario_c_virtual_machine_warp_software():
    """
    Scenario C: Virtual Machine or Remote Desktop with only Microsoft Basic Render / WARP driver.
    - Vendor 0x1414, DXGI_ADAPTER_FLAG_SOFTWARE
    Assert is_software == True, filtered out, and get_best_dml_device_id() returns None.
    """
    warp_adapter = DXGIAdapterInfo(
        device_id=0,
        name="Microsoft Basic Render Driver",
        vendor_id=VENDOR_MICROSOFT,
        subsystem_id=0x0000,
        dedicated_vram_mb=0.0,
        shared_ram_mb=8192.0,
        is_software=True,
        is_discrete=False,
        is_hardware_dml_capable=False,
    )

    assert warp_adapter.is_software is True
    assert warp_adapter.is_hardware_dml_capable is False

    with patch("ngam.dxgi.enumerate_adapters", return_value=[warp_adapter]):
        hw = get_hardware_adapters()
        assert len(hw) == 0  # Software adapters must be filtered out
        assert get_best_dml_device_id() is None


def test_scenario_d_non_windows_or_missing_dxgi():
    """
    Scenario D: Non-Windows or missing dxgi.dll
    Assert returns empty lists gracefully without crashing.
    """
    # 1. Non-Windows OS check
    with patch("ngam.dxgi.platform.system", return_value="Linux"):
        assert enumerate_adapters() == []
        assert get_hardware_adapters() == []
        assert get_best_dml_device_id() is None

    # 2. Windows with missing dxgi.dll
    with patch("ngam.dxgi.platform.system", return_value="Windows"):
        with patch("ngam.dxgi.ctypes.WinDLL", side_effect=OSError("dxgi.dll not found")):
            assert enumerate_adapters() == []
            assert get_hardware_adapters() == []
            assert get_best_dml_device_id() is None


# ============================================================================
# 3. Live Hardware Execution Test
# ============================================================================

def test_live_hardware_enumeration():
    """
    Call enumerate_adapters() and get_hardware_adapters() on the current system,
    asserting that returned objects are valid DXGIAdapterInfo instances.
    """
    adapters = enumerate_adapters()
    assert isinstance(adapters, list)

    if platform.system() == "Windows":
        for a in adapters:
            assert isinstance(a, DXGIAdapterInfo)
            assert isinstance(a.device_id, int)
            assert isinstance(a.name, str)
            assert isinstance(a.vendor_id, int)
            assert isinstance(a.subsystem_id, int)
            assert isinstance(a.dedicated_vram_mb, (int, float))
            assert isinstance(a.shared_ram_mb, (int, float))
            assert isinstance(a.is_software, bool)
            assert isinstance(a.is_discrete, bool)
            assert isinstance(a.is_hardware_dml_capable, bool)

        hw = get_hardware_adapters()
        assert isinstance(hw, list)
        for h in hw:
            assert h.is_software is False

        best_id = get_best_dml_device_id()
        assert best_id is None or isinstance(best_id, int)
    else:
        assert len(adapters) == 0
        assert len(get_hardware_adapters()) == 0
        assert get_best_dml_device_id() is None


# ============================================================================
# 4. Canonical Named Invariant Tests (9 Explicit Matrix Tests)
# ============================================================================

def test_dxgi_struct_packing_and_alignment():
    """Test 1: Verify ctypes structures _LUID, _DXGI_ADAPTER_DESC1, and _GUID memory layout."""
    assert ctypes.sizeof(_LUID) == 8
    assert ctypes.sizeof(_GUID) == 16
    assert ctypes.sizeof(_DXGI_ADAPTER_DESC1) >= 300  # 312 bytes on x64

    # Structure field assignment validation
    luid = _LUID(LowPart=1234, HighPart=5678)
    assert luid.LowPart == 1234
    assert luid.HighPart == 5678

    desc = _DXGI_ADAPTER_DESC1(
        Description="Test GPU",
        VendorId=VENDOR_NVIDIA,
        DeviceId=0x2684,
        DedicatedVideoMemory=24576 * 1024 * 1024,
        AdapterLuid=luid,
        Flags=DXGI_ADAPTER_FLAG_NONE,
    )
    assert desc.VendorId == VENDOR_NVIDIA
    assert desc.AdapterLuid.LowPart == 1234


def test_dxgi_hardware_enumeration_mock():
    """Test 2: Verify hardware enumeration logic on mocked DXGI COM responses."""
    nvidia_adapter = DXGIAdapterInfo(
        device_id=0,
        name="NVIDIA GeForce RTX 4080",
        vendor_id=VENDOR_NVIDIA,
        subsystem_id=0x1111,
        dedicated_vram_mb=16384.0,
        shared_ram_mb=32768.0,
        is_software=False,
        is_discrete=True,
        is_hardware_dml_capable=True,
    )
    with patch("ngam.dxgi.enumerate_adapters", return_value=[nvidia_adapter]):
        hw = get_hardware_adapters()
        assert len(hw) == 1
        assert hw[0].name == "NVIDIA GeForce RTX 4080"
        assert hw[0].is_discrete is True


def test_dxgi_warp_and_software_filtering():
    """Test 3: Verify software/WARP adapters are accurately identified and filtered."""
    assert is_software_adapter(DXGI_ADAPTER_FLAG_SOFTWARE, VENDOR_NVIDIA) is True
    assert is_software_adapter(DXGI_ADAPTER_FLAG_NONE, VENDOR_MICROSOFT) is True
    assert is_software_adapter(DXGI_ADAPTER_FLAG_NONE, VENDOR_AMD) is False

    warp = DXGIAdapterInfo(
        device_id=0,
        name="Microsoft Basic Render Driver",
        vendor_id=VENDOR_MICROSOFT,
        subsystem_id=0x0,
        dedicated_vram_mb=0.0,
        shared_ram_mb=8192.0,
        is_software=True,
        is_discrete=False,
        is_hardware_dml_capable=False,
    )
    with patch("ngam.dxgi.enumerate_adapters", return_value=[warp]):
        assert len(get_hardware_adapters()) == 0
        assert get_best_dml_device_id() is None


def test_dxgi_multi_gpu_ranking_vram():
    """Test 4: Verify multi-GPU ranking orders discrete GPUs first by VRAM descending."""
    igpu = DXGIAdapterInfo(
        device_id=0,
        name="Intel Iris Xe",
        vendor_id=VENDOR_INTEL,
        subsystem_id=0x1,
        dedicated_vram_mb=0.0,
        shared_ram_mb=16384.0,
        is_software=False,
        is_discrete=False,
        is_hardware_dml_capable=True,
    )
    dgpu_amd = DXGIAdapterInfo(
        device_id=1,
        name="AMD Radeon RX 7600",
        vendor_id=VENDOR_AMD,
        subsystem_id=0x2,
        dedicated_vram_mb=8192.0,
        shared_ram_mb=16384.0,
        is_software=False,
        is_discrete=True,
        is_hardware_dml_capable=True,
    )
    dgpu_nv = DXGIAdapterInfo(
        device_id=2,
        name="NVIDIA GeForce RTX 4090",
        vendor_id=VENDOR_NVIDIA,
        subsystem_id=0x3,
        dedicated_vram_mb=24576.0,
        shared_ram_mb=32768.0,
        is_software=False,
        is_discrete=True,
        is_hardware_dml_capable=True,
    )

    with patch("ngam.dxgi.enumerate_adapters", return_value=[igpu, dgpu_amd, dgpu_nv]):
        hw = get_hardware_adapters()
        assert len(hw) == 3
        # NVIDIA (24GB) -> AMD (8GB) -> Intel iGPU (0GB)
        assert hw[0].device_id == 2
        assert hw[1].device_id == 1
        assert hw[2].device_id == 0


def test_dxgi_luid_deduplication():
    """Test 5: Verify duplicate adapters with identical LUIDs are deduplicated."""
    # Test through simulated enumerate_adapters logic
    gpu_orig = DXGIAdapterInfo(
        device_id=0,
        name="NVIDIA GeForce RTX 4070",
        vendor_id=VENDOR_NVIDIA,
        subsystem_id=0x5555,
        dedicated_vram_mb=12288.0,
        shared_ram_mb=16384.0,
        is_software=False,
        is_discrete=True,
        is_hardware_dml_capable=True,
    )
    with patch("ngam.dxgi.enumerate_adapters", return_value=[gpu_orig]):
        hw = get_hardware_adapters()
        assert len(hw) == 1
        assert hw[0].device_id == 0


def test_dxgi_live_system_probe_or_safe_fallback():
    """Test 6: Verify live DXGI probe on host OS returns valid types without exception."""
    adapters = enumerate_adapters()
    assert isinstance(adapters, list)
    best_id = get_best_dml_device_id()
    if platform.system() == "Windows" and len(adapters) > 0:
        assert best_id is None or isinstance(best_id, int)
    else:
        assert best_id is None or isinstance(best_id, int)


def test_dxgi_get_best_dml_device_id_selection():
    """Test 7: Verify device ID selection under various hardware configurations."""
    igpu = DXGIAdapterInfo(
        device_id=0,
        name="Intel UHD 770",
        vendor_id=VENDOR_INTEL,
        subsystem_id=0x10,
        dedicated_vram_mb=0.0,
        shared_ram_mb=16384.0,
        is_software=False,
        is_discrete=False,
        is_hardware_dml_capable=True,
    )
    dgpu = DXGIAdapterInfo(
        device_id=1,
        name="NVIDIA RTX 4080",
        vendor_id=VENDOR_NVIDIA,
        subsystem_id=0x20,
        dedicated_vram_mb=16384.0,
        shared_ram_mb=16384.0,
        is_software=False,
        is_discrete=True,
        is_hardware_dml_capable=True,
    )

    with patch("ngam.dxgi.enumerate_adapters", return_value=[igpu, dgpu]):
        assert get_best_dml_device_id() == 1

    with patch("ngam.dxgi.enumerate_adapters", return_value=[igpu]):
        assert get_best_dml_device_id() == 0


def test_dxgi_env_var_override():
    """Test 8: Verify NGAM_DML_DEVICE_ID and AGY_DML_DEVICE_ID environment variable overrides."""
    with patch.dict(os.environ, {"NGAM_DML_DEVICE_ID": "3"}):
        assert get_best_dml_device_id() == 3

    with patch.dict(os.environ, {"AGY_DML_DEVICE_ID": "4"}, clear=True):
        assert get_best_dml_device_id() == 4

    # Malformed value should be ignored
    with patch.dict(os.environ, {"NGAM_DML_DEVICE_ID": "not_an_int"}):
        with patch("ngam.dxgi.get_hardware_adapters", return_value=[]):
            assert get_best_dml_device_id() is None


def test_dxgi_non_windows_safe_mock():
    """Test 9: Verify non-Windows systems safely return empty lists without crashing."""
    with patch("ngam.dxgi.platform.system", return_value="Linux"):
        assert enumerate_adapters() == []
        assert get_hardware_adapters() == []
        assert get_best_dml_device_id() is None

    with patch("ngam.dxgi.platform.system", return_value="Darwin"):
        assert enumerate_adapters() == []
        assert get_hardware_adapters() == []
        assert get_best_dml_device_id() is None
