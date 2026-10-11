"""
ngam.dxgi
=========
Zero-dependency Windows DXGI 1.1 hardware topology and DirectML capability enumeration.
Directly interfaces with dxgi.dll via ctypes to discover GPUs, discrete/integrated
distinctions, dedicated video memory, and DirectML acceleration suitability.
Provides graceful fallback for non-Windows platforms and environments without DXGI.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import logging
import os
import platform
from typing import List, Optional, Set, Tuple

logger = logging.getLogger("ngam.dxgi")

# Known Hardware Vendor IDs (PCI SIG)
VENDOR_NVIDIA: int = 0x10DE
VENDOR_AMD: int = 0x1002
VENDOR_INTEL: int = 0x8086
VENDOR_MICROSOFT: int = 0x1414
VENDOR_QUALCOMM: int = 0x5143

# Known vendor name mappings for display/logging
VENDOR_NAMES = {
    VENDOR_NVIDIA: "NVIDIA",
    VENDOR_AMD: "AMD",
    VENDOR_INTEL: "Intel",
    VENDOR_MICROSOFT: "Microsoft",
    VENDOR_QUALCOMM: "Qualcomm",
}

# Recognized discrete GPU vendor IDs
DISCRETE_VENDORS = (VENDOR_NVIDIA, VENDOR_AMD, VENDOR_INTEL)

# DXGI Adapter Flags (DXGI_ADAPTER_FLAG enum)
DXGI_ADAPTER_FLAG_NONE: int = 0
DXGI_ADAPTER_FLAG_REMOTE: int = 1
DXGI_ADAPTER_FLAG_SOFTWARE: int = 2

# DXGI HRESULT status codes
S_OK: int = 0x00000000
DXGI_ERROR_NOT_FOUND: int = 0x887A0002


# ---------------------------------------------------------------------------
# COM Structures for DXGI 1.1
# ---------------------------------------------------------------------------

class _LUID(ctypes.Structure):
    _fields_ = [
        ("LowPart", ctypes.c_uint32),
        ("HighPart", ctypes.c_int32),
    ]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint32),
    ]


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_uint8 * 8),
    ]


# IID_IDXGIFactory1 = {770aae78-f26f-4dba-a829-253c83d1b387}
_IID_IDXGIFactory1 = _GUID(
    0x770AAE78,
    0xF26F,
    0x4DBA,
    (ctypes.c_uint8 * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
)

# Portable function prototype helper
_WIN_FUNC = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
_HRESULT = ctypes.c_long
_ULONG = ctypes.c_uint32

# COM vtable signatures
_Release_proto = _WIN_FUNC(_ULONG, ctypes.c_void_p)
_EnumAdapters1_proto = _WIN_FUNC(
    _HRESULT,
    ctypes.c_void_p,                  # this (IDXGIFactory1*)
    ctypes.c_uint32,                  # Adapter index
    ctypes.POINTER(ctypes.c_void_p),  # ppAdapter (IDXGIAdapter1**)
)
_GetDesc1_proto = _WIN_FUNC(
    _HRESULT,
    ctypes.c_void_p,                      # this (IDXGIAdapter1*)
    ctypes.POINTER(_DXGI_ADAPTER_DESC1),  # pDesc
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DXGIAdapterInfo:
    """Detailed information for an enumerated DXGI graphics adapter."""
    device_id: int
    name: str
    vendor_id: int
    subsystem_id: int
    dedicated_vram_mb: float
    shared_ram_mb: float
    is_software: bool
    is_discrete: bool
    is_hardware_dml_capable: bool

    @property
    def vendor_name(self) -> str:
        """Return human-readable vendor name if recognized."""
        return VENDOR_NAMES.get(self.vendor_id, f"Unknown (0x{self.vendor_id:04X})")


# ---------------------------------------------------------------------------
# Adapter Classification Helpers
# ---------------------------------------------------------------------------

def is_software_adapter(flags: int, vendor_id: int) -> bool:
    """
    Determine if an adapter is a software renderer (e.g. WARP, Basic Render Driver).
    Checks DXGI_ADAPTER_FLAG_SOFTWARE (Flags & 0x2) and Microsoft Basic Render (0x1414).
    """
    return bool((flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 or vendor_id == VENDOR_MICROSOFT)


def is_discrete_gpu(vendor_id: int, dedicated_vram_mb: float, is_software: bool = False) -> bool:
    """
    Determine if an adapter is a discrete GPU.
    Recognizes NVIDIA (0x10DE), AMD (0x1002), and Intel Arc (0x8086) with >= 512 MB dedicated VRAM.
    Intel iGPUs typically report 0 MB dedicated VRAM.
    """
    if is_software:
        return False
    return bool(vendor_id in DISCRETE_VENDORS and dedicated_vram_mb >= 512.0)


def is_hardware_dml_capable(
    vendor_id: int,
    dedicated_vram_mb: float,
    shared_ram_mb: float,
    is_software: bool = False,
) -> bool:
    """
    Determine if an adapter can hardware-accelerate DirectML workloads.
    Requires: not a software adapter, vendor != 0x1414, and either dedicated VRAM > 0
    or shared system RAM >= 2048 MB (for modern iGPUs / APUs).
    """
    if is_software or vendor_id == VENDOR_MICROSOFT:
        return False
    return bool(dedicated_vram_mb > 0.0 or shared_ram_mb >= 2048.0)


# ---------------------------------------------------------------------------
# Core Public API
# ---------------------------------------------------------------------------

def enumerate_adapters() -> List[DXGIAdapterInfo]:
    """
    Enumerate all DXGI 1.1 adapters using zero-dependency ctypes.
    Safely returns an empty list on non-Windows platforms or if dxgi.dll cannot be loaded.
    """
    if platform.system() != "Windows":
        logger.debug("DXGI adapter enumeration skipped: non-Windows OS (%s)", platform.system())
        return []

    try:
        dxgi_dll = ctypes.WinDLL("dxgi.dll")
        create_factory1 = dxgi_dll.CreateDXGIFactory1
        create_factory1.argtypes = [
            ctypes.POINTER(_GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        create_factory1.restype = _HRESULT
    except (OSError, AttributeError) as exc:
        logger.warning("Failed to load dxgi.dll or locate CreateDXGIFactory1: %s", exc)
        return []

    factory_p = ctypes.c_void_p()
    hr = create_factory1(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory_p))
    if hr != 0 or not factory_p.value:
        logger.warning("CreateDXGIFactory1 failed with HRESULT: 0x%08X", hr & 0xFFFFFFFF)
        return []

    adapters: List[DXGIAdapterInfo] = []
    seen_configs: Set[Tuple[int, int, int, int]] = set()
    seen_luids: Set[Tuple[int, int]] = set()
    try:
        vtable_factory = ctypes.cast(
            factory_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
        ).contents
        enum_adapters1 = _EnumAdapters1_proto(vtable_factory[12])

        adapter_index = 0
        while True:
            adapter_p = ctypes.c_void_p()
            hr = enum_adapters1(factory_p, adapter_index, ctypes.byref(adapter_p))
            if hr != 0 or not adapter_p.value:
                # DXGI_ERROR_NOT_FOUND or finished enumeration
                break

            try:
                vtable_adapter = ctypes.cast(
                    adapter_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
                ).contents
                get_desc1 = _GetDesc1_proto(vtable_adapter[10])

                desc = _DXGI_ADAPTER_DESC1()
                hr_desc = get_desc1(adapter_p, ctypes.byref(desc))
                if hr_desc == 0:
                    name = str(desc.Description).split("\0")[0].strip()
                    vendor_id = int(desc.VendorId)
                    subsystem_id = int(desc.SubSysId)
                    flags = int(desc.Flags)
                    dedicated_vram_raw = int(desc.DedicatedVideoMemory)
                    shared_ram_raw = int(desc.SharedSystemMemory)
                    luid_tuple = (int(desc.AdapterLuid.HighPart), int(desc.AdapterLuid.LowPart))

                    adapter_key = (vendor_id, subsystem_id, dedicated_vram_raw, shared_ram_raw)
                    if luid_tuple != (0, 0):
                        is_duplicate = luid_tuple in seen_luids
                        if not is_duplicate:
                            seen_luids.add(luid_tuple)
                    else:
                        is_duplicate = adapter_key in seen_configs
                        if not is_duplicate:
                            seen_configs.add(adapter_key)

                    if not is_duplicate:
                        dedicated_vram_mb = round(float(dedicated_vram_raw) / (1024.0 * 1024.0), 2)
                        shared_ram_mb = round(float(shared_ram_raw) / (1024.0 * 1024.0), 2)

                        software = is_software_adapter(flags, vendor_id)
                        discrete = is_discrete_gpu(vendor_id, dedicated_vram_mb, is_software=software)
                        dml_capable = is_hardware_dml_capable(
                            vendor_id, dedicated_vram_mb, shared_ram_mb, is_software=software
                        )

                        info = DXGIAdapterInfo(
                            device_id=adapter_index,
                            name=name,
                            vendor_id=vendor_id,
                            subsystem_id=subsystem_id,
                            dedicated_vram_mb=dedicated_vram_mb,
                            shared_ram_mb=shared_ram_mb,
                            is_software=software,
                            is_discrete=discrete,
                            is_hardware_dml_capable=dml_capable,
                        )
                        adapters.append(info)
                    else:
                        logger.debug("Skipping duplicate DXGI adapter at index %d (%s)", adapter_index, name)
                else:
                    logger.debug("GetDesc1 failed for adapter index %d (hr=0x%08X)", adapter_index, hr_desc & 0xFFFFFFFF)
            finally:
                if adapter_p.value:
                    try:
                        vtable_adapter = ctypes.cast(
                            adapter_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
                        ).contents
                        _Release_proto(vtable_adapter[2])(adapter_p)
                    except Exception:
                        pass

            adapter_index += 1
    except Exception as exc:
        logger.warning("Unexpected error during DXGI adapter enumeration: %s", exc)
    finally:
        if factory_p.value:
            try:
                vtable_factory = ctypes.cast(
                    factory_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
                ).contents
                _Release_proto(vtable_factory[2])(factory_p)
            except Exception:
                pass

    return adapters


def get_hardware_adapters() -> List[DXGIAdapterInfo]:
    """
    Return filtered hardware-accelerated DXGI adapters (excluding software/WARP adapters),
    sorted with discrete GPUs first, then by dedicated VRAM descending.
    """
    all_adapters = enumerate_adapters()
    hw_adapters = [a for a in all_adapters if not a.is_software]
    hw_adapters.sort(
        key=lambda a: (a.is_discrete, a.dedicated_vram_mb, a.shared_ram_mb, -a.device_id),
        reverse=True,
    )
    return hw_adapters


def get_best_dml_device_id() -> Optional[int]:
    """
    Select the best DirectML device ID for hardware acceleration.
    Supports NGAM_DML_DEVICE_ID (primary) and AGY_DML_DEVICE_ID (fallback) environment overrides.

    Returns:
        The device_id of the best hardware adapter (discrete GPU preferred,
        followed by high-VRAM iGPU), or None if only software/WARP adapters
        exist or if no hardware DML-capable adapters are present.
    """
    env_override = os.environ.get("NGAM_DML_DEVICE_ID") or os.environ.get("AGY_DML_DEVICE_ID")
    if env_override:
        raw = env_override.strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                logger.warning("Malformed DML device ID '%s', expected integer.", env_override)

    hw_adapters = get_hardware_adapters()
    dml_capable = [a for a in hw_adapters if a.is_hardware_dml_capable]
    if dml_capable:
        return dml_capable[0].device_id
    return None


__all__ = [
    "DXGIAdapterInfo",
    "enumerate_adapters",
    "get_hardware_adapters",
    "get_best_dml_device_id",
    "is_software_adapter",
    "is_discrete_gpu",
    "is_hardware_dml_capable",
    "VENDOR_NVIDIA",
    "VENDOR_AMD",
    "VENDOR_INTEL",
    "VENDOR_MICROSOFT",
    "VENDOR_QUALCOMM",
    "DXGI_ADAPTER_FLAG_NONE",
    "DXGI_ADAPTER_FLAG_REMOTE",
    "DXGI_ADAPTER_FLAG_SOFTWARE",
]


# ---------------------------------------------------------------------------
# Self-Test and CLI Inspection Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 72)
    print("  ngam.dxgi - DXGI 1.1 Hardware Topology & DirectML Probe")
    print("=" * 72)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:   {sys.version.split()[0]}")
    print("-" * 72)

    all_adapters = enumerate_adapters()
    print(f"Total DXGI Adapters Detected: {len(all_adapters)}\n")

    if not all_adapters:
        print("  No DXGI adapters detected (non-Windows platform or headless VM).")
    else:
        for a in all_adapters:
            sw_str = "YES (Software)" if a.is_software else "No (Hardware)"
            disc_str = "YES (Discrete)" if a.is_discrete else "No (Integrated/WARP)"
            dml_str = "YES (Capable)" if a.is_hardware_dml_capable else "No"
            print(f"  [Adapter {a.device_id}] {a.name}")
            print(f"    Vendor:         {a.vendor_name} (0x{a.vendor_id:04X})")
            print(f"    Subsystem ID:   0x{a.subsystem_id:04X}")
            print(f"    Dedicated VRAM: {a.dedicated_vram_mb:,.1f} MB")
            print(f"    Shared RAM:     {a.shared_ram_mb:,.1f} MB")
            print(f"    Software:       {sw_str}")
            print(f"    Discrete GPU:   {disc_str}")
            print(f"    DML Capable:    {dml_str}")
            print()

    hw_adapters = get_hardware_adapters()
    print("-" * 72)
    print(f"Filtered Hardware Adapters: {len(hw_adapters)}")
    for rank, a in enumerate(hw_adapters, start=1):
        tier = "dGPU" if a.is_discrete else "iGPU"
        print(f"  Rank {rank}: Device {a.device_id} ({a.name}) [{tier}] - {a.dedicated_vram_mb:,.1f} MB VRAM")

    best_id = get_best_dml_device_id()
    print("-" * 72)
    print(f"Best DirectML Device ID: {best_id}")
    if best_id is not None:
        best_adapter = next(a for a in hw_adapters if a.device_id == best_id)
        print(f"  -> Selected: [{best_adapter.device_id}] {best_adapter.name} ({best_adapter.vendor_name})")
    else:
        print("  -> DirectML Hardware Acceleration: Not Available (falling back to CPU)")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Unit Tests for Synthetic Topologies & Graceful Fallbacks
    # -----------------------------------------------------------------------
    print("\nRunning synthetic topology verification tests...")

    # Test 1: Software detection
    assert is_software_adapter(2, 0x10DE) is True, "Software flag 0x2 must flag software"
    assert is_software_adapter(0, 0x1414) is True, "Vendor 0x1414 must flag software"
    assert is_software_adapter(0, 0x10DE) is False, "NVIDIA hardware must not be software"

    # Test 2: Discrete classification
    assert is_discrete_gpu(0x10DE, 6144.0) is True, "NVIDIA 6GB dGPU"
    assert is_discrete_gpu(0x1002, 8192.0) is True, "AMD 8GB dGPU"
    assert is_discrete_gpu(0x8086, 16384.0) is True, "Intel Arc 16GB dGPU"
    assert is_discrete_gpu(0x8086, 0.0) is False, "Intel iGPU (0 VRAM)"
    assert is_discrete_gpu(0x8086, 128.0) is False, "Intel iGPU (< 512MB VRAM)"
    assert is_discrete_gpu(0x1414, 0.0) is False, "Microsoft WARP"

    # Test 3: Hardware DML capability
    assert is_hardware_dml_capable(0x10DE, 4096.0, 8192.0) is True, "dGPU with VRAM"
    assert is_hardware_dml_capable(0x8086, 0.0, 16384.0) is True, "iGPU with >= 2GB shared RAM"
    assert is_hardware_dml_capable(0x8086, 0.0, 1024.0) is False, "iGPU with < 2GB shared RAM"
    assert is_hardware_dml_capable(0x1414, 0.0, 16384.0) is False, "Microsoft software driver"

    # Test 4: Priority ranking - dGPU at index 1 vs iGPU at index 0
    mock_igpu = DXGIAdapterInfo(
        device_id=0,
        name="Intel UHD Graphics 770",
        vendor_id=0x8086,
        subsystem_id=0x1234,
        dedicated_vram_mb=0.0,
        shared_ram_mb=16384.0,
        is_software=False,
        is_discrete=False,
        is_hardware_dml_capable=True,
    )
    mock_dgpu = DXGIAdapterInfo(
        device_id=1,
        name="NVIDIA GeForce RTX 4080",
        vendor_id=0x10DE,
        subsystem_id=0x5678,
        dedicated_vram_mb=16384.0,
        shared_ram_mb=16384.0,
        is_software=False,
        is_discrete=True,
        is_hardware_dml_capable=True,
    )
    mock_warp = DXGIAdapterInfo(
        device_id=2,
        name="Microsoft Basic Render Driver",
        vendor_id=0x1414,
        subsystem_id=0x0000,
        dedicated_vram_mb=0.0,
        shared_ram_mb=16384.0,
        is_software=True,
        is_discrete=False,
        is_hardware_dml_capable=False,
    )

    mod = __name__

    with patch(f"{mod}.enumerate_adapters", return_value=[mock_igpu, mock_dgpu, mock_warp]):
        hw = get_hardware_adapters()
        assert len(hw) == 2, f"Expected 2 hardware adapters, got {len(hw)}"
        assert hw[0].device_id == 1, f"Expected dGPU (id=1) first, got {hw[0].device_id}"
        assert hw[1].device_id == 0, f"Expected iGPU (id=0) second, got {hw[1].device_id}"
        best = get_best_dml_device_id()
        assert best == 1, f"Expected DML device id 1, got {best}"

    # Test 5: iGPU only -> selects iGPU (device_id=0)
    with patch(f"{mod}.enumerate_adapters", return_value=[mock_igpu, mock_warp]):
        hw = get_hardware_adapters()
        assert len(hw) == 1
        assert hw[0].device_id == 0
        best = get_best_dml_device_id()
        assert best == 0, f"Expected DML device id 0, got {best}"

    # Test 6: Software-only pool returns None
    with patch(f"{mod}.enumerate_adapters", return_value=[mock_warp]):
        hw = get_hardware_adapters()
        assert len(hw) == 0, "Software-only pool yields empty hardware list"
        best = get_best_dml_device_id()
        assert best is None, f"Expected None for software-only, got {best}"

    # Test 7: Non-Windows fallback
    with patch(f"{mod}.platform.system", return_value="Linux"):
        assert enumerate_adapters() == []
        assert get_hardware_adapters() == []
        assert get_best_dml_device_id() is None

    # Test 8: WinDLL loading failure fallback
    with patch(f"{mod}.ctypes.WinDLL", side_effect=OSError("dxgi.dll not found")):
        with patch.object(logger, "warning"):
            assert enumerate_adapters() == []
            assert get_hardware_adapters() == []
            assert get_best_dml_device_id() is None

    # Test 9: Environment variable override (NGAM_DML_DEVICE_ID and AGY_DML_DEVICE_ID)
    with patch.dict(os.environ, {"NGAM_DML_DEVICE_ID": "3"}):
        assert get_best_dml_device_id() == 3

    with patch.dict(os.environ, {"AGY_DML_DEVICE_ID": "4"}, clear=True):
        assert get_best_dml_device_id() == 4

    print("All synthetic topology and fallback verification tests passed successfully!")
