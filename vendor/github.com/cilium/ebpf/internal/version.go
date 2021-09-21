package internal

import (
	"fmt"
	"io/ioutil"
	"regexp"
	"sync"

	"github.com/cilium/ebpf/internal/unix"
)

const (
	// Version constant used in ELF binaries indicating that the loader needs to
	// substitute the eBPF program's version with the value of the kernel's
	// KERNEL_VERSION compile-time macro. Used for compatibility with BCC, gobpf
	// and RedSift.
	MagicKernelVersion = 0xFFFFFFFE
)

var (
	// Match between one and three decimals separated by dots, with the last
	// segment (patch level) being optional on some kernels.
	// The x.y.z string must appear at the start of a string or right after
	// whitespace to prevent sequences like 'x.y.z-a.b.c' from matching 'a.b.c'.
	rgxKernelVersion = regexp.MustCompile(`(?:\A|\s)\d{1,3}\.\d{1,3}(?:\.\d{1,3})?`)

	kernelVersion = struct {
		once    sync.Once
		version Version
		err     error
	}{}
)

// A Version in the form Major.Minor.Patch.
type Version [3]uint16

// NewVersion creates a version from a string like "Major.Minor.Patch".
//
// Patch is optional.
func NewVersion(ver string) (Version, error) {
	var major, minor, patch uint16
	n, _ := fmt.Sscanf(ver, "%d.%d.%d", &major, &minor, &patch)
	if n < 2 {
		return Version{}, fmt.Errorf("invalid version: %s", ver)
	}
	return Version{major, minor, patch}, nil
}

func (v Version) String() string {
	if v[2] == 0 {
		return fmt.Sprintf("v%d.%d", v[0], v[1])
	}
	return fmt.Sprintf("v%d.%d.%d", v[0], v[1], v[2])
}

// Less returns true if the version is less than another version.
func (v Version) Less(other Version) bool {
	for i, a := range v {
		if a == other[i] {
			continue
		}
		return a < other[i]
	}
	return false
}

// Unspecified returns true if the version is all zero.
func (v Version) Unspecified() bool {
	return v[0] == 0 && v[1] == 0 && v[2] == 0
}

// Kernel implements the kernel's KERNEL_VERSION macro from linux/version.h.
// It represents the kernel version and patch level as a single value.
func (v Version) Kernel() uint32 {

	// Kernels 4.4 and 4.9 have their SUBLEVEL clamped to 255 to avoid
	// overflowing into PATCHLEVEL.
	// See kernel commit 9b82f13e7ef3 ("kbuild: clamp SUBLEVEL to 255").
	s := v[2]
	if s > 255 {
		s = 255
	}

	// Truncate members to uint8 to prevent them from spilling over into
	// each other when overflowing 8 bits.
	return uint32(uint8(v[0]))<<16 | uint32(uint8(v[1]))<<8 | uint32(uint8(s))
}

// KernelVersion returns the version of the currently running kernel.
func KernelVersion() (Version, error) {
	kernelVersion.once.Do(func() {
		kernelVersion.version, kernelVersion.err = detectKernelVersion()
	})

	if kernelVersion.err != nil {
		return Version{}, kernelVersion.err
	}
	return kernelVersion.version, nil
}

// detectKernelVersion returns the version of the running kernel. It scans the
// following sources in order: /proc/version_signature, uname -v, uname -r.
// In each of those locations, the last-appearing x.y(.z) value is selected
// for parsing. The first location that yields a usable version number is
// returned.
func detectKernelVersion() (Version, error) {

	// Try reading /proc/version_signature for Ubuntu compatibility.
	// Example format: Ubuntu 4.15.0-91.92-generic 4.15.18
	// This method exists in the kernel itself, see d18acd15c
	// ("perf tools: Fix kernel version error in ubuntu").
	if pvs, err := ioutil.ReadFile("/proc/version_signature"); err == nil {
		// If /proc/version_signature exists, failing to parse it is an error.
		// It only exists on Ubuntu, where the real patch level is not obtainable
		// through any other method.
		v, err := findKernelVersion(string(pvs))
		if err != nil {
			return Version{}, err
		}
		return v, nil
	}

	var uname unix.Utsname
	if err := unix.Uname(&uname); err != nil {
		return Version{}, fmt.Errorf("calling uname: %w", err)
	}

	// Debian puts the version including the patch level in uname.Version.
	// It is not an error if there's no version number in uname.Version,
	// as most distributions don't use it. Parsing can continue on uname.Release.
	// Example format: #1 SMP Debian 4.19.37-5+deb10u2 (2019-08-08)
	if v, err := findKernelVersion(unix.ByteSliceToString(uname.Version[:])); err == nil {
		return v, nil
	}

	// Most other distributions have the full kernel version including patch
	// level in uname.Release.
	// Example format: 4.19.0-5-amd64, 5.5.10-arch1-1
	v, err := findKernelVersion(unix.ByteSliceToString(uname.Release[:]))
	if err != nil {
		return Version{}, err
	}

	return v, nil
}

// findKernelVersion matches s against rgxKernelVersion and parses the result
// into a Version. If s contains multiple matches, the last entry is selected.
func findKernelVersion(s string) (Version, error) {
	m := rgxKernelVersion.FindAllString(s, -1)
	if m == nil {
		return Version{}, fmt.Errorf("no kernel version in string: %s", s)
	}
	// Pick the last match of the string in case there are multiple.
	s = m[len(m)-1]

	v, err := NewVersion(s)
	if err != nil {
		return Version{}, fmt.Errorf("parsing version string %s: %w", s, err)
	}

	return v, nil
}
