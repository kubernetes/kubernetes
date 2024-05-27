package internal

import (
	"fmt"
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

// NewVersionFromCode creates a version from a LINUX_VERSION_CODE.
func NewVersionFromCode(code uint32) Version {
	return Version{
		uint16(uint8(code >> 16)),
		uint16(uint8(code >> 8)),
		uint16(uint8(code)),
	}
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
var KernelVersion = sync.OnceValues(func() (Version, error) {
	return detectKernelVersion()
})

// detectKernelVersion returns the version of the running kernel.
func detectKernelVersion() (Version, error) {
	vc, err := vdsoVersion()
	if err != nil {
		return Version{}, err
	}
	return NewVersionFromCode(vc), nil
}

// KernelRelease returns the release string of the running kernel.
// Its format depends on the Linux distribution and corresponds to directory
// names in /lib/modules by convention. Some examples are 5.15.17-1-lts and
// 4.19.0-16-amd64.
func KernelRelease() (string, error) {
	var uname unix.Utsname
	if err := unix.Uname(&uname); err != nil {
		return "", fmt.Errorf("uname failed: %w", err)
	}

	return unix.ByteSliceToString(uname.Release[:]), nil
}
