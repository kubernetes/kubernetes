package kernel

import (
	"fmt"
	"syscall"
	"unsafe"
)

// VersionInfo holds information about the kernel.
type VersionInfo struct {
	kvi   string // Version of the kernel (e.g. 6.1.7601.17592 -> 6)
	major int    // Major part of the kernel version (e.g. 6.1.7601.17592 -> 1)
	minor int    // Minor part of the kernel version (e.g. 6.1.7601.17592 -> 7601)
	build int    // Build number of the kernel version (e.g. 6.1.7601.17592 -> 17592)
}

func (k *VersionInfo) String() string {
	return fmt.Sprintf("%d.%d %d (%s)", k.major, k.minor, k.build, k.kvi)
}

// GetKernelVersion gets the current kernel version.
func GetKernelVersion() (*VersionInfo, error) {

	var (
		h         syscall.Handle
		dwVersion uint32
		err       error
	)

	KVI := &VersionInfo{"Unknown", 0, 0, 0}

	if err = syscall.RegOpenKeyEx(syscall.HKEY_LOCAL_MACHINE,
		syscall.StringToUTF16Ptr(`SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\`),
		0,
		syscall.KEY_READ,
		&h); err != nil {
		return KVI, err
	}
	defer syscall.RegCloseKey(h)

	var buf [1 << 10]uint16
	var typ uint32
	n := uint32(len(buf) * 2) // api expects array of bytes, not uint16

	if err = syscall.RegQueryValueEx(h,
		syscall.StringToUTF16Ptr("BuildLabEx"),
		nil,
		&typ,
		(*byte)(unsafe.Pointer(&buf[0])),
		&n); err != nil {
		return KVI, err
	}

	KVI.kvi = syscall.UTF16ToString(buf[:])

	// Important - docker.exe MUST be manifested for this API to return
	// the correct information.
	if dwVersion, err = syscall.GetVersion(); err != nil {
		return KVI, err
	}

	KVI.major = int(dwVersion & 0xFF)
	KVI.minor = int((dwVersion & 0XFF00) >> 8)
	KVI.build = int((dwVersion & 0xFFFF0000) >> 16)

	return KVI, nil
}
