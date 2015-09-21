package operatingsystem

import (
	"syscall"
	"unsafe"
)

// See https://code.google.com/p/go/source/browse/src/pkg/mime/type_windows.go?r=d14520ac25bf6940785aabb71f5be453a286f58c
// for a similar sample

func GetOperatingSystem() (string, error) {

	var h syscall.Handle

	// Default return value
	ret := "Unknown Operating System"

	if err := syscall.RegOpenKeyEx(syscall.HKEY_LOCAL_MACHINE,
		syscall.StringToUTF16Ptr(`SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\`),
		0,
		syscall.KEY_READ,
		&h); err != nil {
		return ret, err
	}
	defer syscall.RegCloseKey(h)

	var buf [1 << 10]uint16
	var typ uint32
	n := uint32(len(buf) * 2) // api expects array of bytes, not uint16

	if err := syscall.RegQueryValueEx(h,
		syscall.StringToUTF16Ptr("ProductName"),
		nil,
		&typ,
		(*byte)(unsafe.Pointer(&buf[0])),
		&n); err != nil {
		return ret, err
	}
	ret = syscall.UTF16ToString(buf[:])

	return ret, nil
}

// No-op on Windows
func IsContainerized() (bool, error) {
	return false, nil
}
