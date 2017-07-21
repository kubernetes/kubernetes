// +build linux

package water

import (
	"os"
	"strings"
	"syscall"
	"unsafe"
)

const (
	cIFF_TUN   = 0x0001
	cIFF_TAP   = 0x0002
	cIFF_NO_PI = 0x1000
)

type ifReq struct {
	Name  [0x10]byte
	Flags uint16
	pad   [0x28 - 0x10 - 2]byte
}

func ioctl(fd uintptr, request int, argp uintptr) error {
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, fd, uintptr(request), argp)
	if errno != 0 {
		return os.NewSyscallError("ioctl", errno)
	}
	return nil
}

func newTAP(config Config) (ifce *Interface, err error) {
	file, err := os.OpenFile("/dev/net/tun", os.O_RDWR, 0)
	if err != nil {
		return nil, err
	}
	name, err := createInterface(file.Fd(), config.Name, cIFF_TAP|cIFF_NO_PI)
	if err != nil {
		return nil, err
	}

	if err = setDeviceOptions(file.Fd(), config); err != nil {
		return nil, err
	}

	ifce = &Interface{isTAP: true, ReadWriteCloser: file, name: name}
	return
}

func newTUN(config Config) (ifce *Interface, err error) {
	file, err := os.OpenFile("/dev/net/tun", os.O_RDWR, 0)
	if err != nil {
		return nil, err
	}
	name, err := createInterface(file.Fd(), config.Name, cIFF_TUN|cIFF_NO_PI)
	if err != nil {
		return nil, err
	}

	if err = setDeviceOptions(file.Fd(), config); err != nil {
		return nil, err
	}

	ifce = &Interface{isTAP: false, ReadWriteCloser: file, name: name}
	return
}

func createInterface(fd uintptr, ifName string, flags uint16) (createdIFName string, err error) {
	var req ifReq
	req.Flags = flags
	copy(req.Name[:], ifName)

	err = ioctl(fd, syscall.TUNSETIFF, uintptr(unsafe.Pointer(&req)))
	if err != nil {
		return
	}

	createdIFName = strings.Trim(string(req.Name[:]), "\x00")
	return
}

func setDeviceOptions(fd uintptr, config Config) (err error) {

	// Device Permissions
	if config.Permissions != nil {

		// Set Owner
		if err = ioctl(fd, syscall.TUNSETOWNER, uintptr(config.Permissions.Owner)); err != nil {
			return
		}

		// Set Group
		if err = ioctl(fd, syscall.TUNSETGROUP, uintptr(config.Permissions.Group)); err != nil {
			return
		}
	}

	// Set/Clear Persist Device Flag
	value := 0
	if config.Persist {
		value = 1
	}
	return ioctl(fd, syscall.TUNSETPERSIST, uintptr(value))

}
