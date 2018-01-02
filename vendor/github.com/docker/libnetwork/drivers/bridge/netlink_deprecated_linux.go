package bridge

import (
	"fmt"
	"math/rand"
	"net"
	"syscall"
	"time"
	"unsafe"

	"github.com/docker/libnetwork/netutils"
)

const (
	ifNameSize   = 16
	ioctlBrAdd   = 0x89a0
	ioctlBrAddIf = 0x89a2
)

type ifreqIndex struct {
	IfrnName  [ifNameSize]byte
	IfruIndex int32
}

type ifreqHwaddr struct {
	IfrnName   [ifNameSize]byte
	IfruHwaddr syscall.RawSockaddr
}

var rnd = rand.New(rand.NewSource(time.Now().UnixNano()))

// THIS CODE DOES NOT COMMUNICATE WITH KERNEL VIA RTNETLINK INTERFACE
// IT IS HERE FOR BACKWARDS COMPATIBILITY WITH OLDER LINUX KERNELS
// WHICH SHIP WITH OLDER NOT ENTIRELY FUNCTIONAL VERSION OF NETLINK
func getIfSocket() (fd int, err error) {
	for _, socket := range []int{
		syscall.AF_INET,
		syscall.AF_PACKET,
		syscall.AF_INET6,
	} {
		if fd, err = syscall.Socket(socket, syscall.SOCK_DGRAM, 0); err == nil {
			break
		}
	}
	if err == nil {
		return fd, nil
	}
	return -1, err
}

func ifIoctBridge(iface, master *net.Interface, op uintptr) error {
	if len(master.Name) >= ifNameSize {
		return fmt.Errorf("Interface name %s too long", master.Name)
	}

	s, err := getIfSocket()
	if err != nil {
		return err
	}
	defer syscall.Close(s)

	ifr := ifreqIndex{}
	copy(ifr.IfrnName[:len(ifr.IfrnName)-1], master.Name)
	ifr.IfruIndex = int32(iface.Index)

	if _, _, err := syscall.Syscall(syscall.SYS_IOCTL, uintptr(s), op, uintptr(unsafe.Pointer(&ifr))); err != 0 {
		return err
	}

	return nil
}

// Add a slave to a bridge device.  This is more backward-compatible than
// netlink.NetworkSetMaster and works on RHEL 6.
func ioctlAddToBridge(iface, master *net.Interface) error {
	return ifIoctBridge(iface, master, ioctlBrAddIf)
}

func ioctlSetMacAddress(name, addr string) error {
	if len(name) >= ifNameSize {
		return fmt.Errorf("Interface name %s too long", name)
	}

	hw, err := net.ParseMAC(addr)
	if err != nil {
		return err
	}

	s, err := getIfSocket()
	if err != nil {
		return err
	}
	defer syscall.Close(s)

	ifr := ifreqHwaddr{}
	ifr.IfruHwaddr.Family = syscall.ARPHRD_ETHER
	copy(ifr.IfrnName[:len(ifr.IfrnName)-1], name)

	for i := 0; i < 6; i++ {
		ifr.IfruHwaddr.Data[i] = ifrDataByte(hw[i])
	}

	if _, _, err := syscall.Syscall(syscall.SYS_IOCTL, uintptr(s), syscall.SIOCSIFHWADDR, uintptr(unsafe.Pointer(&ifr))); err != 0 {
		return err
	}
	return nil
}

func ioctlCreateBridge(name string, setMacAddr bool) error {
	if len(name) >= ifNameSize {
		return fmt.Errorf("Interface name %s too long", name)
	}

	s, err := getIfSocket()
	if err != nil {
		return err
	}
	defer syscall.Close(s)

	nameBytePtr, err := syscall.BytePtrFromString(name)
	if err != nil {
		return err
	}
	if _, _, err := syscall.Syscall(syscall.SYS_IOCTL, uintptr(s), ioctlBrAdd, uintptr(unsafe.Pointer(nameBytePtr))); err != 0 {
		return err
	}
	if setMacAddr {
		return ioctlSetMacAddress(name, netutils.GenerateRandomMAC().String())
	}
	return nil
}
