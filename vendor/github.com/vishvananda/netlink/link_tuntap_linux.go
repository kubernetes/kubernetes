package netlink

import (
	"fmt"
	"os"
	"strings"
	"syscall"

	"golang.org/x/sys/unix"
)

// ideally golang.org/x/sys/unix would define IfReq but it only has
// IFNAMSIZ, hence this minimalistic implementation
const (
	SizeOfIfReq = 40
	IFNAMSIZ    = 16
)

const TUN = "/dev/net/tun"

type ifReq struct {
	Name  [IFNAMSIZ]byte
	Flags uint16
	pad   [SizeOfIfReq - IFNAMSIZ - 2]byte
}

// AddQueues opens and attaches multiple queue file descriptors to an existing
// TUN/TAP interface in multi-queue mode.
//
// It performs TUNSETIFF ioctl on each opened file descriptor with the current
// tuntap configuration. Each resulting fd is set to non-blocking mode and
// returned as *os.File.
//
// If the interface was created with a name pattern (e.g. "tap%d"),
// the first successful TUNSETIFF call will return the resolved name,
// which is saved back into tuntap.Name.
//
// This method assumes that the interface already exists and is in multi-queue mode.
// The returned FDs are also appended to tuntap.Fds and tuntap.Queues is updated.
//
// It is the caller's responsibility to close the FDs when they are no longer needed.
func (tuntap *Tuntap) AddQueues(count int) ([]*os.File, error) {
	if tuntap.Mode < unix.IFF_TUN || tuntap.Mode > unix.IFF_TAP {
		return nil, fmt.Errorf("Tuntap.Mode %v unknown", tuntap.Mode)
	}
	if tuntap.Flags&TUNTAP_MULTI_QUEUE == 0 {
		return nil, fmt.Errorf("TUNTAP_MULTI_QUEUE not set")
	}
	if count < 1 {
		return nil, fmt.Errorf("count must be >= 1")
	}

	req, err := unix.NewIfreq(tuntap.Name)
	if err != nil {
		return nil, err
	}
	req.SetUint16(uint16(tuntap.Mode) | uint16(tuntap.Flags))

	var fds []*os.File
	for i := 0; i < count; i++ {
		localReq := req
		fd, err := unix.Open(TUN, os.O_RDWR|syscall.O_CLOEXEC, 0)
		if err != nil {
			cleanupFds(fds)
			return nil, err
		}

		err = unix.IoctlIfreq(fd, unix.TUNSETIFF, req)
		if err != nil {
			// close the new fd
			unix.Close(fd)
			// and the already opened ones
			cleanupFds(fds)
			return nil, fmt.Errorf("tuntap IOCTL TUNSETIFF failed [%d]: %w", i, err)
		}

		// Set the tun device to non-blocking before use. The below comment
		// taken from:
		//
		// https://github.com/mistsys/tuntap/commit/161418c25003bbee77d085a34af64d189df62bea
		//
		// Note there is a complication because in go, if a device node is
		// opened, go sets it to use nonblocking I/O. However a /dev/net/tun
		// doesn't work with epoll until after the TUNSETIFF ioctl has been
		// done. So we open the unix fd directly, do the ioctl, then put the
		// fd in nonblocking mode, an then finally wrap it in a os.File,
		// which will see the nonblocking mode and add the fd to the
		// pollable set, so later on when we Read() from it blocked the
		// calling thread in the kernel.
		//
		// See
		//   https://github.com/golang/go/issues/30426
		// which got exposed in go 1.13 by the fix to
		//   https://github.com/golang/go/issues/30624
		err = unix.SetNonblock(fd, true)
		if err != nil {
			cleanupFds(fds)
			return nil, fmt.Errorf("tuntap set to non-blocking failed [%d]: %w", i, err)
		}

		// create the file from the file descriptor and store it
		file := os.NewFile(uintptr(fd), TUN)
		fds = append(fds, file)

		// 1) we only care for the name of the first tap in the multi queue set
		// 2) if the original name was empty, the localReq has now the actual name
		//
		// In addition:
		// This ensures that the link name is always identical to what the kernel returns.
		// Not only in case of an empty name, but also when using name templates.
		// e.g. when the provided name is "tap%d", the kernel replaces %d with the next available number.
		if i == 0 {
			tuntap.Name = strings.Trim(localReq.Name(), "\x00")
		}
	}

	tuntap.Fds = append(tuntap.Fds, fds...)
	tuntap.Queues = len(tuntap.Fds)
	return fds, nil
}

// RemoveQueues closes the given TAP queue file descriptors and removes them
// from the tuntap.Fds list.
//
// This is a logical counterpart to AddQueues and allows releasing specific queues
// (e.g., to simulate queue failure or perform partial detach).
//
// The method updates tuntap.Queues to reflect the number of remaining active queues.
//
// It is safe to call with a subset of tuntap.Fds, but the caller must ensure
// that the passed *os.File descriptors belong to this interface.
func (tuntap *Tuntap) RemoveQueues(fds ...*os.File) error {
	toClose := make(map[uintptr]struct{}, len(fds))
	for _, fd := range fds {
		toClose[fd.Fd()] = struct{}{}
	}

	var newFds []*os.File
	for _, fd := range tuntap.Fds {
		if _, shouldClose := toClose[fd.Fd()]; shouldClose {
			if err := fd.Close(); err != nil {
				return fmt.Errorf("failed to close queue fd %d: %w", fd.Fd(), err)
			}
			tuntap.Queues--
		} else {
			newFds = append(newFds, fd)
		}
	}
	tuntap.Fds = newFds
	return nil
}
