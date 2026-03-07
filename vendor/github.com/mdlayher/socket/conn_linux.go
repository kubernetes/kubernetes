//go:build linux
// +build linux

package socket

import (
	"context"
	"os"
	"unsafe"

	"golang.org/x/net/bpf"
	"golang.org/x/sys/unix"
)

// IoctlKCMClone wraps ioctl(2) for unix.KCMClone values, but returns a Conn
// rather than a raw file descriptor.
func (c *Conn) IoctlKCMClone() (*Conn, error) {
	info, err := controlT(c, "ioctl", unix.IoctlKCMClone)
	if err != nil {
		return nil, err
	}

	// Successful clone, wrap in a Conn for use by the caller.
	return New(int(info.Fd), c.name)
}

// IoctlKCMAttach wraps ioctl(2) for unix.KCMAttach values.
func (c *Conn) IoctlKCMAttach(info unix.KCMAttach) error {
	return c.control("ioctl", func(fd int) error {
		return unix.IoctlKCMAttach(fd, info)
	})
}

// IoctlKCMUnattach wraps ioctl(2) for unix.KCMUnattach values.
func (c *Conn) IoctlKCMUnattach(info unix.KCMUnattach) error {
	return c.control("ioctl", func(fd int) error {
		return unix.IoctlKCMUnattach(fd, info)
	})
}

// PidfdGetfd wraps pidfd_getfd(2) for a Conn which wraps a pidfd, but returns a
// Conn rather than a raw file descriptor.
func (c *Conn) PidfdGetfd(targetFD, flags int) (*Conn, error) {
	outFD, err := controlT(c, "pidfd_getfd", func(fd int) (int, error) {
		return unix.PidfdGetfd(fd, targetFD, flags)
	})
	if err != nil {
		return nil, err
	}

	// Successful getfd, wrap in a Conn for use by the caller.
	return New(outFD, c.name)
}

// PidfdSendSignal wraps pidfd_send_signal(2) for a Conn which wraps a Linux
// pidfd.
func (c *Conn) PidfdSendSignal(sig unix.Signal, info *unix.Siginfo, flags int) error {
	return c.control("pidfd_send_signal", func(fd int) error {
		return unix.PidfdSendSignal(fd, sig, info, flags)
	})
}

// SetBPF attaches an assembled BPF program to a Conn.
func (c *Conn) SetBPF(filter []bpf.RawInstruction) error {
	// We can't point to the first instruction in the array if no instructions
	// are present.
	if len(filter) == 0 {
		return os.NewSyscallError("setsockopt", unix.EINVAL)
	}

	prog := unix.SockFprog{
		Len:    uint16(len(filter)),
		Filter: (*unix.SockFilter)(unsafe.Pointer(&filter[0])),
	}

	return c.SetsockoptSockFprog(unix.SOL_SOCKET, unix.SO_ATTACH_FILTER, &prog)
}

// RemoveBPF removes a BPF filter from a Conn.
func (c *Conn) RemoveBPF() error {
	// 0 argument is ignored.
	return c.SetsockoptInt(unix.SOL_SOCKET, unix.SO_DETACH_FILTER, 0)
}

// SetsockoptPacketMreq wraps setsockopt(2) for unix.PacketMreq values.
func (c *Conn) SetsockoptPacketMreq(level, opt int, mreq *unix.PacketMreq) error {
	return c.control("setsockopt", func(fd int) error {
		return unix.SetsockoptPacketMreq(fd, level, opt, mreq)
	})
}

// SetsockoptSockFprog wraps setsockopt(2) for unix.SockFprog values.
func (c *Conn) SetsockoptSockFprog(level, opt int, fprog *unix.SockFprog) error {
	return c.control("setsockopt", func(fd int) error {
		return unix.SetsockoptSockFprog(fd, level, opt, fprog)
	})
}

// GetsockoptTpacketStats wraps getsockopt(2) for unix.TpacketStats values.
func (c *Conn) GetsockoptTpacketStats(level, name int) (*unix.TpacketStats, error) {
	return controlT(c, "getsockopt", func(fd int) (*unix.TpacketStats, error) {
		return unix.GetsockoptTpacketStats(fd, level, name)
	})
}

// GetsockoptTpacketStatsV3 wraps getsockopt(2) for unix.TpacketStatsV3 values.
func (c *Conn) GetsockoptTpacketStatsV3(level, name int) (*unix.TpacketStatsV3, error) {
	return controlT(c, "getsockopt", func(fd int) (*unix.TpacketStatsV3, error) {
		return unix.GetsockoptTpacketStatsV3(fd, level, name)
	})
}

// Waitid wraps waitid(2).
func (c *Conn) Waitid(idType int, info *unix.Siginfo, options int, rusage *unix.Rusage) error {
	return c.read(context.Background(), "waitid", func(fd int) error {
		return unix.Waitid(idType, fd, info, options, rusage)
	})
}
