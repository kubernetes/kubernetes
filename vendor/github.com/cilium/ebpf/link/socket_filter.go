package link

import (
	"syscall"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal/unix"
)

// AttachSocketFilter attaches a SocketFilter BPF program to a socket.
func AttachSocketFilter(conn syscall.Conn, program *ebpf.Program) error {
	rawConn, err := conn.SyscallConn()
	if err != nil {
		return err
	}
	var ssoErr error
	err = rawConn.Control(func(fd uintptr) {
		ssoErr = syscall.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_ATTACH_BPF, program.FD())
	})
	if ssoErr != nil {
		return ssoErr
	}
	return err
}

// DetachSocketFilter detaches a SocketFilter BPF program from a socket.
func DetachSocketFilter(conn syscall.Conn) error {
	rawConn, err := conn.SyscallConn()
	if err != nil {
		return err
	}
	var ssoErr error
	err = rawConn.Control(func(fd uintptr) {
		ssoErr = syscall.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_DETACH_BPF, 0)
	})
	if ssoErr != nil {
		return ssoErr
	}
	return err
}
