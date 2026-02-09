// The UnixCredentials system call is currently only implemented on Linux
// http://golang.org/src/pkg/syscall/sockcmsg_linux.go
// https://golang.org/s/go1.4-syscall
// http://code.google.com/p/go/source/browse/unix/sockcmsg_linux.go?repo=sys

// Local implementation of the UnixCredentials system call for FreeBSD

package dbus

import (
	"io"
	"os"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

// http://golang.org/src/pkg/syscall/ztypes_linux_amd64.go
// https://golang.org/src/syscall/ztypes_freebsd_amd64.go
//
// Note: FreeBSD actually uses a 'struct cmsgcred' which starts with
// these fields and adds a list of the additional groups for the
// sender.
type Ucred struct {
	Pid  int32
	Uid  uint32
	Euid uint32
	Gid  uint32
}

// https://github.com/freebsd/freebsd/blob/master/sys/sys/socket.h
//
// The cmsgcred structure contains the above four fields, followed by
// a uint16 count of additional groups, uint16 padding to align and a
// 16 element array of uint32 for the additional groups. The size is
// the same across all supported platforms.
const (
	SizeofCmsgcred = 84 // 4*4 + 2*2 + 16*4
)

// http://golang.org/src/pkg/syscall/sockcmsg_unix.go
func cmsgAlignOf(salen int) int {
	salign := unix.SizeofPtr

	return (salen + salign - 1) & ^(salign - 1)
}

// http://golang.org/src/pkg/syscall/sockcmsg_unix.go
func cmsgData(h *syscall.Cmsghdr) unsafe.Pointer {
	return unsafe.Pointer(uintptr(unsafe.Pointer(h)) + uintptr(cmsgAlignOf(syscall.SizeofCmsghdr)))
}

// http://golang.org/src/pkg/syscall/sockcmsg_linux.go
// UnixCredentials encodes credentials into a socket control message
// for sending to another process. This can be used for
// authentication.
func UnixCredentials(ucred *Ucred) []byte {
	b := make([]byte, syscall.CmsgSpace(SizeofCmsgcred))
	h := (*syscall.Cmsghdr)(unsafe.Pointer(&b[0]))
	h.Level = syscall.SOL_SOCKET
	h.Type = syscall.SCM_CREDS
	h.SetLen(syscall.CmsgLen(SizeofCmsgcred))
	*((*Ucred)(cmsgData(h))) = *ucred
	return b
}

// http://golang.org/src/pkg/syscall/sockcmsg_linux.go
// ParseUnixCredentials decodes a socket control message that contains
// credentials in a Ucred structure. To receive such a message, the
// SO_PASSCRED option must be enabled on the socket.
func ParseUnixCredentials(m *syscall.SocketControlMessage) (*Ucred, error) {
	if m.Header.Level != syscall.SOL_SOCKET {
		return nil, syscall.EINVAL
	}
	if m.Header.Type != syscall.SCM_CREDS {
		return nil, syscall.EINVAL
	}
	ucred := *(*Ucred)(unsafe.Pointer(&m.Data[0]))
	return &ucred, nil
}

func (t *unixTransport) SendNullByte() error {
	ucred := &Ucred{Pid: int32(os.Getpid()), Uid: uint32(os.Getuid()), Gid: uint32(os.Getgid())}
	b := UnixCredentials(ucred)
	_, oobn, err := t.UnixConn.WriteMsgUnix([]byte{0}, b, nil)
	if err != nil {
		return err
	}
	if oobn != len(b) {
		return io.ErrShortWrite
	}
	return nil
}
