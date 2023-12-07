//go:build windows

package socket

import (
	"errors"
	"fmt"
	"net"
	"sync"
	"syscall"
	"unsafe"

	"github.com/Microsoft/go-winio/pkg/guid"
	"golang.org/x/sys/windows"
)

//go:generate go run github.com/Microsoft/go-winio/tools/mkwinsyscall -output zsyscall_windows.go socket.go

//sys getsockname(s windows.Handle, name unsafe.Pointer, namelen *int32) (err error) [failretval==socketError] = ws2_32.getsockname
//sys getpeername(s windows.Handle, name unsafe.Pointer, namelen *int32) (err error) [failretval==socketError] = ws2_32.getpeername
//sys bind(s windows.Handle, name unsafe.Pointer, namelen int32) (err error) [failretval==socketError] = ws2_32.bind

const socketError = uintptr(^uint32(0))

var (
	// todo(helsaawy): create custom error types to store the desired vs actual size and addr family?

	ErrBufferSize     = errors.New("buffer size")
	ErrAddrFamily     = errors.New("address family")
	ErrInvalidPointer = errors.New("invalid pointer")
	ErrSocketClosed   = fmt.Errorf("socket closed: %w", net.ErrClosed)
)

// todo(helsaawy): replace these with generics, ie: GetSockName[S RawSockaddr](s windows.Handle) (S, error)

// GetSockName writes the local address of socket s to the [RawSockaddr] rsa.
// If rsa is not large enough, the [windows.WSAEFAULT] is returned.
func GetSockName(s windows.Handle, rsa RawSockaddr) error {
	ptr, l, err := rsa.Sockaddr()
	if err != nil {
		return fmt.Errorf("could not retrieve socket pointer and size: %w", err)
	}

	// although getsockname returns WSAEFAULT if the buffer is too small, it does not set
	// &l to the correct size, so--apart from doubling the buffer repeatedly--there is no remedy
	return getsockname(s, ptr, &l)
}

// GetPeerName returns the remote address the socket is connected to.
//
// See [GetSockName] for more information.
func GetPeerName(s windows.Handle, rsa RawSockaddr) error {
	ptr, l, err := rsa.Sockaddr()
	if err != nil {
		return fmt.Errorf("could not retrieve socket pointer and size: %w", err)
	}

	return getpeername(s, ptr, &l)
}

func Bind(s windows.Handle, rsa RawSockaddr) (err error) {
	ptr, l, err := rsa.Sockaddr()
	if err != nil {
		return fmt.Errorf("could not retrieve socket pointer and size: %w", err)
	}

	return bind(s, ptr, l)
}

// "golang.org/x/sys/windows".ConnectEx and .Bind only accept internal implementations of the
// their sockaddr interface, so they cannot be used with HvsockAddr
// Replicate functionality here from
// https://cs.opensource.google/go/x/sys/+/master:windows/syscall_windows.go

// The function pointers to `AcceptEx`, `ConnectEx` and `GetAcceptExSockaddrs` must be loaded at
// runtime via a WSAIoctl call:
// https://docs.microsoft.com/en-us/windows/win32/api/Mswsock/nc-mswsock-lpfn_connectex#remarks

type runtimeFunc struct {
	id   guid.GUID
	once sync.Once
	addr uintptr
	err  error
}

func (f *runtimeFunc) Load() error {
	f.once.Do(func() {
		var s windows.Handle
		s, f.err = windows.Socket(windows.AF_INET, windows.SOCK_STREAM, windows.IPPROTO_TCP)
		if f.err != nil {
			return
		}
		defer windows.CloseHandle(s) //nolint:errcheck

		var n uint32
		f.err = windows.WSAIoctl(s,
			windows.SIO_GET_EXTENSION_FUNCTION_POINTER,
			(*byte)(unsafe.Pointer(&f.id)),
			uint32(unsafe.Sizeof(f.id)),
			(*byte)(unsafe.Pointer(&f.addr)),
			uint32(unsafe.Sizeof(f.addr)),
			&n,
			nil, //overlapped
			0,   //completionRoutine
		)
	})
	return f.err
}

var (
	// todo: add `AcceptEx` and `GetAcceptExSockaddrs`
	WSAID_CONNECTEX = guid.GUID{ //revive:disable-line:var-naming ALL_CAPS
		Data1: 0x25a207b9,
		Data2: 0xddf3,
		Data3: 0x4660,
		Data4: [8]byte{0x8e, 0xe9, 0x76, 0xe5, 0x8c, 0x74, 0x06, 0x3e},
	}

	connectExFunc = runtimeFunc{id: WSAID_CONNECTEX}
)

func ConnectEx(
	fd windows.Handle,
	rsa RawSockaddr,
	sendBuf *byte,
	sendDataLen uint32,
	bytesSent *uint32,
	overlapped *windows.Overlapped,
) error {
	if err := connectExFunc.Load(); err != nil {
		return fmt.Errorf("failed to load ConnectEx function pointer: %w", err)
	}
	ptr, n, err := rsa.Sockaddr()
	if err != nil {
		return err
	}
	return connectEx(fd, ptr, n, sendBuf, sendDataLen, bytesSent, overlapped)
}

// BOOL LpfnConnectex(
//   [in]           SOCKET s,
//   [in]           const sockaddr *name,
//   [in]           int namelen,
//   [in, optional] PVOID lpSendBuffer,
//   [in]           DWORD dwSendDataLength,
//   [out]          LPDWORD lpdwBytesSent,
//   [in]           LPOVERLAPPED lpOverlapped
// )

func connectEx(
	s windows.Handle,
	name unsafe.Pointer,
	namelen int32,
	sendBuf *byte,
	sendDataLen uint32,
	bytesSent *uint32,
	overlapped *windows.Overlapped,
) (err error) {
	// todo: after upgrading to 1.18, switch from syscall.Syscall9 to syscall.SyscallN
	r1, _, e1 := syscall.Syscall9(connectExFunc.addr,
		7,
		uintptr(s),
		uintptr(name),
		uintptr(namelen),
		uintptr(unsafe.Pointer(sendBuf)),
		uintptr(sendDataLen),
		uintptr(unsafe.Pointer(bytesSent)),
		uintptr(unsafe.Pointer(overlapped)),
		0,
		0)
	if r1 == 0 {
		if e1 != 0 {
			err = error(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return err
}
