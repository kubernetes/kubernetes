// +build windows

package winio

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"runtime"
	"syscall"
	"time"
	"unsafe"
)

//sys connectNamedPipe(pipe syscall.Handle, o *syscall.Overlapped) (err error) = ConnectNamedPipe
//sys createNamedPipe(name string, flags uint32, pipeMode uint32, maxInstances uint32, outSize uint32, inSize uint32, defaultTimeout uint32, sa *syscall.SecurityAttributes) (handle syscall.Handle, err error)  [failretval==syscall.InvalidHandle] = CreateNamedPipeW
//sys createFile(name string, access uint32, mode uint32, sa *syscall.SecurityAttributes, createmode uint32, attrs uint32, templatefile syscall.Handle) (handle syscall.Handle, err error) [failretval==syscall.InvalidHandle] = CreateFileW
//sys getNamedPipeInfo(pipe syscall.Handle, flags *uint32, outSize *uint32, inSize *uint32, maxInstances *uint32) (err error) = GetNamedPipeInfo
//sys getNamedPipeHandleState(pipe syscall.Handle, state *uint32, curInstances *uint32, maxCollectionCount *uint32, collectDataTimeout *uint32, userName *uint16, maxUserNameSize uint32) (err error) = GetNamedPipeHandleStateW
//sys localAlloc(uFlags uint32, length uint32) (ptr uintptr) = LocalAlloc
//sys ntCreateNamedPipeFile(pipe *syscall.Handle, access uint32, oa *objectAttributes, iosb *ioStatusBlock, share uint32, disposition uint32, options uint32, typ uint32, readMode uint32, completionMode uint32, maxInstances uint32, inboundQuota uint32, outputQuota uint32, timeout *int64) (status ntstatus) = ntdll.NtCreateNamedPipeFile
//sys rtlNtStatusToDosError(status ntstatus) (winerr error) = ntdll.RtlNtStatusToDosErrorNoTeb
//sys rtlDosPathNameToNtPathName(name *uint16, ntName *unicodeString, filePart uintptr, reserved uintptr) (status ntstatus) = ntdll.RtlDosPathNameToNtPathName_U
//sys rtlDefaultNpAcl(dacl *uintptr) (status ntstatus) = ntdll.RtlDefaultNpAcl

type ioStatusBlock struct {
	Status, Information uintptr
}

type objectAttributes struct {
	Length             uintptr
	RootDirectory      uintptr
	ObjectName         *unicodeString
	Attributes         uintptr
	SecurityDescriptor *securityDescriptor
	SecurityQoS        uintptr
}

type unicodeString struct {
	Length        uint16
	MaximumLength uint16
	Buffer        uintptr
}

type securityDescriptor struct {
	Revision byte
	Sbz1     byte
	Control  uint16
	Owner    uintptr
	Group    uintptr
	Sacl     uintptr
	Dacl     uintptr
}

type ntstatus int32

func (status ntstatus) Err() error {
	if status >= 0 {
		return nil
	}
	return rtlNtStatusToDosError(status)
}

const (
	cERROR_PIPE_BUSY      = syscall.Errno(231)
	cERROR_NO_DATA        = syscall.Errno(232)
	cERROR_PIPE_CONNECTED = syscall.Errno(535)
	cERROR_SEM_TIMEOUT    = syscall.Errno(121)

	cSECURITY_SQOS_PRESENT = 0x100000
	cSECURITY_ANONYMOUS    = 0

	cPIPE_TYPE_MESSAGE = 4

	cPIPE_READMODE_MESSAGE = 2

	cFILE_OPEN   = 1
	cFILE_CREATE = 2

	cFILE_PIPE_MESSAGE_TYPE          = 1
	cFILE_PIPE_REJECT_REMOTE_CLIENTS = 2

	cSE_DACL_PRESENT = 4
)

var (
	// ErrPipeListenerClosed is returned for pipe operations on listeners that have been closed.
	// This error should match net.errClosing since docker takes a dependency on its text.
	ErrPipeListenerClosed = errors.New("use of closed network connection")

	errPipeWriteClosed = errors.New("pipe has been closed for write")
)

type win32Pipe struct {
	*win32File
	path string
}

type win32MessageBytePipe struct {
	win32Pipe
	writeClosed bool
	readEOF     bool
}

type pipeAddress string

func (f *win32Pipe) LocalAddr() net.Addr {
	return pipeAddress(f.path)
}

func (f *win32Pipe) RemoteAddr() net.Addr {
	return pipeAddress(f.path)
}

func (f *win32Pipe) SetDeadline(t time.Time) error {
	f.SetReadDeadline(t)
	f.SetWriteDeadline(t)
	return nil
}

// CloseWrite closes the write side of a message pipe in byte mode.
func (f *win32MessageBytePipe) CloseWrite() error {
	if f.writeClosed {
		return errPipeWriteClosed
	}
	err := f.win32File.Flush()
	if err != nil {
		return err
	}
	_, err = f.win32File.Write(nil)
	if err != nil {
		return err
	}
	f.writeClosed = true
	return nil
}

// Write writes bytes to a message pipe in byte mode. Zero-byte writes are ignored, since
// they are used to implement CloseWrite().
func (f *win32MessageBytePipe) Write(b []byte) (int, error) {
	if f.writeClosed {
		return 0, errPipeWriteClosed
	}
	if len(b) == 0 {
		return 0, nil
	}
	return f.win32File.Write(b)
}

// Read reads bytes from a message pipe in byte mode. A read of a zero-byte message on a message
// mode pipe will return io.EOF, as will all subsequent reads.
func (f *win32MessageBytePipe) Read(b []byte) (int, error) {
	if f.readEOF {
		return 0, io.EOF
	}
	n, err := f.win32File.Read(b)
	if err == io.EOF {
		// If this was the result of a zero-byte read, then
		// it is possible that the read was due to a zero-size
		// message. Since we are simulating CloseWrite with a
		// zero-byte message, ensure that all future Read() calls
		// also return EOF.
		f.readEOF = true
	} else if err == syscall.ERROR_MORE_DATA {
		// ERROR_MORE_DATA indicates that the pipe's read mode is message mode
		// and the message still has more bytes. Treat this as a success, since
		// this package presents all named pipes as byte streams.
		err = nil
	}
	return n, err
}

func (s pipeAddress) Network() string {
	return "pipe"
}

func (s pipeAddress) String() string {
	return string(s)
}

// tryDialPipe attempts to dial the pipe at `path` until `ctx` cancellation or timeout.
func tryDialPipe(ctx context.Context, path *string) (syscall.Handle, error) {
	for {
		select {
		case <-ctx.Done():
			return syscall.Handle(0), ctx.Err()
		default:
			h, err := createFile(*path, syscall.GENERIC_READ|syscall.GENERIC_WRITE, 0, nil, syscall.OPEN_EXISTING, syscall.FILE_FLAG_OVERLAPPED|cSECURITY_SQOS_PRESENT|cSECURITY_ANONYMOUS, 0)
			if err == nil {
				return h, nil
			}
			if err != cERROR_PIPE_BUSY {
				return h, &os.PathError{Err: err, Op: "open", Path: *path}
			}
			// Wait 10 msec and try again. This is a rather simplistic
			// view, as we always try each 10 milliseconds.
			time.Sleep(time.Millisecond * 10)
		}
	}
}

// DialPipe connects to a named pipe by path, timing out if the connection
// takes longer than the specified duration. If timeout is nil, then we use
// a default timeout of 2 seconds.  (We do not use WaitNamedPipe.)
func DialPipe(path string, timeout *time.Duration) (net.Conn, error) {
	var absTimeout time.Time
	if timeout != nil {
		absTimeout = time.Now().Add(*timeout)
	} else {
		absTimeout = time.Now().Add(time.Second * 2)
	}
	ctx, _ := context.WithDeadline(context.Background(), absTimeout)
	conn, err := DialPipeContext(ctx, path)
	if err == context.DeadlineExceeded {
		return nil, ErrTimeout
	}
	return conn, err
}

// DialPipeContext attempts to connect to a named pipe by `path` until `ctx`
// cancellation or timeout.
func DialPipeContext(ctx context.Context, path string) (net.Conn, error) {
	var err error
	var h syscall.Handle
	h, err = tryDialPipe(ctx, &path)
	if err != nil {
		return nil, err
	}

	var flags uint32
	err = getNamedPipeInfo(h, &flags, nil, nil, nil)
	if err != nil {
		return nil, err
	}

	f, err := makeWin32File(h)
	if err != nil {
		syscall.Close(h)
		return nil, err
	}

	// If the pipe is in message mode, return a message byte pipe, which
	// supports CloseWrite().
	if flags&cPIPE_TYPE_MESSAGE != 0 {
		return &win32MessageBytePipe{
			win32Pipe: win32Pipe{win32File: f, path: path},
		}, nil
	}
	return &win32Pipe{win32File: f, path: path}, nil
}

type acceptResponse struct {
	f   *win32File
	err error
}

type win32PipeListener struct {
	firstHandle syscall.Handle
	path        string
	config      PipeConfig
	acceptCh    chan (chan acceptResponse)
	closeCh     chan int
	doneCh      chan int
}

func makeServerPipeHandle(path string, sd []byte, c *PipeConfig, first bool) (syscall.Handle, error) {
	path16, err := syscall.UTF16FromString(path)
	if err != nil {
		return 0, &os.PathError{Op: "open", Path: path, Err: err}
	}

	var oa objectAttributes
	oa.Length = unsafe.Sizeof(oa)

	var ntPath unicodeString
	if err := rtlDosPathNameToNtPathName(&path16[0], &ntPath, 0, 0).Err(); err != nil {
		return 0, &os.PathError{Op: "open", Path: path, Err: err}
	}
	defer localFree(ntPath.Buffer)
	oa.ObjectName = &ntPath

	// The security descriptor is only needed for the first pipe.
	if first {
		if sd != nil {
			len := uint32(len(sd))
			sdb := localAlloc(0, len)
			defer localFree(sdb)
			copy((*[0xffff]byte)(unsafe.Pointer(sdb))[:], sd)
			oa.SecurityDescriptor = (*securityDescriptor)(unsafe.Pointer(sdb))
		} else {
			// Construct the default named pipe security descriptor.
			var dacl uintptr
			if err := rtlDefaultNpAcl(&dacl).Err(); err != nil {
				return 0, fmt.Errorf("getting default named pipe ACL: %s", err)
			}
			defer localFree(dacl)

			sdb := &securityDescriptor{
				Revision: 1,
				Control:  cSE_DACL_PRESENT,
				Dacl:     dacl,
			}
			oa.SecurityDescriptor = sdb
		}
	}

	typ := uint32(cFILE_PIPE_REJECT_REMOTE_CLIENTS)
	if c.MessageMode {
		typ |= cFILE_PIPE_MESSAGE_TYPE
	}

	disposition := uint32(cFILE_OPEN)
	access := uint32(syscall.GENERIC_READ | syscall.GENERIC_WRITE | syscall.SYNCHRONIZE)
	if first {
		disposition = cFILE_CREATE
		// By not asking for read or write access, the named pipe file system
		// will put this pipe into an initially disconnected state, blocking
		// client connections until the next call with first == false.
		access = syscall.SYNCHRONIZE
	}

	timeout := int64(-50 * 10000) // 50ms

	var (
		h    syscall.Handle
		iosb ioStatusBlock
	)
	err = ntCreateNamedPipeFile(&h, access, &oa, &iosb, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE, disposition, 0, typ, 0, 0, 0xffffffff, uint32(c.InputBufferSize), uint32(c.OutputBufferSize), &timeout).Err()
	if err != nil {
		return 0, &os.PathError{Op: "open", Path: path, Err: err}
	}

	runtime.KeepAlive(ntPath)
	return h, nil
}

func (l *win32PipeListener) makeServerPipe() (*win32File, error) {
	h, err := makeServerPipeHandle(l.path, nil, &l.config, false)
	if err != nil {
		return nil, err
	}
	f, err := makeWin32File(h)
	if err != nil {
		syscall.Close(h)
		return nil, err
	}
	return f, nil
}

func (l *win32PipeListener) makeConnectedServerPipe() (*win32File, error) {
	p, err := l.makeServerPipe()
	if err != nil {
		return nil, err
	}

	// Wait for the client to connect.
	ch := make(chan error)
	go func(p *win32File) {
		ch <- connectPipe(p)
	}(p)

	select {
	case err = <-ch:
		if err != nil {
			p.Close()
			p = nil
		}
	case <-l.closeCh:
		// Abort the connect request by closing the handle.
		p.Close()
		p = nil
		err = <-ch
		if err == nil || err == ErrFileClosed {
			err = ErrPipeListenerClosed
		}
	}
	return p, err
}

func (l *win32PipeListener) listenerRoutine() {
	closed := false
	for !closed {
		select {
		case <-l.closeCh:
			closed = true
		case responseCh := <-l.acceptCh:
			var (
				p   *win32File
				err error
			)
			for {
				p, err = l.makeConnectedServerPipe()
				// If the connection was immediately closed by the client, try
				// again.
				if err != cERROR_NO_DATA {
					break
				}
			}
			responseCh <- acceptResponse{p, err}
			closed = err == ErrPipeListenerClosed
		}
	}
	syscall.Close(l.firstHandle)
	l.firstHandle = 0
	// Notify Close() and Accept() callers that the handle has been closed.
	close(l.doneCh)
}

// PipeConfig contain configuration for the pipe listener.
type PipeConfig struct {
	// SecurityDescriptor contains a Windows security descriptor in SDDL format.
	SecurityDescriptor string

	// MessageMode determines whether the pipe is in byte or message mode. In either
	// case the pipe is read in byte mode by default. The only practical difference in
	// this implementation is that CloseWrite() is only supported for message mode pipes;
	// CloseWrite() is implemented as a zero-byte write, but zero-byte writes are only
	// transferred to the reader (and returned as io.EOF in this implementation)
	// when the pipe is in message mode.
	MessageMode bool

	// InputBufferSize specifies the size the input buffer, in bytes.
	InputBufferSize int32

	// OutputBufferSize specifies the size the input buffer, in bytes.
	OutputBufferSize int32
}

// ListenPipe creates a listener on a Windows named pipe path, e.g. \\.\pipe\mypipe.
// The pipe must not already exist.
func ListenPipe(path string, c *PipeConfig) (net.Listener, error) {
	var (
		sd  []byte
		err error
	)
	if c == nil {
		c = &PipeConfig{}
	}
	if c.SecurityDescriptor != "" {
		sd, err = SddlToSecurityDescriptor(c.SecurityDescriptor)
		if err != nil {
			return nil, err
		}
	}
	h, err := makeServerPipeHandle(path, sd, c, true)
	if err != nil {
		return nil, err
	}
	l := &win32PipeListener{
		firstHandle: h,
		path:        path,
		config:      *c,
		acceptCh:    make(chan (chan acceptResponse)),
		closeCh:     make(chan int),
		doneCh:      make(chan int),
	}
	go l.listenerRoutine()
	return l, nil
}

func connectPipe(p *win32File) error {
	c, err := p.prepareIo()
	if err != nil {
		return err
	}
	defer p.wg.Done()

	err = connectNamedPipe(p.handle, &c.o)
	_, err = p.asyncIo(c, nil, 0, err)
	if err != nil && err != cERROR_PIPE_CONNECTED {
		return err
	}
	return nil
}

func (l *win32PipeListener) Accept() (net.Conn, error) {
	ch := make(chan acceptResponse)
	select {
	case l.acceptCh <- ch:
		response := <-ch
		err := response.err
		if err != nil {
			return nil, err
		}
		if l.config.MessageMode {
			return &win32MessageBytePipe{
				win32Pipe: win32Pipe{win32File: response.f, path: l.path},
			}, nil
		}
		return &win32Pipe{win32File: response.f, path: l.path}, nil
	case <-l.doneCh:
		return nil, ErrPipeListenerClosed
	}
}

func (l *win32PipeListener) Close() error {
	select {
	case l.closeCh <- 1:
		<-l.doneCh
	case <-l.doneCh:
	}
	return nil
}

func (l *win32PipeListener) Addr() net.Addr {
	return pipeAddress(l.path)
}
