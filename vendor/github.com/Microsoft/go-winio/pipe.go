// +build windows

package winio

import (
	"errors"
	"io"
	"net"
	"os"
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

const (
	cERROR_PIPE_BUSY      = syscall.Errno(231)
	cERROR_NO_DATA        = syscall.Errno(232)
	cERROR_PIPE_CONNECTED = syscall.Errno(535)
	cERROR_SEM_TIMEOUT    = syscall.Errno(121)

	cPIPE_ACCESS_DUPLEX            = 0x3
	cFILE_FLAG_FIRST_PIPE_INSTANCE = 0x80000
	cSECURITY_SQOS_PRESENT         = 0x100000
	cSECURITY_ANONYMOUS            = 0

	cPIPE_REJECT_REMOTE_CLIENTS = 0x8

	cPIPE_UNLIMITED_INSTANCES = 255

	cNMPWAIT_USE_DEFAULT_WAIT = 0
	cNMPWAIT_NOWAIT           = 1

	cPIPE_TYPE_MESSAGE = 4

	cPIPE_READMODE_MESSAGE = 2
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

// DialPipe connects to a named pipe by path, timing out if the connection
// takes longer than the specified duration. If timeout is nil, then we use
// a default timeout of 5 seconds.  (We do not use WaitNamedPipe.)
func DialPipe(path string, timeout *time.Duration) (net.Conn, error) {
	var absTimeout time.Time
	if timeout != nil {
		absTimeout = time.Now().Add(*timeout)
	} else {
		absTimeout = time.Now().Add(time.Second * 2)
	}
	var err error
	var h syscall.Handle
	for {
		h, err = createFile(path, syscall.GENERIC_READ|syscall.GENERIC_WRITE, 0, nil, syscall.OPEN_EXISTING, syscall.FILE_FLAG_OVERLAPPED|cSECURITY_SQOS_PRESENT|cSECURITY_ANONYMOUS, 0)
		if err != cERROR_PIPE_BUSY {
			break
		}
		if time.Now().After(absTimeout) {
			return nil, ErrTimeout
		}

		// Wait 10 msec and try again. This is a rather simplistic
		// view, as we always try each 10 milliseconds.
		time.Sleep(time.Millisecond * 10)
	}
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: path, Err: err}
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
	firstHandle        syscall.Handle
	path               string
	securityDescriptor []byte
	config             PipeConfig
	acceptCh           chan (chan acceptResponse)
	closeCh            chan int
	doneCh             chan int
}

func makeServerPipeHandle(path string, securityDescriptor []byte, c *PipeConfig, first bool) (syscall.Handle, error) {
	var flags uint32 = cPIPE_ACCESS_DUPLEX | syscall.FILE_FLAG_OVERLAPPED
	if first {
		flags |= cFILE_FLAG_FIRST_PIPE_INSTANCE
	}

	var mode uint32 = cPIPE_REJECT_REMOTE_CLIENTS
	if c.MessageMode {
		mode |= cPIPE_TYPE_MESSAGE
	}

	sa := &syscall.SecurityAttributes{}
	sa.Length = uint32(unsafe.Sizeof(*sa))
	if securityDescriptor != nil {
		len := uint32(len(securityDescriptor))
		sa.SecurityDescriptor = localAlloc(0, len)
		defer localFree(sa.SecurityDescriptor)
		copy((*[0xffff]byte)(unsafe.Pointer(sa.SecurityDescriptor))[:], securityDescriptor)
	}
	h, err := createNamedPipe(path, flags, mode, cPIPE_UNLIMITED_INSTANCES, uint32(c.OutputBufferSize), uint32(c.InputBufferSize), 0, sa)
	if err != nil {
		return 0, &os.PathError{Op: "open", Path: path, Err: err}
	}
	return h, nil
}

func (l *win32PipeListener) makeServerPipe() (*win32File, error) {
	h, err := makeServerPipeHandle(l.path, l.securityDescriptor, &l.config, false)
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
	// Create a client handle and connect it.  This results in the pipe
	// instance always existing, so that clients see ERROR_PIPE_BUSY
	// rather than ERROR_FILE_NOT_FOUND.  This ties the first instance
	// up so that no other instances can be used.  This would have been
	// cleaner if the Win32 API matched CreateFile with ConnectNamedPipe
	// instead of CreateNamedPipe.  (Apparently created named pipes are
	// considered to be in listening state regardless of whether any
	// active calls to ConnectNamedPipe are outstanding.)
	h2, err := createFile(path, 0, 0, nil, syscall.OPEN_EXISTING, cSECURITY_SQOS_PRESENT|cSECURITY_ANONYMOUS, 0)
	if err != nil {
		syscall.Close(h)
		return nil, err
	}
	// Close the client handle. The server side of the instance will
	// still be busy, leading to ERROR_PIPE_BUSY instead of
	// ERROR_NOT_FOUND, as long as we don't close the server handle,
	// or disconnect the client with DisconnectNamedPipe.
	syscall.Close(h2)
	l := &win32PipeListener{
		firstHandle:        h,
		path:               path,
		securityDescriptor: sd,
		config:             *c,
		acceptCh:           make(chan (chan acceptResponse)),
		closeCh:            make(chan int),
		doneCh:             make(chan int),
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
