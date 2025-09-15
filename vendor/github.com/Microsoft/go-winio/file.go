//go:build windows
// +build windows

package winio

import (
	"errors"
	"io"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/sys/windows"
)

//sys cancelIoEx(file windows.Handle, o *windows.Overlapped) (err error) = CancelIoEx
//sys createIoCompletionPort(file windows.Handle, port windows.Handle, key uintptr, threadCount uint32) (newport windows.Handle, err error) = CreateIoCompletionPort
//sys getQueuedCompletionStatus(port windows.Handle, bytes *uint32, key *uintptr, o **ioOperation, timeout uint32) (err error) = GetQueuedCompletionStatus
//sys setFileCompletionNotificationModes(h windows.Handle, flags uint8) (err error) = SetFileCompletionNotificationModes
//sys wsaGetOverlappedResult(h windows.Handle, o *windows.Overlapped, bytes *uint32, wait bool, flags *uint32) (err error) = ws2_32.WSAGetOverlappedResult

var (
	ErrFileClosed = errors.New("file has already been closed")
	ErrTimeout    = &timeoutError{}
)

type timeoutError struct{}

func (*timeoutError) Error() string   { return "i/o timeout" }
func (*timeoutError) Timeout() bool   { return true }
func (*timeoutError) Temporary() bool { return true }

type timeoutChan chan struct{}

var ioInitOnce sync.Once
var ioCompletionPort windows.Handle

// ioResult contains the result of an asynchronous IO operation.
type ioResult struct {
	bytes uint32
	err   error
}

// ioOperation represents an outstanding asynchronous Win32 IO.
type ioOperation struct {
	o  windows.Overlapped
	ch chan ioResult
}

func initIO() {
	h, err := createIoCompletionPort(windows.InvalidHandle, 0, 0, 0xffffffff)
	if err != nil {
		panic(err)
	}
	ioCompletionPort = h
	go ioCompletionProcessor(h)
}

// win32File implements Reader, Writer, and Closer on a Win32 handle without blocking in a syscall.
// It takes ownership of this handle and will close it if it is garbage collected.
type win32File struct {
	handle        windows.Handle
	wg            sync.WaitGroup
	wgLock        sync.RWMutex
	closing       atomic.Bool
	socket        bool
	readDeadline  deadlineHandler
	writeDeadline deadlineHandler
}

type deadlineHandler struct {
	setLock     sync.Mutex
	channel     timeoutChan
	channelLock sync.RWMutex
	timer       *time.Timer
	timedout    atomic.Bool
}

// makeWin32File makes a new win32File from an existing file handle.
func makeWin32File(h windows.Handle) (*win32File, error) {
	f := &win32File{handle: h}
	ioInitOnce.Do(initIO)
	_, err := createIoCompletionPort(h, ioCompletionPort, 0, 0xffffffff)
	if err != nil {
		return nil, err
	}
	err = setFileCompletionNotificationModes(h, windows.FILE_SKIP_COMPLETION_PORT_ON_SUCCESS|windows.FILE_SKIP_SET_EVENT_ON_HANDLE)
	if err != nil {
		return nil, err
	}
	f.readDeadline.channel = make(timeoutChan)
	f.writeDeadline.channel = make(timeoutChan)
	return f, nil
}

// Deprecated: use NewOpenFile instead.
func MakeOpenFile(h syscall.Handle) (io.ReadWriteCloser, error) {
	return NewOpenFile(windows.Handle(h))
}

func NewOpenFile(h windows.Handle) (io.ReadWriteCloser, error) {
	// If we return the result of makeWin32File directly, it can result in an
	// interface-wrapped nil, rather than a nil interface value.
	f, err := makeWin32File(h)
	if err != nil {
		return nil, err
	}
	return f, nil
}

// closeHandle closes the resources associated with a Win32 handle.
func (f *win32File) closeHandle() {
	f.wgLock.Lock()
	// Atomically set that we are closing, releasing the resources only once.
	if !f.closing.Swap(true) {
		f.wgLock.Unlock()
		// cancel all IO and wait for it to complete
		_ = cancelIoEx(f.handle, nil)
		f.wg.Wait()
		// at this point, no new IO can start
		windows.Close(f.handle)
		f.handle = 0
	} else {
		f.wgLock.Unlock()
	}
}

// Close closes a win32File.
func (f *win32File) Close() error {
	f.closeHandle()
	return nil
}

// IsClosed checks if the file has been closed.
func (f *win32File) IsClosed() bool {
	return f.closing.Load()
}

// prepareIO prepares for a new IO operation.
// The caller must call f.wg.Done() when the IO is finished, prior to Close() returning.
func (f *win32File) prepareIO() (*ioOperation, error) {
	f.wgLock.RLock()
	if f.closing.Load() {
		f.wgLock.RUnlock()
		return nil, ErrFileClosed
	}
	f.wg.Add(1)
	f.wgLock.RUnlock()
	c := &ioOperation{}
	c.ch = make(chan ioResult)
	return c, nil
}

// ioCompletionProcessor processes completed async IOs forever.
func ioCompletionProcessor(h windows.Handle) {
	for {
		var bytes uint32
		var key uintptr
		var op *ioOperation
		err := getQueuedCompletionStatus(h, &bytes, &key, &op, windows.INFINITE)
		if op == nil {
			panic(err)
		}
		op.ch <- ioResult{bytes, err}
	}
}

// todo: helsaawy - create an asyncIO version that takes a context

// asyncIO processes the return value from ReadFile or WriteFile, blocking until
// the operation has actually completed.
func (f *win32File) asyncIO(c *ioOperation, d *deadlineHandler, bytes uint32, err error) (int, error) {
	if err != windows.ERROR_IO_PENDING { //nolint:errorlint // err is Errno
		return int(bytes), err
	}

	if f.closing.Load() {
		_ = cancelIoEx(f.handle, &c.o)
	}

	var timeout timeoutChan
	if d != nil {
		d.channelLock.Lock()
		timeout = d.channel
		d.channelLock.Unlock()
	}

	var r ioResult
	select {
	case r = <-c.ch:
		err = r.err
		if err == windows.ERROR_OPERATION_ABORTED { //nolint:errorlint // err is Errno
			if f.closing.Load() {
				err = ErrFileClosed
			}
		} else if err != nil && f.socket {
			// err is from Win32. Query the overlapped structure to get the winsock error.
			var bytes, flags uint32
			err = wsaGetOverlappedResult(f.handle, &c.o, &bytes, false, &flags)
		}
	case <-timeout:
		_ = cancelIoEx(f.handle, &c.o)
		r = <-c.ch
		err = r.err
		if err == windows.ERROR_OPERATION_ABORTED { //nolint:errorlint // err is Errno
			err = ErrTimeout
		}
	}

	// runtime.KeepAlive is needed, as c is passed via native
	// code to ioCompletionProcessor, c must remain alive
	// until the channel read is complete.
	// todo: (de)allocate *ioOperation via win32 heap functions, instead of needing to KeepAlive?
	runtime.KeepAlive(c)
	return int(r.bytes), err
}

// Read reads from a file handle.
func (f *win32File) Read(b []byte) (int, error) {
	c, err := f.prepareIO()
	if err != nil {
		return 0, err
	}
	defer f.wg.Done()

	if f.readDeadline.timedout.Load() {
		return 0, ErrTimeout
	}

	var bytes uint32
	err = windows.ReadFile(f.handle, b, &bytes, &c.o)
	n, err := f.asyncIO(c, &f.readDeadline, bytes, err)
	runtime.KeepAlive(b)

	// Handle EOF conditions.
	if err == nil && n == 0 && len(b) != 0 {
		return 0, io.EOF
	} else if err == windows.ERROR_BROKEN_PIPE { //nolint:errorlint // err is Errno
		return 0, io.EOF
	}
	return n, err
}

// Write writes to a file handle.
func (f *win32File) Write(b []byte) (int, error) {
	c, err := f.prepareIO()
	if err != nil {
		return 0, err
	}
	defer f.wg.Done()

	if f.writeDeadline.timedout.Load() {
		return 0, ErrTimeout
	}

	var bytes uint32
	err = windows.WriteFile(f.handle, b, &bytes, &c.o)
	n, err := f.asyncIO(c, &f.writeDeadline, bytes, err)
	runtime.KeepAlive(b)
	return n, err
}

func (f *win32File) SetReadDeadline(deadline time.Time) error {
	return f.readDeadline.set(deadline)
}

func (f *win32File) SetWriteDeadline(deadline time.Time) error {
	return f.writeDeadline.set(deadline)
}

func (f *win32File) Flush() error {
	return windows.FlushFileBuffers(f.handle)
}

func (f *win32File) Fd() uintptr {
	return uintptr(f.handle)
}

func (d *deadlineHandler) set(deadline time.Time) error {
	d.setLock.Lock()
	defer d.setLock.Unlock()

	if d.timer != nil {
		if !d.timer.Stop() {
			<-d.channel
		}
		d.timer = nil
	}
	d.timedout.Store(false)

	select {
	case <-d.channel:
		d.channelLock.Lock()
		d.channel = make(chan struct{})
		d.channelLock.Unlock()
	default:
	}

	if deadline.IsZero() {
		return nil
	}

	timeoutIO := func() {
		d.timedout.Store(true)
		close(d.channel)
	}

	now := time.Now()
	duration := deadline.Sub(now)
	if deadline.After(now) {
		// Deadline is in the future, set a timer to wait
		d.timer = time.AfterFunc(duration, timeoutIO)
	} else {
		// Deadline is in the past. Cancel all pending IO now.
		timeoutIO()
	}
	return nil
}
