package winio

import (
	"errors"
	"io"
	"runtime"
	"sync"
	"syscall"
	"time"
)

//sys cancelIoEx(file syscall.Handle, o *syscall.Overlapped) (err error) = CancelIoEx
//sys createIoCompletionPort(file syscall.Handle, port syscall.Handle, key uintptr, threadCount uint32) (newport syscall.Handle, err error) = CreateIoCompletionPort
//sys getQueuedCompletionStatus(port syscall.Handle, bytes *uint32, key *uintptr, o **ioOperation, timeout uint32) (err error) = GetQueuedCompletionStatus
//sys setFileCompletionNotificationModes(h syscall.Handle, flags uint8) (err error) = SetFileCompletionNotificationModes
//sys timeBeginPeriod(period uint32) (n int32) = winmm.timeBeginPeriod

const (
	cFILE_SKIP_COMPLETION_PORT_ON_SUCCESS = 1
	cFILE_SKIP_SET_EVENT_ON_HANDLE        = 2
)

var (
	ErrFileClosed = errors.New("file has already been closed")
	ErrTimeout    = &timeoutError{}
)

type timeoutError struct{}

func (e *timeoutError) Error() string   { return "i/o timeout" }
func (e *timeoutError) Timeout() bool   { return true }
func (e *timeoutError) Temporary() bool { return true }

var ioInitOnce sync.Once
var ioCompletionPort syscall.Handle

// ioResult contains the result of an asynchronous IO operation
type ioResult struct {
	bytes uint32
	err   error
}

// ioOperation represents an outstanding asynchronous Win32 IO
type ioOperation struct {
	o  syscall.Overlapped
	ch chan ioResult
}

func initIo() {
	h, err := createIoCompletionPort(syscall.InvalidHandle, 0, 0, 0xffffffff)
	if err != nil {
		panic(err)
	}
	ioCompletionPort = h
	go ioCompletionProcessor(h)
}

// win32File implements Reader, Writer, and Closer on a Win32 handle without blocking in a syscall.
// It takes ownership of this handle and will close it if it is garbage collected.
type win32File struct {
	handle        syscall.Handle
	wg            sync.WaitGroup
	closing       bool
	readDeadline  time.Time
	writeDeadline time.Time
}

// makeWin32File makes a new win32File from an existing file handle
func makeWin32File(h syscall.Handle) (*win32File, error) {
	f := &win32File{handle: h}
	ioInitOnce.Do(initIo)
	_, err := createIoCompletionPort(h, ioCompletionPort, 0, 0xffffffff)
	if err != nil {
		return nil, err
	}
	err = setFileCompletionNotificationModes(h, cFILE_SKIP_COMPLETION_PORT_ON_SUCCESS|cFILE_SKIP_SET_EVENT_ON_HANDLE)
	if err != nil {
		return nil, err
	}
	runtime.SetFinalizer(f, (*win32File).closeHandle)
	return f, nil
}

func MakeOpenFile(h syscall.Handle) (io.ReadWriteCloser, error) {
	return makeWin32File(h)
}

// closeHandle closes the resources associated with a Win32 handle
func (f *win32File) closeHandle() {
	if !f.closing {
		// cancel all IO and wait for it to complete
		f.closing = true
		cancelIoEx(f.handle, nil)
		f.wg.Wait()
		// at this point, no new IO can start
		syscall.Close(f.handle)
		f.handle = 0
	}
}

// Close closes a win32File.
func (f *win32File) Close() error {
	f.closeHandle()
	runtime.SetFinalizer(f, nil)
	return nil
}

// prepareIo prepares for a new IO operation
func (f *win32File) prepareIo() (*ioOperation, error) {
	f.wg.Add(1)
	if f.closing {
		return nil, ErrFileClosed
	}
	c := &ioOperation{}
	c.ch = make(chan ioResult)
	return c, nil
}

// ioCompletionProcessor processes completed async IOs forever
func ioCompletionProcessor(h syscall.Handle) {
	// Set the timer resolution to 1. This fixes a performance regression in golang 1.6.
	timeBeginPeriod(1)
	for {
		var bytes uint32
		var key uintptr
		var op *ioOperation
		err := getQueuedCompletionStatus(h, &bytes, &key, &op, syscall.INFINITE)
		if op == nil {
			panic(err)
		}
		op.ch <- ioResult{bytes, err}
	}
}

// asyncIo processes the return value from ReadFile or WriteFile, blocking until
// the operation has actually completed.
func (f *win32File) asyncIo(c *ioOperation, deadline time.Time, bytes uint32, err error) (int, error) {
	if err != syscall.ERROR_IO_PENDING {
		f.wg.Done()
		return int(bytes), err
	} else {
		var r ioResult
		wait := true
		timedout := false
		if f.closing {
			cancelIoEx(f.handle, &c.o)
		} else if !deadline.IsZero() {
			now := time.Now()
			if !deadline.After(now) {
				timedout = true
			} else {
				timeout := time.After(deadline.Sub(now))
				select {
				case r = <-c.ch:
					wait = false
				case <-timeout:
					timedout = true
				}
			}
		}
		if timedout {
			cancelIoEx(f.handle, &c.o)
		}
		if wait {
			r = <-c.ch
		}
		err = r.err
		if err == syscall.ERROR_OPERATION_ABORTED {
			if f.closing {
				err = ErrFileClosed
			} else if timedout {
				err = ErrTimeout
			}
		}
		f.wg.Done()
		return int(r.bytes), err
	}
}

// Read reads from a file handle.
func (f *win32File) Read(b []byte) (int, error) {
	c, err := f.prepareIo()
	if err != nil {
		return 0, err
	}
	var bytes uint32
	err = syscall.ReadFile(f.handle, b, &bytes, &c.o)
	n, err := f.asyncIo(c, f.readDeadline, bytes, err)

	// Handle EOF conditions.
	if err == nil && n == 0 && len(b) != 0 {
		return 0, io.EOF
	} else if err == syscall.ERROR_BROKEN_PIPE {
		return 0, io.EOF
	} else {
		return n, err
	}
}

// Write writes to a file handle.
func (f *win32File) Write(b []byte) (int, error) {
	c, err := f.prepareIo()
	if err != nil {
		return 0, err
	}
	var bytes uint32
	err = syscall.WriteFile(f.handle, b, &bytes, &c.o)
	return f.asyncIo(c, f.writeDeadline, bytes, err)
}

func (f *win32File) SetReadDeadline(t time.Time) error {
	f.readDeadline = t
	return nil
}

func (f *win32File) SetWriteDeadline(t time.Time) error {
	f.writeDeadline = t
	return nil
}
