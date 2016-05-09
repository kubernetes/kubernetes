package ioutils

import (
	"errors"
	"io"
	"sync"
)

// maxCap is the highest capacity to use in byte slices that buffer data.
const maxCap = 1e6

// blockThreshold is the minimum number of bytes in the buffer which will cause
// a write to BytesPipe to block when allocating a new slice.
const blockThreshold = 1e6

// ErrClosed is returned when Write is called on a closed BytesPipe.
var ErrClosed = errors.New("write to closed BytesPipe")

// BytesPipe is io.ReadWriteCloser which works similarly to pipe(queue).
// All written data may be read at most once. Also, BytesPipe allocates
// and releases new byte slices to adjust to current needs, so the buffer
// won't be overgrown after peak loads.
type BytesPipe struct {
	mu       sync.Mutex
	wait     *sync.Cond
	buf      [][]byte // slice of byte-slices of buffered data
	lastRead int      // index in the first slice to a read point
	bufLen   int      // length of data buffered over the slices
	closeErr error    // error to return from next Read. set to nil if not closed.
}

// NewBytesPipe creates new BytesPipe, initialized by specified slice.
// If buf is nil, then it will be initialized with slice which cap is 64.
// buf will be adjusted in a way that len(buf) == 0, cap(buf) == cap(buf).
func NewBytesPipe(buf []byte) *BytesPipe {
	if cap(buf) == 0 {
		buf = make([]byte, 0, 64)
	}
	bp := &BytesPipe{
		buf: [][]byte{buf[:0]},
	}
	bp.wait = sync.NewCond(&bp.mu)
	return bp
}

// Write writes p to BytesPipe.
// It can allocate new []byte slices in a process of writing.
func (bp *BytesPipe) Write(p []byte) (int, error) {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	written := 0
	for {
		if bp.closeErr != nil {
			return written, ErrClosed
		}
		// write data to the last buffer
		b := bp.buf[len(bp.buf)-1]
		// copy data to the current empty allocated area
		n := copy(b[len(b):cap(b)], p)
		// increment buffered data length
		bp.bufLen += n
		// include written data in last buffer
		bp.buf[len(bp.buf)-1] = b[:len(b)+n]

		written += n

		// if there was enough room to write all then break
		if len(p) == n {
			break
		}

		// more data: write to the next slice
		p = p[n:]

		// block if too much data is still in the buffer
		for bp.bufLen >= blockThreshold {
			bp.wait.Wait()
		}

		// allocate slice that has twice the size of the last unless maximum reached
		nextCap := 2 * cap(bp.buf[len(bp.buf)-1])
		if nextCap > maxCap {
			nextCap = maxCap
		}
		// add new byte slice to the buffers slice and continue writing
		bp.buf = append(bp.buf, make([]byte, 0, nextCap))
	}
	bp.wait.Broadcast()
	return written, nil
}

// CloseWithError causes further reads from a BytesPipe to return immediately.
func (bp *BytesPipe) CloseWithError(err error) error {
	bp.mu.Lock()
	if err != nil {
		bp.closeErr = err
	} else {
		bp.closeErr = io.EOF
	}
	bp.wait.Broadcast()
	bp.mu.Unlock()
	return nil
}

// Close causes further reads from a BytesPipe to return immediately.
func (bp *BytesPipe) Close() error {
	return bp.CloseWithError(nil)
}

func (bp *BytesPipe) len() int {
	return bp.bufLen - bp.lastRead
}

// Read reads bytes from BytesPipe.
// Data could be read only once.
func (bp *BytesPipe) Read(p []byte) (n int, err error) {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	if bp.len() == 0 {
		if bp.closeErr != nil {
			return 0, bp.closeErr
		}
		bp.wait.Wait()
		if bp.len() == 0 && bp.closeErr != nil {
			return 0, bp.closeErr
		}
	}
	for {
		read := copy(p, bp.buf[0][bp.lastRead:])
		n += read
		bp.lastRead += read
		if bp.len() == 0 {
			// we have read everything. reset to the beginning.
			bp.lastRead = 0
			bp.bufLen -= len(bp.buf[0])
			bp.buf[0] = bp.buf[0][:0]
			break
		}
		// break if everything was read
		if len(p) == read {
			break
		}
		// more buffered data and more asked. read from next slice.
		p = p[read:]
		bp.lastRead = 0
		bp.bufLen -= len(bp.buf[0])
		bp.buf[0] = nil     // throw away old slice
		bp.buf = bp.buf[1:] // switch to next
	}
	bp.wait.Broadcast()
	return
}
