package ioutils

import (
	"errors"
	"io"
	"sync"
)

// maxCap is the highest capacity to use in byte slices that buffer data.
const maxCap = 1e6

// minCap is the lowest capacity to use in byte slices that buffer data
const minCap = 64

// blockThreshold is the minimum number of bytes in the buffer which will cause
// a write to BytesPipe to block when allocating a new slice.
const blockThreshold = 1e6

var (
	// ErrClosed is returned when Write is called on a closed BytesPipe.
	ErrClosed = errors.New("write to closed BytesPipe")

	bufPools     = make(map[int]*sync.Pool)
	bufPoolsLock sync.Mutex
)

// BytesPipe is io.ReadWriteCloser which works similarly to pipe(queue).
// All written data may be read at most once. Also, BytesPipe allocates
// and releases new byte slices to adjust to current needs, so the buffer
// won't be overgrown after peak loads.
type BytesPipe struct {
	mu       sync.Mutex
	wait     *sync.Cond
	buf      []*fixedBuffer
	bufLen   int
	closeErr error // error to return from next Read. set to nil if not closed.
}

// NewBytesPipe creates new BytesPipe, initialized by specified slice.
// If buf is nil, then it will be initialized with slice which cap is 64.
// buf will be adjusted in a way that len(buf) == 0, cap(buf) == cap(buf).
func NewBytesPipe() *BytesPipe {
	bp := &BytesPipe{}
	bp.buf = append(bp.buf, getBuffer(minCap))
	bp.wait = sync.NewCond(&bp.mu)
	return bp
}

// Write writes p to BytesPipe.
// It can allocate new []byte slices in a process of writing.
func (bp *BytesPipe) Write(p []byte) (int, error) {
	bp.mu.Lock()

	written := 0
loop0:
	for {
		if bp.closeErr != nil {
			bp.mu.Unlock()
			return written, ErrClosed
		}

		if len(bp.buf) == 0 {
			bp.buf = append(bp.buf, getBuffer(64))
		}
		// get the last buffer
		b := bp.buf[len(bp.buf)-1]

		n, err := b.Write(p)
		written += n
		bp.bufLen += n

		// errBufferFull is an error we expect to get if the buffer is full
		if err != nil && err != errBufferFull {
			bp.wait.Broadcast()
			bp.mu.Unlock()
			return written, err
		}

		// if there was enough room to write all then break
		if len(p) == n {
			break
		}

		// more data: write to the next slice
		p = p[n:]

		// make sure the buffer doesn't grow too big from this write
		for bp.bufLen >= blockThreshold {
			bp.wait.Wait()
			if bp.closeErr != nil {
				continue loop0
			}
		}

		// add new byte slice to the buffers slice and continue writing
		nextCap := b.Cap() * 2
		if nextCap > maxCap {
			nextCap = maxCap
		}
		bp.buf = append(bp.buf, getBuffer(nextCap))
	}
	bp.wait.Broadcast()
	bp.mu.Unlock()
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

// Read reads bytes from BytesPipe.
// Data could be read only once.
func (bp *BytesPipe) Read(p []byte) (n int, err error) {
	bp.mu.Lock()
	if bp.bufLen == 0 {
		if bp.closeErr != nil {
			bp.mu.Unlock()
			return 0, bp.closeErr
		}
		bp.wait.Wait()
		if bp.bufLen == 0 && bp.closeErr != nil {
			err := bp.closeErr
			bp.mu.Unlock()
			return 0, err
		}
	}

	for bp.bufLen > 0 {
		b := bp.buf[0]
		read, _ := b.Read(p) // ignore error since fixedBuffer doesn't really return an error
		n += read
		bp.bufLen -= read

		if b.Len() == 0 {
			// it's empty so return it to the pool and move to the next one
			returnBuffer(b)
			bp.buf[0] = nil
			bp.buf = bp.buf[1:]
		}

		if len(p) == read {
			break
		}

		p = p[read:]
	}

	bp.wait.Broadcast()
	bp.mu.Unlock()
	return
}

func returnBuffer(b *fixedBuffer) {
	b.Reset()
	bufPoolsLock.Lock()
	pool := bufPools[b.Cap()]
	bufPoolsLock.Unlock()
	if pool != nil {
		pool.Put(b)
	}
}

func getBuffer(size int) *fixedBuffer {
	bufPoolsLock.Lock()
	pool, ok := bufPools[size]
	if !ok {
		pool = &sync.Pool{New: func() interface{} { return &fixedBuffer{buf: make([]byte, 0, size)} }}
		bufPools[size] = pool
	}
	bufPoolsLock.Unlock()
	return pool.Get().(*fixedBuffer)
}
