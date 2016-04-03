package ioutils

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"math/big"
	"sync"
	"time"
)

type readCloserWrapper struct {
	io.Reader
	closer func() error
}

func (r *readCloserWrapper) Close() error {
	return r.closer()
}

func NewReadCloserWrapper(r io.Reader, closer func() error) io.ReadCloser {
	return &readCloserWrapper{
		Reader: r,
		closer: closer,
	}
}

type readerErrWrapper struct {
	reader io.Reader
	closer func()
}

func (r *readerErrWrapper) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	if err != nil {
		r.closer()
	}
	return n, err
}

func NewReaderErrWrapper(r io.Reader, closer func()) io.Reader {
	return &readerErrWrapper{
		reader: r,
		closer: closer,
	}
}

// bufReader allows the underlying reader to continue to produce
// output by pre-emptively reading from the wrapped reader.
// This is achieved by buffering this data in bufReader's
// expanding buffer.
type bufReader struct {
	sync.Mutex
	buf                  *bytes.Buffer
	reader               io.Reader
	err                  error
	wait                 sync.Cond
	drainBuf             []byte
	reuseBuf             []byte
	maxReuse             int64
	resetTimeout         time.Duration
	bufLenResetThreshold int64
	maxReadDataReset     int64
}

func NewBufReader(r io.Reader) *bufReader {
	var timeout int
	if randVal, err := rand.Int(rand.Reader, big.NewInt(120)); err == nil {
		timeout = int(randVal.Int64()) + 180
	} else {
		timeout = 300
	}
	reader := &bufReader{
		buf:                  &bytes.Buffer{},
		drainBuf:             make([]byte, 1024),
		reuseBuf:             make([]byte, 4096),
		maxReuse:             1000,
		resetTimeout:         time.Second * time.Duration(timeout),
		bufLenResetThreshold: 100 * 1024,
		maxReadDataReset:     10 * 1024 * 1024,
		reader:               r,
	}
	reader.wait.L = &reader.Mutex
	go reader.drain()
	return reader
}

func NewBufReaderWithDrainbufAndBuffer(r io.Reader, drainBuffer []byte, buffer *bytes.Buffer) *bufReader {
	reader := &bufReader{
		buf:      buffer,
		drainBuf: drainBuffer,
		reader:   r,
	}
	reader.wait.L = &reader.Mutex
	go reader.drain()
	return reader
}

func (r *bufReader) drain() {
	var (
		duration       time.Duration
		lastReset      time.Time
		now            time.Time
		reset          bool
		bufLen         int64
		dataSinceReset int64
		maxBufLen      int64
		reuseBufLen    int64
		reuseCount     int64
	)
	reuseBufLen = int64(len(r.reuseBuf))
	lastReset = time.Now()
	for {
		n, err := r.reader.Read(r.drainBuf)
		dataSinceReset += int64(n)
		r.Lock()
		bufLen = int64(r.buf.Len())
		if bufLen > maxBufLen {
			maxBufLen = bufLen
		}

		// Avoid unbounded growth of the buffer over time.
		// This has been discovered to be the only non-intrusive
		// solution to the unbounded growth of the buffer.
		// Alternative solutions such as compression, multiple
		// buffers, channels and other similar pieces of code
		// were reducing throughput, overall Docker performance
		// or simply crashed Docker.
		// This solution releases the buffer when specific
		// conditions are met to avoid the continuous resizing
		// of the buffer for long lived containers.
		//
		// Move data to the front of the buffer if it's
		// smaller than what reuseBuf can store
		if bufLen > 0 && reuseBufLen >= bufLen {
			n, _ := r.buf.Read(r.reuseBuf)
			r.buf.Write(r.reuseBuf[0:n])
			// Take action if the buffer has been reused too many
			// times and if there's data in the buffer.
			// The timeout is also used as means to avoid doing
			// these operations more often or less often than
			// required.
			// The various conditions try to detect heavy activity
			// in the buffer which might be indicators of heavy
			// growth of the buffer.
		} else if reuseCount >= r.maxReuse && bufLen > 0 {
			now = time.Now()
			duration = now.Sub(lastReset)
			timeoutReached := duration >= r.resetTimeout

			// The timeout has been reached and the
			// buffered data couldn't be moved to the front
			// of the buffer, so the buffer gets reset.
			if timeoutReached && bufLen > reuseBufLen {
				reset = true
			}
			// The amount of buffered data is too high now,
			// reset the buffer.
			if timeoutReached && maxBufLen >= r.bufLenResetThreshold {
				reset = true
			}
			// Reset the buffer if a certain amount of
			// data has gone through the buffer since the
			// last reset.
			if timeoutReached && dataSinceReset >= r.maxReadDataReset {
				reset = true
			}
			// The buffered data is moved to a fresh buffer,
			// swap the old buffer with the new one and
			// reset all counters.
			if reset {
				newbuf := &bytes.Buffer{}
				newbuf.ReadFrom(r.buf)
				r.buf = newbuf
				lastReset = now
				reset = false
				dataSinceReset = 0
				maxBufLen = 0
				reuseCount = 0
			}
		}
		if err != nil {
			r.err = err
		} else {
			r.buf.Write(r.drainBuf[0:n])
		}
		reuseCount++
		r.wait.Signal()
		r.Unlock()
		callSchedulerIfNecessary()
		if err != nil {
			break
		}
	}
}

func (r *bufReader) Read(p []byte) (n int, err error) {
	r.Lock()
	defer r.Unlock()
	for {
		n, err = r.buf.Read(p)
		if n > 0 {
			return n, err
		}
		if r.err != nil {
			return 0, r.err
		}
		r.wait.Wait()
	}
}

func (r *bufReader) Close() error {
	closer, ok := r.reader.(io.ReadCloser)
	if !ok {
		return nil
	}
	return closer.Close()
}

func HashData(src io.Reader) (string, error) {
	h := sha256.New()
	if _, err := io.Copy(h, src); err != nil {
		return "", err
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

type OnEOFReader struct {
	Rc io.ReadCloser
	Fn func()
}

func (r *OnEOFReader) Read(p []byte) (n int, err error) {
	n, err = r.Rc.Read(p)
	if err == io.EOF {
		r.runFunc()
	}
	return
}

func (r *OnEOFReader) Close() error {
	err := r.Rc.Close()
	r.runFunc()
	return err
}

func (r *OnEOFReader) runFunc() {
	if fn := r.Fn; fn != nil {
		fn()
		r.Fn = nil
	}
}
