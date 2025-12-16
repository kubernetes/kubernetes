//go:build !js

package websocket

import (
	"compress/flate"
	"io"
	"sync"
)

// CompressionMode represents the modes available to the permessage-deflate extension.
// See https://tools.ietf.org/html/rfc7692
//
// Works in all modern browsers except Safari which does not implement the permessage-deflate extension.
//
// Compression is only used if the peer supports the mode selected.
type CompressionMode int

const (
	// CompressionDisabled disables the negotiation of the permessage-deflate extension.
	//
	// This is the default. Do not enable compression without benchmarking for your particular use case first.
	CompressionDisabled CompressionMode = iota

	// CompressionContextTakeover compresses each message greater than 128 bytes reusing the 32 KB sliding window from
	// previous messages. i.e compression context across messages is preserved.
	//
	// As most WebSocket protocols are text based and repetitive, this compression mode can be very efficient.
	//
	// The memory overhead is a fixed 32 KB sliding window, a fixed 1.2 MB flate.Writer and a sync.Pool of 40 KB flate.Reader's
	// that are used when reading and then returned.
	//
	// Thus, it uses more memory than CompressionNoContextTakeover but compresses more efficiently.
	//
	// If the peer does not support CompressionContextTakeover then we will fall back to CompressionNoContextTakeover.
	CompressionContextTakeover

	// CompressionNoContextTakeover compresses each message greater than 512 bytes. Each message is compressed with
	// a new 1.2 MB flate.Writer pulled from a sync.Pool. Each message is read with a 40 KB flate.Reader pulled from
	// a sync.Pool.
	//
	// This means less efficient compression as the sliding window from previous messages will not be used but the
	// memory overhead will be lower as there will be no fixed cost for the flate.Writer nor the 32 KB sliding window.
	// Especially if the connections are long lived and seldom written to.
	//
	// Thus, it uses less memory than CompressionContextTakeover but compresses less efficiently.
	//
	// If the peer does not support CompressionNoContextTakeover then we will fall back to CompressionDisabled.
	CompressionNoContextTakeover
)

func (m CompressionMode) opts() *compressionOptions {
	return &compressionOptions{
		clientNoContextTakeover: m == CompressionNoContextTakeover,
		serverNoContextTakeover: m == CompressionNoContextTakeover,
	}
}

type compressionOptions struct {
	clientNoContextTakeover bool
	serverNoContextTakeover bool
}

func (copts *compressionOptions) String() string {
	s := "permessage-deflate"
	if copts.clientNoContextTakeover {
		s += "; client_no_context_takeover"
	}
	if copts.serverNoContextTakeover {
		s += "; server_no_context_takeover"
	}
	return s
}

// These bytes are required to get flate.Reader to return.
// They are removed when sending to avoid the overhead as
// WebSocket framing tell's when the message has ended but then
// we need to add them back otherwise flate.Reader keeps
// trying to read more bytes.
const deflateMessageTail = "\x00\x00\xff\xff"

type trimLastFourBytesWriter struct {
	w    io.Writer
	tail []byte
}

func (tw *trimLastFourBytesWriter) reset() {
	if tw != nil && tw.tail != nil {
		tw.tail = tw.tail[:0]
	}
}

func (tw *trimLastFourBytesWriter) Write(p []byte) (int, error) {
	if tw.tail == nil {
		tw.tail = make([]byte, 0, 4)
	}

	extra := len(tw.tail) + len(p) - 4

	if extra <= 0 {
		tw.tail = append(tw.tail, p...)
		return len(p), nil
	}

	// Now we need to write as many extra bytes as we can from the previous tail.
	if extra > len(tw.tail) {
		extra = len(tw.tail)
	}
	if extra > 0 {
		_, err := tw.w.Write(tw.tail[:extra])
		if err != nil {
			return 0, err
		}

		// Shift remaining bytes in tail over.
		n := copy(tw.tail, tw.tail[extra:])
		tw.tail = tw.tail[:n]
	}

	// If p is less than or equal to 4 bytes,
	// all of it is is part of the tail.
	if len(p) <= 4 {
		tw.tail = append(tw.tail, p...)
		return len(p), nil
	}

	// Otherwise, only the last 4 bytes are.
	tw.tail = append(tw.tail, p[len(p)-4:]...)

	p = p[:len(p)-4]
	n, err := tw.w.Write(p)
	return n + 4, err
}

var flateReaderPool sync.Pool

func getFlateReader(r io.Reader, dict []byte) io.Reader {
	fr, ok := flateReaderPool.Get().(io.Reader)
	if !ok {
		return flate.NewReaderDict(r, dict)
	}
	fr.(flate.Resetter).Reset(r, dict)
	return fr
}

func putFlateReader(fr io.Reader) {
	flateReaderPool.Put(fr)
}

var flateWriterPool sync.Pool

func getFlateWriter(w io.Writer) *flate.Writer {
	fw, ok := flateWriterPool.Get().(*flate.Writer)
	if !ok {
		fw, _ = flate.NewWriter(w, flate.BestSpeed)
		return fw
	}
	fw.Reset(w)
	return fw
}

func putFlateWriter(w *flate.Writer) {
	flateWriterPool.Put(w)
}

type slidingWindow struct {
	buf []byte
}

var (
	swPoolMu sync.RWMutex
	swPool   = map[int]*sync.Pool{}
)

func slidingWindowPool(n int) *sync.Pool {
	swPoolMu.RLock()
	p, ok := swPool[n]
	swPoolMu.RUnlock()
	if ok {
		return p
	}

	p = &sync.Pool{}

	swPoolMu.Lock()
	swPool[n] = p
	swPoolMu.Unlock()

	return p
}

func (sw *slidingWindow) init(n int) {
	if sw.buf != nil {
		return
	}

	if n == 0 {
		n = 32768
	}

	p := slidingWindowPool(n)
	sw2, ok := p.Get().(*slidingWindow)
	if ok {
		*sw = *sw2
	} else {
		sw.buf = make([]byte, 0, n)
	}
}

func (sw *slidingWindow) close() {
	sw.buf = sw.buf[:0]
	swPoolMu.Lock()
	swPool[cap(sw.buf)].Put(sw)
	swPoolMu.Unlock()
}

func (sw *slidingWindow) write(p []byte) {
	if len(p) >= cap(sw.buf) {
		sw.buf = sw.buf[:cap(sw.buf)]
		p = p[len(p)-cap(sw.buf):]
		copy(sw.buf, p)
		return
	}

	left := cap(sw.buf) - len(sw.buf)
	if left < len(p) {
		// We need to shift spaceNeeded bytes from the end to make room for p at the end.
		spaceNeeded := len(p) - left
		copy(sw.buf, sw.buf[spaceNeeded:])
		sw.buf = sw.buf[:len(sw.buf)-spaceNeeded]
	}

	sw.buf = append(sw.buf, p...)
}
