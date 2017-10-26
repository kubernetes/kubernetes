package jsoniter

import (
	"io"
)

// Stream is a io.Writer like object, with JSON specific write functions.
// Error is not returned as return value, but stored as Error member on this stream instance.
type Stream struct {
	cfg       *frozenConfig
	out       io.Writer
	buf       []byte
	n         int
	Error     error
	indention int
}

// NewStream create new stream instance.
// cfg can be jsoniter.ConfigDefault.
// out can be nil if write to internal buffer.
// bufSize is the initial size for the internal buffer in bytes.
func NewStream(cfg API, out io.Writer, bufSize int) *Stream {
	return &Stream{
		cfg:       cfg.(*frozenConfig),
		out:       out,
		buf:       make([]byte, bufSize),
		n:         0,
		Error:     nil,
		indention: 0,
	}
}

// Pool returns a pool can provide more stream with same configuration
func (stream *Stream) Pool() StreamPool {
	return stream.cfg
}

// Reset reuse this stream instance by assign a new writer
func (stream *Stream) Reset(out io.Writer) {
	stream.out = out
	stream.n = 0
}

// Available returns how many bytes are unused in the buffer.
func (stream *Stream) Available() int {
	return len(stream.buf) - stream.n
}

// Buffered returns the number of bytes that have been written into the current buffer.
func (stream *Stream) Buffered() int {
	return stream.n
}

// Buffer if writer is nil, use this method to take the result
func (stream *Stream) Buffer() []byte {
	return stream.buf[:stream.n]
}

// Write writes the contents of p into the buffer.
// It returns the number of bytes written.
// If nn < len(p), it also returns an error explaining
// why the write is short.
func (stream *Stream) Write(p []byte) (nn int, err error) {
	for len(p) > stream.Available() && stream.Error == nil {
		if stream.out == nil {
			stream.growAtLeast(len(p))
		} else {
			var n int
			if stream.Buffered() == 0 {
				// Large write, empty buffer.
				// Write directly from p to avoid copy.
				n, stream.Error = stream.out.Write(p)
			} else {
				n = copy(stream.buf[stream.n:], p)
				stream.n += n
				stream.Flush()
			}
			nn += n
			p = p[n:]
		}
	}
	if stream.Error != nil {
		return nn, stream.Error
	}
	n := copy(stream.buf[stream.n:], p)
	stream.n += n
	nn += n
	return nn, nil
}

// WriteByte writes a single byte.
func (stream *Stream) writeByte(c byte) {
	if stream.Error != nil {
		return
	}
	if stream.Available() < 1 {
		stream.growAtLeast(1)
	}
	stream.buf[stream.n] = c
	stream.n++
}

func (stream *Stream) writeTwoBytes(c1 byte, c2 byte) {
	if stream.Error != nil {
		return
	}
	if stream.Available() < 2 {
		stream.growAtLeast(2)
	}
	stream.buf[stream.n] = c1
	stream.buf[stream.n+1] = c2
	stream.n += 2
}

func (stream *Stream) writeThreeBytes(c1 byte, c2 byte, c3 byte) {
	if stream.Error != nil {
		return
	}
	if stream.Available() < 3 {
		stream.growAtLeast(3)
	}
	stream.buf[stream.n] = c1
	stream.buf[stream.n+1] = c2
	stream.buf[stream.n+2] = c3
	stream.n += 3
}

func (stream *Stream) writeFourBytes(c1 byte, c2 byte, c3 byte, c4 byte) {
	if stream.Error != nil {
		return
	}
	if stream.Available() < 4 {
		stream.growAtLeast(4)
	}
	stream.buf[stream.n] = c1
	stream.buf[stream.n+1] = c2
	stream.buf[stream.n+2] = c3
	stream.buf[stream.n+3] = c4
	stream.n += 4
}

func (stream *Stream) writeFiveBytes(c1 byte, c2 byte, c3 byte, c4 byte, c5 byte) {
	if stream.Error != nil {
		return
	}
	if stream.Available() < 5 {
		stream.growAtLeast(5)
	}
	stream.buf[stream.n] = c1
	stream.buf[stream.n+1] = c2
	stream.buf[stream.n+2] = c3
	stream.buf[stream.n+3] = c4
	stream.buf[stream.n+4] = c5
	stream.n += 5
}

// Flush writes any buffered data to the underlying io.Writer.
func (stream *Stream) Flush() error {
	if stream.out == nil {
		return nil
	}
	if stream.Error != nil {
		return stream.Error
	}
	if stream.n == 0 {
		return nil
	}
	n, err := stream.out.Write(stream.buf[0:stream.n])
	if n < stream.n && err == nil {
		err = io.ErrShortWrite
	}
	if err != nil {
		if n > 0 && n < stream.n {
			copy(stream.buf[0:stream.n-n], stream.buf[n:stream.n])
		}
		stream.n -= n
		stream.Error = err
		return err
	}
	stream.n = 0
	return nil
}

func (stream *Stream) ensure(minimal int) {
	available := stream.Available()
	if available < minimal {
		stream.growAtLeast(minimal)
	}
}

func (stream *Stream) growAtLeast(minimal int) {
	if stream.out != nil {
		stream.Flush()
		if stream.Available() >= minimal {
			return
		}
	}
	toGrow := len(stream.buf)
	if toGrow < minimal {
		toGrow = minimal
	}
	newBuf := make([]byte, len(stream.buf)+toGrow)
	copy(newBuf, stream.Buffer())
	stream.buf = newBuf
}

// WriteRaw write string out without quotes, just like []byte
func (stream *Stream) WriteRaw(s string) {
	stream.ensure(len(s))
	if stream.Error != nil {
		return
	}
	n := copy(stream.buf[stream.n:], s)
	stream.n += n
}

// WriteNil write null to stream
func (stream *Stream) WriteNil() {
	stream.writeFourBytes('n', 'u', 'l', 'l')
}

// WriteTrue write true to stream
func (stream *Stream) WriteTrue() {
	stream.writeFourBytes('t', 'r', 'u', 'e')
}

// WriteFalse write false to stream
func (stream *Stream) WriteFalse() {
	stream.writeFiveBytes('f', 'a', 'l', 's', 'e')
}

// WriteBool write true or false into stream
func (stream *Stream) WriteBool(val bool) {
	if val {
		stream.WriteTrue()
	} else {
		stream.WriteFalse()
	}
}

// WriteObjectStart write { with possible indention
func (stream *Stream) WriteObjectStart() {
	stream.indention += stream.cfg.indentionStep
	stream.writeByte('{')
	stream.writeIndention(0)
}

// WriteObjectField write "field": with possible indention
func (stream *Stream) WriteObjectField(field string) {
	stream.WriteString(field)
	if stream.indention > 0 {
		stream.writeTwoBytes(':', ' ')
	} else {
		stream.writeByte(':')
	}
}

// WriteObjectEnd write } with possible indention
func (stream *Stream) WriteObjectEnd() {
	stream.writeIndention(stream.cfg.indentionStep)
	stream.indention -= stream.cfg.indentionStep
	stream.writeByte('}')
}

// WriteEmptyObject write {}
func (stream *Stream) WriteEmptyObject() {
	stream.writeByte('{')
	stream.writeByte('}')
}

// WriteMore write , with possible indention
func (stream *Stream) WriteMore() {
	stream.writeByte(',')
	stream.writeIndention(0)
}

// WriteArrayStart write [ with possible indention
func (stream *Stream) WriteArrayStart() {
	stream.indention += stream.cfg.indentionStep
	stream.writeByte('[')
	stream.writeIndention(0)
}

// WriteEmptyArray write []
func (stream *Stream) WriteEmptyArray() {
	stream.writeTwoBytes('[', ']')
}

// WriteArrayEnd write ] with possible indention
func (stream *Stream) WriteArrayEnd() {
	stream.writeIndention(stream.cfg.indentionStep)
	stream.indention -= stream.cfg.indentionStep
	stream.writeByte(']')
}

func (stream *Stream) writeIndention(delta int) {
	if stream.indention == 0 {
		return
	}
	stream.writeByte('\n')
	toWrite := stream.indention - delta
	stream.ensure(toWrite)
	for i := 0; i < toWrite && stream.n < len(stream.buf); i++ {
		stream.buf[stream.n] = ' '
		stream.n++
	}
}
