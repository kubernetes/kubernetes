package logdriver

import (
	"encoding/binary"
	"io"
)

const binaryEncodeLen = 4

// LogEntryEncoder encodes a LogEntry to a protobuf stream
// The stream should look like:
//
// [uint32 binary encoded message size][protobuf message]
//
// To decode an entry, read the first 4 bytes to get the size of the entry,
// then read `size` bytes from the stream.
type LogEntryEncoder interface {
	Encode(*LogEntry) error
}

// NewLogEntryEncoder creates a protobuf stream encoder for log entries.
// This is used to write out  log entries to a stream.
func NewLogEntryEncoder(w io.Writer) LogEntryEncoder {
	return &logEntryEncoder{
		w:   w,
		buf: make([]byte, 1024),
	}
}

type logEntryEncoder struct {
	buf []byte
	w   io.Writer
}

func (e *logEntryEncoder) Encode(l *LogEntry) error {
	n := l.Size()

	total := n + binaryEncodeLen
	if total > len(e.buf) {
		e.buf = make([]byte, total)
	}
	binary.BigEndian.PutUint32(e.buf, uint32(n))

	if _, err := l.MarshalTo(e.buf[binaryEncodeLen:]); err != nil {
		return err
	}
	_, err := e.w.Write(e.buf[:total])
	return err
}

// LogEntryDecoder decodes log entries from a stream
// It is expected that the wire format is as defined by LogEntryEncoder.
type LogEntryDecoder interface {
	Decode(*LogEntry) error
}

// NewLogEntryDecoder creates a new stream decoder for log entries
func NewLogEntryDecoder(r io.Reader) LogEntryDecoder {
	return &logEntryDecoder{
		lenBuf: make([]byte, binaryEncodeLen),
		buf:    make([]byte, 1024),
		r:      r,
	}
}

type logEntryDecoder struct {
	r      io.Reader
	lenBuf []byte
	buf    []byte
}

func (d *logEntryDecoder) Decode(l *LogEntry) error {
	_, err := io.ReadFull(d.r, d.lenBuf)
	if err != nil {
		return err
	}

	size := int(binary.BigEndian.Uint32(d.lenBuf))
	if len(d.buf) < size {
		d.buf = make([]byte, size)
	}

	if _, err := io.ReadFull(d.r, d.buf[:size]); err != nil {
		return err
	}
	return l.Unmarshal(d.buf[:size])
}
