package roaring

import (
	"encoding/binary"
	"io"
)

type byteInput interface {
	// next returns a slice containing the next n bytes from the buffer,
	// advancing the buffer as if the bytes had been returned by Read.
	next(n int) ([]byte, error)
	// readUInt32 reads uint32 with LittleEndian order
	readUInt32() (uint32, error)
	// readUInt16 reads uint16 with LittleEndian order
	readUInt16() (uint16, error)
	// getReadBytes returns read bytes
	getReadBytes() int64
	// skipBytes skips exactly n bytes
	skipBytes(n int) error
}

func newByteInputFromReader(reader io.Reader) byteInput {
	return &byteInputAdapter{
		r:         reader,
		readBytes: 0,
	}
}

func newByteInput(buf []byte) byteInput {
	return &byteBuffer{
		buf: buf,
		off: 0,
	}
}

type byteBuffer struct {
	buf []byte
	off int
}

// next returns a slice containing the next n bytes from the reader
// If there are fewer bytes than the given n, io.ErrUnexpectedEOF will be returned
func (b *byteBuffer) next(n int) ([]byte, error) {
	m := len(b.buf) - b.off

	if n > m {
		return nil, io.ErrUnexpectedEOF
	}

	data := b.buf[b.off : b.off+n]
	b.off += n

	return data, nil
}

// readUInt32 reads uint32 with LittleEndian order
func (b *byteBuffer) readUInt32() (uint32, error) {
	if len(b.buf)-b.off < 4 {
		return 0, io.ErrUnexpectedEOF
	}

	v := binary.LittleEndian.Uint32(b.buf[b.off:])
	b.off += 4

	return v, nil
}

// readUInt16 reads uint16 with LittleEndian order
func (b *byteBuffer) readUInt16() (uint16, error) {
	if len(b.buf)-b.off < 2 {
		return 0, io.ErrUnexpectedEOF
	}

	v := binary.LittleEndian.Uint16(b.buf[b.off:])
	b.off += 2

	return v, nil
}

// getReadBytes returns read bytes
func (b *byteBuffer) getReadBytes() int64 {
	return int64(b.off)
}

// skipBytes skips exactly n bytes
func (b *byteBuffer) skipBytes(n int) error {
	m := len(b.buf) - b.off

	if n > m {
		return io.ErrUnexpectedEOF
	}

	b.off += n

	return nil
}

// reset resets the given buffer with a new byte slice
func (b *byteBuffer) reset(buf []byte) {
	b.buf = buf
	b.off = 0
}

type byteInputAdapter struct {
	r         io.Reader
	readBytes int
}

// next returns a slice containing the next n bytes from the buffer,
// advancing the buffer as if the bytes had been returned by Read.
func (b *byteInputAdapter) next(n int) ([]byte, error) {
	buf := make([]byte, n)
	m, err := io.ReadAtLeast(b.r, buf, n)
	b.readBytes += m

	if err != nil {
		return nil, err
	}

	return buf, nil
}

// readUInt32 reads uint32 with LittleEndian order
func (b *byteInputAdapter) readUInt32() (uint32, error) {
	buf, err := b.next(4)

	if err != nil {
		return 0, err
	}

	return binary.LittleEndian.Uint32(buf), nil
}

// readUInt16 reads uint16 with LittleEndian order
func (b *byteInputAdapter) readUInt16() (uint16, error) {
	buf, err := b.next(2)

	if err != nil {
		return 0, err
	}

	return binary.LittleEndian.Uint16(buf), nil
}

// getReadBytes returns read bytes
func (b *byteInputAdapter) getReadBytes() int64 {
	return int64(b.readBytes)
}

// skipBytes skips exactly n bytes
func (b *byteInputAdapter) skipBytes(n int) error {
	_, err := b.next(n)

	return err
}

// reset resets the given buffer with a new stream
func (b *byteInputAdapter) reset(stream io.Reader) {
	b.r = stream
	b.readBytes = 0
}
