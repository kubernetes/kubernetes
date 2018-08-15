package pq

import (
	"bytes"
	"encoding/binary"

	"github.com/lib/pq/oid"
)

type readBuf []byte

func (b *readBuf) int32() (n int) {
	n = int(int32(binary.BigEndian.Uint32(*b)))
	*b = (*b)[4:]
	return
}

func (b *readBuf) oid() (n oid.Oid) {
	n = oid.Oid(binary.BigEndian.Uint32(*b))
	*b = (*b)[4:]
	return
}

// N.B: this is actually an unsigned 16-bit integer, unlike int32
func (b *readBuf) int16() (n int) {
	n = int(binary.BigEndian.Uint16(*b))
	*b = (*b)[2:]
	return
}

func (b *readBuf) string() string {
	i := bytes.IndexByte(*b, 0)
	if i < 0 {
		errorf("invalid message format; expected string terminator")
	}
	s := (*b)[:i]
	*b = (*b)[i+1:]
	return string(s)
}

func (b *readBuf) next(n int) (v []byte) {
	v = (*b)[:n]
	*b = (*b)[n:]
	return
}

func (b *readBuf) byte() byte {
	return b.next(1)[0]
}

type writeBuf struct {
	buf []byte
	pos int
}

func (b *writeBuf) int32(n int) {
	x := make([]byte, 4)
	binary.BigEndian.PutUint32(x, uint32(n))
	b.buf = append(b.buf, x...)
}

func (b *writeBuf) int16(n int) {
	x := make([]byte, 2)
	binary.BigEndian.PutUint16(x, uint16(n))
	b.buf = append(b.buf, x...)
}

func (b *writeBuf) string(s string) {
	b.buf = append(b.buf, (s + "\000")...)
}

func (b *writeBuf) byte(c byte) {
	b.buf = append(b.buf, c)
}

func (b *writeBuf) bytes(v []byte) {
	b.buf = append(b.buf, v...)
}

func (b *writeBuf) wrap() []byte {
	p := b.buf[b.pos:]
	binary.BigEndian.PutUint32(p, uint32(len(p)))
	return b.buf
}

func (b *writeBuf) next(c byte) {
	p := b.buf[b.pos:]
	binary.BigEndian.PutUint32(p, uint32(len(p)))
	b.pos = len(b.buf) + 1
	b.buf = append(b.buf, c, 0, 0, 0, 0)
}
