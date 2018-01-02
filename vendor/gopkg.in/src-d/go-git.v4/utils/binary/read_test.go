package binary

import (
	"bytes"
	"encoding/binary"
	"testing"

	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-git.v4/plumbing"
)

func Test(t *testing.T) { TestingT(t) }

type BinarySuite struct{}

var _ = Suite(&BinarySuite{})

func (s *BinarySuite) TestRead(c *C) {
	buf := bytes.NewBuffer(nil)
	err := binary.Write(buf, binary.BigEndian, int64(42))
	c.Assert(err, IsNil)
	err = binary.Write(buf, binary.BigEndian, int32(42))
	c.Assert(err, IsNil)

	var i64 int64
	var i32 int32
	err = Read(buf, &i64, &i32)
	c.Assert(err, IsNil)
	c.Assert(i64, Equals, int64(42))
	c.Assert(i32, Equals, int32(42))
}

func (s *BinarySuite) TestReadUntil(c *C) {
	buf := bytes.NewBuffer([]byte("foo bar"))

	b, err := ReadUntil(buf, ' ')
	c.Assert(err, IsNil)
	c.Assert(b, HasLen, 3)
	c.Assert(string(b), Equals, "foo")
}

func (s *BinarySuite) TestReadVariableWidthInt(c *C) {
	buf := bytes.NewBuffer([]byte{129, 110})

	i, err := ReadVariableWidthInt(buf)
	c.Assert(err, IsNil)
	c.Assert(i, Equals, int64(366))
}

func (s *BinarySuite) TestReadVariableWidthIntShort(c *C) {
	buf := bytes.NewBuffer([]byte{19})

	i, err := ReadVariableWidthInt(buf)
	c.Assert(err, IsNil)
	c.Assert(i, Equals, int64(19))
}

func (s *BinarySuite) TestReadUint32(c *C) {
	buf := bytes.NewBuffer(nil)
	err := binary.Write(buf, binary.BigEndian, uint32(42))
	c.Assert(err, IsNil)

	i32, err := ReadUint32(buf)
	c.Assert(err, IsNil)
	c.Assert(i32, Equals, uint32(42))
}

func (s *BinarySuite) TestReadUint16(c *C) {
	buf := bytes.NewBuffer(nil)
	err := binary.Write(buf, binary.BigEndian, uint16(42))
	c.Assert(err, IsNil)

	i32, err := ReadUint16(buf)
	c.Assert(err, IsNil)
	c.Assert(i32, Equals, uint16(42))
}

func (s *BinarySuite) TestReadHash(c *C) {
	expected := plumbing.NewHash("43aec75c611f22c73b27ece2841e6ccca592f285")
	buf := bytes.NewBuffer(nil)
	err := binary.Write(buf, binary.BigEndian, expected)
	c.Assert(err, IsNil)

	hash, err := ReadHash(buf)
	c.Assert(err, IsNil)
	c.Assert(hash.String(), Equals, expected.String())
}

func (s *BinarySuite) TestIsBinary(c *C) {
	buf := bytes.NewBuffer(nil)
	buf.Write(bytes.Repeat([]byte{'A'}, sniffLen))
	buf.Write([]byte{0})
	ok, err := IsBinary(buf)
	c.Assert(err, IsNil)
	c.Assert(ok, Equals, false)

	buf.Reset()

	buf.Write(bytes.Repeat([]byte{'A'}, sniffLen-1))
	buf.Write([]byte{0})
	ok, err = IsBinary(buf)
	c.Assert(err, IsNil)
	c.Assert(ok, Equals, true)

	buf.Reset()

	buf.Write(bytes.Repeat([]byte{'A'}, 10))
	ok, err = IsBinary(buf)
	c.Assert(err, IsNil)
	c.Assert(ok, Equals, false)
}
