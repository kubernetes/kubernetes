package binary

import (
	"bytes"
	"encoding/binary"

	. "gopkg.in/check.v1"
)

func (s *BinarySuite) TestWrite(c *C) {
	expected := bytes.NewBuffer(nil)
	err := binary.Write(expected, binary.BigEndian, int64(42))
	c.Assert(err, IsNil)
	err = binary.Write(expected, binary.BigEndian, int32(42))
	c.Assert(err, IsNil)

	buf := bytes.NewBuffer(nil)
	err = Write(buf, int64(42), int32(42))
	c.Assert(err, IsNil)
	c.Assert(buf, DeepEquals, expected)
}

func (s *BinarySuite) TestWriteUint32(c *C) {
	expected := bytes.NewBuffer(nil)
	err := binary.Write(expected, binary.BigEndian, int32(42))
	c.Assert(err, IsNil)

	buf := bytes.NewBuffer(nil)
	err = WriteUint32(buf, 42)
	c.Assert(err, IsNil)
	c.Assert(buf, DeepEquals, expected)
}

func (s *BinarySuite) TestWriteUint16(c *C) {
	expected := bytes.NewBuffer(nil)
	err := binary.Write(expected, binary.BigEndian, int16(42))
	c.Assert(err, IsNil)

	buf := bytes.NewBuffer(nil)
	err = WriteUint16(buf, 42)
	c.Assert(err, IsNil)
	c.Assert(buf, DeepEquals, expected)
}

func (s *BinarySuite) TestWriteVariableWidthInt(c *C) {
	buf := bytes.NewBuffer(nil)

	err := WriteVariableWidthInt(buf, 366)
	c.Assert(err, IsNil)
	c.Assert(buf.Bytes(), DeepEquals, []byte{129, 110})
}

func (s *BinarySuite) TestWriteVariableWidthIntShort(c *C) {
	buf := bytes.NewBuffer(nil)

	err := WriteVariableWidthInt(buf, 19)
	c.Assert(err, IsNil)
	c.Assert(buf.Bytes(), DeepEquals, []byte{19})
}
