package sideband

import (
	"bytes"

	. "gopkg.in/check.v1"
)

func (s *SidebandSuite) TestMuxerWrite(c *C) {
	buf := bytes.NewBuffer(nil)

	m := NewMuxer(Sideband, buf)

	n, err := m.Write(bytes.Repeat([]byte{'F'}, (MaxPackedSize-1)*2))
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 1998)
	c.Assert(buf.Len(), Equals, 2008)
}

func (s *SidebandSuite) TestMuxerWriteChannelMultipleChannels(c *C) {
	buf := bytes.NewBuffer(nil)

	m := NewMuxer(Sideband, buf)

	n, err := m.WriteChannel(PackData, bytes.Repeat([]byte{'D'}, 4))
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 4)

	n, err = m.WriteChannel(ProgressMessage, bytes.Repeat([]byte{'P'}, 4))
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 4)

	n, err = m.WriteChannel(PackData, bytes.Repeat([]byte{'D'}, 4))
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 4)

	c.Assert(buf.Len(), Equals, 27)
	c.Assert(buf.String(), Equals, "0009\x01DDDD0009\x02PPPP0009\x01DDDD")
}
