package packp

import (
	"bufio"
	"bytes"

	"gopkg.in/src-d/go-git.v4/plumbing"

	. "gopkg.in/check.v1"
)

type ServerResponseSuite struct{}

var _ = Suite(&ServerResponseSuite{})

func (s *ServerResponseSuite) TestDecodeNAK(c *C) {
	raw := "0008NAK\n"

	sr := &ServerResponse{}
	err := sr.Decode(bufio.NewReader(bytes.NewBufferString(raw)), false)
	c.Assert(err, IsNil)

	c.Assert(sr.ACKs, HasLen, 0)
}

func (s *ServerResponseSuite) TestDecodeACK(c *C) {
	raw := "0031ACK 6ecf0ef2c2dffb796033e5a02219af86ec6584e5\n"

	sr := &ServerResponse{}
	err := sr.Decode(bufio.NewReader(bytes.NewBufferString(raw)), false)
	c.Assert(err, IsNil)

	c.Assert(sr.ACKs, HasLen, 1)
	c.Assert(sr.ACKs[0], Equals, plumbing.NewHash("6ecf0ef2c2dffb796033e5a02219af86ec6584e5"))
}

func (s *ServerResponseSuite) TestDecodeMultipleACK(c *C) {
	raw := "" +
		"0031ACK 1111111111111111111111111111111111111111\n" +
		"0031ACK 6ecf0ef2c2dffb796033e5a02219af86ec6584e5\n" +
		"00080PACK\n"

	sr := &ServerResponse{}
	err := sr.Decode(bufio.NewReader(bytes.NewBufferString(raw)), false)
	c.Assert(err, IsNil)

	c.Assert(sr.ACKs, HasLen, 2)
	c.Assert(sr.ACKs[0], Equals, plumbing.NewHash("1111111111111111111111111111111111111111"))
	c.Assert(sr.ACKs[1], Equals, plumbing.NewHash("6ecf0ef2c2dffb796033e5a02219af86ec6584e5"))
}

func (s *ServerResponseSuite) TestDecodeMultipleACKWithSideband(c *C) {
	raw := "" +
		"0031ACK 1111111111111111111111111111111111111111\n" +
		"0031ACK 6ecf0ef2c2dffb796033e5a02219af86ec6584e5\n" +
		"00080aaaa\n"

	sr := &ServerResponse{}
	err := sr.Decode(bufio.NewReader(bytes.NewBufferString(raw)), false)
	c.Assert(err, IsNil)

	c.Assert(sr.ACKs, HasLen, 2)
	c.Assert(sr.ACKs[0], Equals, plumbing.NewHash("1111111111111111111111111111111111111111"))
	c.Assert(sr.ACKs[1], Equals, plumbing.NewHash("6ecf0ef2c2dffb796033e5a02219af86ec6584e5"))
}

func (s *ServerResponseSuite) TestDecodeMalformed(c *C) {
	raw := "0029ACK 6ecf0ef2c2dffb796033e5a02219af86ec6584e\n"

	sr := &ServerResponse{}
	err := sr.Decode(bufio.NewReader(bytes.NewBufferString(raw)), false)
	c.Assert(err, NotNil)
}

func (s *ServerResponseSuite) TestDecodeMultiACK(c *C) {
	sr := &ServerResponse{}
	err := sr.Decode(bufio.NewReader(bytes.NewBuffer(nil)), true)
	c.Assert(err, NotNil)
}
