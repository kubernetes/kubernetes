package packp

import (
	"bytes"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/pktline"

	. "gopkg.in/check.v1"
	"io/ioutil"
)

type UpdReqEncodeSuite struct{}

var _ = Suite(&UpdReqEncodeSuite{})

func (s *UpdReqEncodeSuite) testEncode(c *C, input *ReferenceUpdateRequest,
	expected []byte) {

	var buf bytes.Buffer
	c.Assert(input.Encode(&buf), IsNil)
	obtained := buf.Bytes()

	comment := Commentf("\nobtained = %s\nexpected = %s\n", string(obtained), string(expected))
	c.Assert(obtained, DeepEquals, expected, comment)
}

func (s *UpdReqEncodeSuite) TestZeroValue(c *C) {
	r := &ReferenceUpdateRequest{}
	var buf bytes.Buffer
	c.Assert(r.Encode(&buf), Equals, ErrEmptyCommands)

	r = NewReferenceUpdateRequest()
	c.Assert(r.Encode(&buf), Equals, ErrEmptyCommands)
}

func (s *UpdReqEncodeSuite) TestOneUpdateCommand(c *C) {
	hash1 := plumbing.NewHash("1ecf0ef2c2dffb796033e5a02219af86ec6584e5")
	hash2 := plumbing.NewHash("2ecf0ef2c2dffb796033e5a02219af86ec6584e5")
	name := plumbing.ReferenceName("myref")

	r := NewReferenceUpdateRequest()
	r.Commands = []*Command{
		{Name: name, Old: hash1, New: hash2},
	}

	expected := pktlines(c,
		"1ecf0ef2c2dffb796033e5a02219af86ec6584e5 2ecf0ef2c2dffb796033e5a02219af86ec6584e5 myref\x00",
		pktline.FlushString,
	)

	s.testEncode(c, r, expected)
}

func (s *UpdReqEncodeSuite) TestMultipleCommands(c *C) {
	hash1 := plumbing.NewHash("1ecf0ef2c2dffb796033e5a02219af86ec6584e5")
	hash2 := plumbing.NewHash("2ecf0ef2c2dffb796033e5a02219af86ec6584e5")

	r := NewReferenceUpdateRequest()
	r.Commands = []*Command{
		{Name: plumbing.ReferenceName("myref1"), Old: hash1, New: hash2},
		{Name: plumbing.ReferenceName("myref2"), Old: plumbing.ZeroHash, New: hash2},
		{Name: plumbing.ReferenceName("myref3"), Old: hash1, New: plumbing.ZeroHash},
	}

	expected := pktlines(c,
		"1ecf0ef2c2dffb796033e5a02219af86ec6584e5 2ecf0ef2c2dffb796033e5a02219af86ec6584e5 myref1\x00",
		"0000000000000000000000000000000000000000 2ecf0ef2c2dffb796033e5a02219af86ec6584e5 myref2",
		"1ecf0ef2c2dffb796033e5a02219af86ec6584e5 0000000000000000000000000000000000000000 myref3",
		pktline.FlushString,
	)

	s.testEncode(c, r, expected)
}

func (s *UpdReqEncodeSuite) TestMultipleCommandsAndCapabilities(c *C) {
	hash1 := plumbing.NewHash("1ecf0ef2c2dffb796033e5a02219af86ec6584e5")
	hash2 := plumbing.NewHash("2ecf0ef2c2dffb796033e5a02219af86ec6584e5")

	r := NewReferenceUpdateRequest()
	r.Commands = []*Command{
		{Name: plumbing.ReferenceName("myref1"), Old: hash1, New: hash2},
		{Name: plumbing.ReferenceName("myref2"), Old: plumbing.ZeroHash, New: hash2},
		{Name: plumbing.ReferenceName("myref3"), Old: hash1, New: plumbing.ZeroHash},
	}
	r.Capabilities.Add("shallow")

	expected := pktlines(c,
		"1ecf0ef2c2dffb796033e5a02219af86ec6584e5 2ecf0ef2c2dffb796033e5a02219af86ec6584e5 myref1\x00shallow",
		"0000000000000000000000000000000000000000 2ecf0ef2c2dffb796033e5a02219af86ec6584e5 myref2",
		"1ecf0ef2c2dffb796033e5a02219af86ec6584e5 0000000000000000000000000000000000000000 myref3",
		pktline.FlushString,
	)

	s.testEncode(c, r, expected)
}

func (s *UpdReqEncodeSuite) TestMultipleCommandsAndCapabilitiesShallow(c *C) {
	hash1 := plumbing.NewHash("1ecf0ef2c2dffb796033e5a02219af86ec6584e5")
	hash2 := plumbing.NewHash("2ecf0ef2c2dffb796033e5a02219af86ec6584e5")

	r := NewReferenceUpdateRequest()
	r.Commands = []*Command{
		{Name: plumbing.ReferenceName("myref1"), Old: hash1, New: hash2},
		{Name: plumbing.ReferenceName("myref2"), Old: plumbing.ZeroHash, New: hash2},
		{Name: plumbing.ReferenceName("myref3"), Old: hash1, New: plumbing.ZeroHash},
	}
	r.Capabilities.Add("shallow")
	r.Shallow = &hash1

	expected := pktlines(c,
		"shallow 1ecf0ef2c2dffb796033e5a02219af86ec6584e5",
		"1ecf0ef2c2dffb796033e5a02219af86ec6584e5 2ecf0ef2c2dffb796033e5a02219af86ec6584e5 myref1\x00shallow",
		"0000000000000000000000000000000000000000 2ecf0ef2c2dffb796033e5a02219af86ec6584e5 myref2",
		"1ecf0ef2c2dffb796033e5a02219af86ec6584e5 0000000000000000000000000000000000000000 myref3",
		pktline.FlushString,
	)

	s.testEncode(c, r, expected)
}

func (s *UpdReqEncodeSuite) TestWithPackfile(c *C) {
	hash1 := plumbing.NewHash("1ecf0ef2c2dffb796033e5a02219af86ec6584e5")
	hash2 := plumbing.NewHash("2ecf0ef2c2dffb796033e5a02219af86ec6584e5")
	name := plumbing.ReferenceName("myref")

	packfileContent := []byte("PACKabc")
	packfileReader := bytes.NewReader(packfileContent)
	packfileReadCloser := ioutil.NopCloser(packfileReader)

	r := NewReferenceUpdateRequest()
	r.Commands = []*Command{
		{Name: name, Old: hash1, New: hash2},
	}
	r.Packfile = packfileReadCloser

	expected := pktlines(c,
		"1ecf0ef2c2dffb796033e5a02219af86ec6584e5 2ecf0ef2c2dffb796033e5a02219af86ec6584e5 myref\x00",
		pktline.FlushString,
	)
	expected = append(expected, packfileContent...)

	s.testEncode(c, r, expected)
}
