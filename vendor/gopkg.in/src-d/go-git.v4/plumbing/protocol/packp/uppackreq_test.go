package packp

import (
	"bytes"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability"

	. "gopkg.in/check.v1"
)

type UploadPackRequestSuite struct{}

var _ = Suite(&UploadPackRequestSuite{})

func (s *UploadPackRequestSuite) TestNewUploadPackRequestFromCapabilities(c *C) {
	cap := capability.NewList()
	cap.Set(capability.Agent, "foo")

	r := NewUploadPackRequestFromCapabilities(cap)
	c.Assert(r.Capabilities.String(), Equals, "agent=go-git/4.x")
}

func (s *UploadPackRequestSuite) TestIsEmpty(c *C) {
	r := NewUploadPackRequest()
	r.Wants = append(r.Wants, plumbing.NewHash("d82f291cde9987322c8a0c81a325e1ba6159684c"))
	r.Wants = append(r.Wants, plumbing.NewHash("2b41ef280fdb67a9b250678686a0c3e03b0a9989"))
	r.Haves = append(r.Haves, plumbing.NewHash("6ecf0ef2c2dffb796033e5a02219af86ec6584e5"))

	c.Assert(r.IsEmpty(), Equals, false)

	r = NewUploadPackRequest()
	r.Wants = append(r.Wants, plumbing.NewHash("d82f291cde9987322c8a0c81a325e1ba6159684c"))
	r.Wants = append(r.Wants, plumbing.NewHash("2b41ef280fdb67a9b250678686a0c3e03b0a9989"))
	r.Haves = append(r.Haves, plumbing.NewHash("d82f291cde9987322c8a0c81a325e1ba6159684c"))

	c.Assert(r.IsEmpty(), Equals, false)

	r = NewUploadPackRequest()
	r.Wants = append(r.Wants, plumbing.NewHash("d82f291cde9987322c8a0c81a325e1ba6159684c"))
	r.Haves = append(r.Haves, plumbing.NewHash("d82f291cde9987322c8a0c81a325e1ba6159684c"))

	c.Assert(r.IsEmpty(), Equals, true)
}

type UploadHavesSuite struct{}

var _ = Suite(&UploadHavesSuite{})

func (s *UploadHavesSuite) TestEncode(c *C) {
	uh := &UploadHaves{}
	uh.Haves = append(uh.Haves,
		plumbing.NewHash("1111111111111111111111111111111111111111"),
		plumbing.NewHash("3333333333333333333333333333333333333333"),
		plumbing.NewHash("1111111111111111111111111111111111111111"),
		plumbing.NewHash("2222222222222222222222222222222222222222"),
		plumbing.NewHash("1111111111111111111111111111111111111111"),
	)

	buf := bytes.NewBuffer(nil)
	err := uh.Encode(buf, true)
	c.Assert(err, IsNil)
	c.Assert(buf.String(), Equals, ""+
		"0032have 1111111111111111111111111111111111111111\n"+
		"0032have 2222222222222222222222222222222222222222\n"+
		"0032have 3333333333333333333333333333333333333333\n"+
		"0000",
	)
}
