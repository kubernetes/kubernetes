package packp

import (
	"fmt"
	"os"
	"strings"
	"time"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/pktline"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability"

	. "gopkg.in/check.v1"
)

type UlReqSuite struct{}

var _ = Suite(&UlReqSuite{})

func (s *UlReqSuite) TestNewUploadRequestFromCapabilities(c *C) {
	cap := capability.NewList()
	cap.Set(capability.Sideband)
	cap.Set(capability.Sideband64k)
	cap.Set(capability.MultiACK)
	cap.Set(capability.MultiACKDetailed)
	cap.Set(capability.ThinPack)
	cap.Set(capability.OFSDelta)
	cap.Set(capability.Agent, "foo")

	r := NewUploadRequestFromCapabilities(cap)
	c.Assert(r.Capabilities.String(), Equals,
		"multi_ack_detailed side-band-64k thin-pack ofs-delta agent=go-git/4.x",
	)
}

func (s *UlReqSuite) TestValidateWants(c *C) {
	r := NewUploadRequest()
	err := r.Validate()
	c.Assert(err, NotNil)

	r.Wants = append(r.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	err = r.Validate()
	c.Assert(err, IsNil)
}

func (s *UlReqSuite) TestValidateShallows(c *C) {
	r := NewUploadRequest()
	r.Wants = append(r.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	r.Shallows = append(r.Shallows, plumbing.NewHash("2222222222222222222222222222222222222222"))
	err := r.Validate()
	c.Assert(err, NotNil)

	r.Capabilities.Set(capability.Shallow)
	err = r.Validate()
	c.Assert(err, IsNil)
}

func (s *UlReqSuite) TestValidateDepthCommits(c *C) {
	r := NewUploadRequest()
	r.Wants = append(r.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	r.Depth = DepthCommits(42)

	err := r.Validate()
	c.Assert(err, NotNil)

	r.Capabilities.Set(capability.Shallow)
	err = r.Validate()
	c.Assert(err, IsNil)
}

func (s *UlReqSuite) TestValidateDepthReference(c *C) {
	r := NewUploadRequest()
	r.Wants = append(r.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	r.Depth = DepthReference("1111111111111111111111111111111111111111")

	err := r.Validate()
	c.Assert(err, NotNil)

	r.Capabilities.Set(capability.DeepenNot)
	err = r.Validate()
	c.Assert(err, IsNil)
}

func (s *UlReqSuite) TestValidateDepthSince(c *C) {
	r := NewUploadRequest()
	r.Wants = append(r.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	r.Depth = DepthSince(time.Now())

	err := r.Validate()
	c.Assert(err, NotNil)

	r.Capabilities.Set(capability.DeepenSince)
	err = r.Validate()
	c.Assert(err, IsNil)
}

func (s *UlReqSuite) TestValidateConflictSideband(c *C) {
	r := NewUploadRequest()
	r.Wants = append(r.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	r.Capabilities.Set(capability.Sideband)
	r.Capabilities.Set(capability.Sideband64k)
	err := r.Validate()
	c.Assert(err, NotNil)
}

func (s *UlReqSuite) TestValidateConflictMultiACK(c *C) {
	r := NewUploadRequest()
	r.Wants = append(r.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	r.Capabilities.Set(capability.MultiACK)
	r.Capabilities.Set(capability.MultiACKDetailed)
	err := r.Validate()
	c.Assert(err, NotNil)
}

func ExampleUploadRequest_Encode() {
	// Create an empty UlReq with the contents you want...
	ur := NewUploadRequest()

	// Add a couple of wants
	ur.Wants = append(ur.Wants, plumbing.NewHash("3333333333333333333333333333333333333333"))
	ur.Wants = append(ur.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	ur.Wants = append(ur.Wants, plumbing.NewHash("2222222222222222222222222222222222222222"))

	// And some capabilities you will like the server to use
	ur.Capabilities.Add(capability.OFSDelta)
	ur.Capabilities.Add(capability.SymRef, "HEAD:/refs/heads/master")

	// Add a couple of shallows
	ur.Shallows = append(ur.Shallows, plumbing.NewHash("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"))
	ur.Shallows = append(ur.Shallows, plumbing.NewHash("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))

	// And retrict the answer of the server to commits newer than "2015-01-02 03:04:05 UTC"
	since := time.Date(2015, time.January, 2, 3, 4, 5, 0, time.UTC)
	ur.Depth = DepthSince(since)

	// Create a new Encode for the stdout...
	e := newUlReqEncoder(os.Stdout)
	// ...and encode the upload-request to it.
	_ = e.Encode(ur) // ignoring errors for brevity
	// Output:
	// 005bwant 1111111111111111111111111111111111111111 ofs-delta symref=HEAD:/refs/heads/master
	// 0032want 2222222222222222222222222222222222222222
	// 0032want 3333333333333333333333333333333333333333
	// 0035shallow aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
	// 0035shallow bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
	// 001cdeepen-since 1420167845
	// 0000
}

func ExampleUploadRequest_Decode() {
	// Here is a raw advertised-ref message.
	raw := "" +
		"005bwant 1111111111111111111111111111111111111111 ofs-delta symref=HEAD:/refs/heads/master\n" +
		"0032want 2222222222222222222222222222222222222222\n" +
		"0032want 3333333333333333333333333333333333333333\n" +
		"0035shallow aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n" +
		"0035shallow bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n" +
		"001cdeepen-since 1420167845\n" + // 2015-01-02 03:04:05 +0000 UTC
		pktline.FlushString

	// Use the raw message as our input.
	input := strings.NewReader(raw)

	// Create the Decoder reading from our input.
	d := newUlReqDecoder(input)

	// Decode the input into a newly allocated UlReq value.
	ur := NewUploadRequest()
	_ = d.Decode(ur) // error check ignored for brevity

	// Do something interesting with the UlReq, e.g. print its contents.
	fmt.Println("capabilities =", ur.Capabilities.String())
	fmt.Println("wants =", ur.Wants)
	fmt.Println("shallows =", ur.Shallows)
	switch depth := ur.Depth.(type) {
	case DepthCommits:
		fmt.Println("depth =", int(depth))
	case DepthSince:
		fmt.Println("depth =", time.Time(depth))
	case DepthReference:
		fmt.Println("depth =", string(depth))
	}
	// Output:
	// capabilities = ofs-delta symref=HEAD:/refs/heads/master
	// wants = [1111111111111111111111111111111111111111 2222222222222222222222222222222222222222 3333333333333333333333333333333333333333]
	// shallows = [aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb]
	// depth = 2015-01-02 03:04:05 +0000 UTC
}
