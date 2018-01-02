package packp

import (
	"bytes"
	"time"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/pktline"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability"

	. "gopkg.in/check.v1"
)

type UlReqEncodeSuite struct{}

var _ = Suite(&UlReqEncodeSuite{})

func testUlReqEncode(c *C, ur *UploadRequest, expectedPayloads []string) {
	var buf bytes.Buffer
	e := newUlReqEncoder(&buf)

	err := e.Encode(ur)
	c.Assert(err, IsNil)
	obtained := buf.Bytes()

	expected := pktlines(c, expectedPayloads...)

	comment := Commentf("\nobtained = %s\nexpected = %s\n", string(obtained), string(expected))

	c.Assert(obtained, DeepEquals, expected, comment)
}

func testUlReqEncodeError(c *C, ur *UploadRequest, expectedErrorRegEx string) {
	var buf bytes.Buffer
	e := newUlReqEncoder(&buf)

	err := e.Encode(ur)
	c.Assert(err, ErrorMatches, expectedErrorRegEx)
}

func (s *UlReqEncodeSuite) TestZeroValue(c *C) {
	ur := NewUploadRequest()
	expectedErrorRegEx := ".*empty wants.*"

	testUlReqEncodeError(c, ur, expectedErrorRegEx)
}

func (s *UlReqEncodeSuite) TestOneWant(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))

	expected := []string{
		"want 1111111111111111111111111111111111111111\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}

func (s *UlReqEncodeSuite) TestOneWantWithCapabilities(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	ur.Capabilities.Add(capability.MultiACK)
	ur.Capabilities.Add(capability.OFSDelta)
	ur.Capabilities.Add(capability.Sideband)
	ur.Capabilities.Add(capability.SymRef, "HEAD:/refs/heads/master")
	ur.Capabilities.Add(capability.ThinPack)

	expected := []string{
		"want 1111111111111111111111111111111111111111 multi_ack ofs-delta side-band symref=HEAD:/refs/heads/master thin-pack\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}

func (s *UlReqEncodeSuite) TestWants(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants,
		plumbing.NewHash("4444444444444444444444444444444444444444"),
		plumbing.NewHash("1111111111111111111111111111111111111111"),
		plumbing.NewHash("3333333333333333333333333333333333333333"),
		plumbing.NewHash("2222222222222222222222222222222222222222"),
		plumbing.NewHash("5555555555555555555555555555555555555555"),
	)

	expected := []string{
		"want 1111111111111111111111111111111111111111\n",
		"want 2222222222222222222222222222222222222222\n",
		"want 3333333333333333333333333333333333333333\n",
		"want 4444444444444444444444444444444444444444\n",
		"want 5555555555555555555555555555555555555555\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}

func (s *UlReqEncodeSuite) TestWantsDuplicates(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants,
		plumbing.NewHash("4444444444444444444444444444444444444444"),
		plumbing.NewHash("1111111111111111111111111111111111111111"),
		plumbing.NewHash("3333333333333333333333333333333333333333"),
		plumbing.NewHash("1111111111111111111111111111111111111111"),
		plumbing.NewHash("2222222222222222222222222222222222222222"),
		plumbing.NewHash("1111111111111111111111111111111111111111"),
	)

	expected := []string{
		"want 1111111111111111111111111111111111111111\n",
		"want 2222222222222222222222222222222222222222\n",
		"want 3333333333333333333333333333333333333333\n",
		"want 4444444444444444444444444444444444444444\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}

func (s *UlReqEncodeSuite) TestWantsWithCapabilities(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants,
		plumbing.NewHash("4444444444444444444444444444444444444444"),
		plumbing.NewHash("1111111111111111111111111111111111111111"),
		plumbing.NewHash("3333333333333333333333333333333333333333"),
		plumbing.NewHash("2222222222222222222222222222222222222222"),
		plumbing.NewHash("5555555555555555555555555555555555555555"),
	)

	ur.Capabilities.Add(capability.MultiACK)
	ur.Capabilities.Add(capability.OFSDelta)
	ur.Capabilities.Add(capability.Sideband)
	ur.Capabilities.Add(capability.SymRef, "HEAD:/refs/heads/master")
	ur.Capabilities.Add(capability.ThinPack)

	expected := []string{
		"want 1111111111111111111111111111111111111111 multi_ack ofs-delta side-band symref=HEAD:/refs/heads/master thin-pack\n",
		"want 2222222222222222222222222222222222222222\n",
		"want 3333333333333333333333333333333333333333\n",
		"want 4444444444444444444444444444444444444444\n",
		"want 5555555555555555555555555555555555555555\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}

func (s *UlReqEncodeSuite) TestShallow(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	ur.Capabilities.Add(capability.MultiACK)
	ur.Shallows = append(ur.Shallows, plumbing.NewHash("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))

	expected := []string{
		"want 1111111111111111111111111111111111111111 multi_ack\n",
		"shallow aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}

func (s *UlReqEncodeSuite) TestManyShallows(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	ur.Capabilities.Add(capability.MultiACK)
	ur.Shallows = append(ur.Shallows,
		plumbing.NewHash("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
		plumbing.NewHash("dddddddddddddddddddddddddddddddddddddddd"),
		plumbing.NewHash("cccccccccccccccccccccccccccccccccccccccc"),
		plumbing.NewHash("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
	)

	expected := []string{
		"want 1111111111111111111111111111111111111111 multi_ack\n",
		"shallow aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
		"shallow bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n",
		"shallow cccccccccccccccccccccccccccccccccccccccc\n",
		"shallow dddddddddddddddddddddddddddddddddddddddd\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}

func (s *UlReqEncodeSuite) TestShallowsDuplicate(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	ur.Capabilities.Add(capability.MultiACK)
	ur.Shallows = append(ur.Shallows,
		plumbing.NewHash("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
		plumbing.NewHash("cccccccccccccccccccccccccccccccccccccccc"),
		plumbing.NewHash("cccccccccccccccccccccccccccccccccccccccc"),
		plumbing.NewHash("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
	)

	expected := []string{
		"want 1111111111111111111111111111111111111111 multi_ack\n",
		"shallow aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
		"shallow bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n",
		"shallow cccccccccccccccccccccccccccccccccccccccc\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}

func (s *UlReqEncodeSuite) TestDepthCommits(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	ur.Depth = DepthCommits(1234)

	expected := []string{
		"want 1111111111111111111111111111111111111111\n",
		"deepen 1234\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}

func (s *UlReqEncodeSuite) TestDepthSinceUTC(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	since := time.Date(2015, time.January, 2, 3, 4, 5, 0, time.UTC)
	ur.Depth = DepthSince(since)

	expected := []string{
		"want 1111111111111111111111111111111111111111\n",
		"deepen-since 1420167845\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}

func (s *UlReqEncodeSuite) TestDepthSinceNonUTC(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	berlin, err := time.LoadLocation("Europe/Berlin")
	c.Assert(err, IsNil)
	since := time.Date(2015, time.January, 2, 3, 4, 5, 0, berlin)
	// since value is 2015-01-02 03:04:05 +0100 UTC (Europe/Berlin) or
	// 2015-01-02 02:04:05 +0000 UTC, which is 1420164245 Unix seconds.
	ur.Depth = DepthSince(since)

	expected := []string{
		"want 1111111111111111111111111111111111111111\n",
		"deepen-since 1420164245\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}

func (s *UlReqEncodeSuite) TestDepthReference(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants, plumbing.NewHash("1111111111111111111111111111111111111111"))
	ur.Depth = DepthReference("refs/heads/feature-foo")

	expected := []string{
		"want 1111111111111111111111111111111111111111\n",
		"deepen-not refs/heads/feature-foo\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}

func (s *UlReqEncodeSuite) TestAll(c *C) {
	ur := NewUploadRequest()
	ur.Wants = append(ur.Wants,
		plumbing.NewHash("4444444444444444444444444444444444444444"),
		plumbing.NewHash("1111111111111111111111111111111111111111"),
		plumbing.NewHash("3333333333333333333333333333333333333333"),
		plumbing.NewHash("2222222222222222222222222222222222222222"),
		plumbing.NewHash("5555555555555555555555555555555555555555"),
	)

	ur.Capabilities.Add(capability.MultiACK)
	ur.Capabilities.Add(capability.OFSDelta)
	ur.Capabilities.Add(capability.Sideband)
	ur.Capabilities.Add(capability.SymRef, "HEAD:/refs/heads/master")
	ur.Capabilities.Add(capability.ThinPack)

	ur.Shallows = append(ur.Shallows, plumbing.NewHash("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"))
	ur.Shallows = append(ur.Shallows, plumbing.NewHash("dddddddddddddddddddddddddddddddddddddddd"))
	ur.Shallows = append(ur.Shallows, plumbing.NewHash("cccccccccccccccccccccccccccccccccccccccc"))
	ur.Shallows = append(ur.Shallows, plumbing.NewHash("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))

	since := time.Date(2015, time.January, 2, 3, 4, 5, 0, time.UTC)
	ur.Depth = DepthSince(since)

	expected := []string{
		"want 1111111111111111111111111111111111111111 multi_ack ofs-delta side-band symref=HEAD:/refs/heads/master thin-pack\n",
		"want 2222222222222222222222222222222222222222\n",
		"want 3333333333333333333333333333333333333333\n",
		"want 4444444444444444444444444444444444444444\n",
		"want 5555555555555555555555555555555555555555\n",
		"shallow aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
		"shallow bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n",
		"shallow cccccccccccccccccccccccccccccccccccccccc\n",
		"shallow dddddddddddddddddddddddddddddddddddddddd\n",
		"deepen-since 1420167845\n",
		pktline.FlushString,
	}

	testUlReqEncode(c, ur, expected)
}
