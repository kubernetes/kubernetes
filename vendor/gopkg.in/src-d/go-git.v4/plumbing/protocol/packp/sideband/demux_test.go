package sideband

import (
	"bytes"
	"errors"
	"io"
	"io/ioutil"
	"testing"

	"gopkg.in/src-d/go-git.v4/plumbing/format/pktline"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type SidebandSuite struct{}

var _ = Suite(&SidebandSuite{})

func (s *SidebandSuite) TestDecode(c *C) {
	expected := []byte("abcdefghijklmnopqrstuvwxyz")

	buf := bytes.NewBuffer(nil)
	e := pktline.NewEncoder(buf)
	e.Encode(PackData.WithPayload(expected[0:8]))
	e.Encode(ProgressMessage.WithPayload([]byte{'F', 'O', 'O', '\n'}))
	e.Encode(PackData.WithPayload(expected[8:16]))
	e.Encode(PackData.WithPayload(expected[16:26]))

	content := make([]byte, 26)
	d := NewDemuxer(Sideband64k, buf)
	n, err := io.ReadFull(d, content)
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 26)
	c.Assert(content, DeepEquals, expected)
}

func (s *SidebandSuite) TestDecodeMoreThanContain(c *C) {
	expected := []byte("abcdefghijklmnopqrstuvwxyz")

	buf := bytes.NewBuffer(nil)
	e := pktline.NewEncoder(buf)
	e.Encode(PackData.WithPayload(expected))

	content := make([]byte, 42)
	d := NewDemuxer(Sideband64k, buf)
	n, err := io.ReadFull(d, content)
	c.Assert(err, Equals, io.ErrUnexpectedEOF)
	c.Assert(n, Equals, 26)
	c.Assert(content[0:26], DeepEquals, expected)
}

func (s *SidebandSuite) TestDecodeWithError(c *C) {
	expected := []byte("abcdefghijklmnopqrstuvwxyz")

	buf := bytes.NewBuffer(nil)
	e := pktline.NewEncoder(buf)
	e.Encode(PackData.WithPayload(expected[0:8]))
	e.Encode(ErrorMessage.WithPayload([]byte{'F', 'O', 'O', '\n'}))
	e.Encode(PackData.WithPayload(expected[8:16]))
	e.Encode(PackData.WithPayload(expected[16:26]))

	content := make([]byte, 26)
	d := NewDemuxer(Sideband64k, buf)
	n, err := io.ReadFull(d, content)
	c.Assert(err, ErrorMatches, "unexpected error: FOO\n")
	c.Assert(n, Equals, 8)
	c.Assert(content[0:8], DeepEquals, expected[0:8])
}

type mockReader struct{}

func (r *mockReader) Read([]byte) (int, error) { return 0, errors.New("foo") }

func (s *SidebandSuite) TestDecodeFromFailingReader(c *C) {
	content := make([]byte, 26)
	d := NewDemuxer(Sideband64k, &mockReader{})
	n, err := io.ReadFull(d, content)
	c.Assert(err, ErrorMatches, "foo")
	c.Assert(n, Equals, 0)
}

func (s *SidebandSuite) TestDecodeWithProgress(c *C) {
	expected := []byte("abcdefghijklmnopqrstuvwxyz")

	input := bytes.NewBuffer(nil)
	e := pktline.NewEncoder(input)
	e.Encode(PackData.WithPayload(expected[0:8]))
	e.Encode(ProgressMessage.WithPayload([]byte{'F', 'O', 'O', '\n'}))
	e.Encode(PackData.WithPayload(expected[8:16]))
	e.Encode(PackData.WithPayload(expected[16:26]))

	output := bytes.NewBuffer(nil)
	content := make([]byte, 26)
	d := NewDemuxer(Sideband64k, input)
	d.Progress = output

	n, err := io.ReadFull(d, content)
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 26)
	c.Assert(content, DeepEquals, expected)

	progress, err := ioutil.ReadAll(output)
	c.Assert(err, IsNil)
	c.Assert(progress, DeepEquals, []byte{'F', 'O', 'O', '\n'})
}

func (s *SidebandSuite) TestDecodeWithUnknownChannel(c *C) {

	buf := bytes.NewBuffer(nil)
	e := pktline.NewEncoder(buf)
	e.Encode([]byte{'4', 'F', 'O', 'O', '\n'})

	content := make([]byte, 26)
	d := NewDemuxer(Sideband64k, buf)
	n, err := io.ReadFull(d, content)
	c.Assert(err, ErrorMatches, "unknown channel 4FOO\n")
	c.Assert(n, Equals, 0)
}

func (s *SidebandSuite) TestDecodeWithPending(c *C) {
	expected := []byte("abcdefghijklmnopqrstuvwxyz")

	buf := bytes.NewBuffer(nil)
	e := pktline.NewEncoder(buf)
	e.Encode(PackData.WithPayload(expected[0:8]))
	e.Encode(PackData.WithPayload(expected[8:16]))
	e.Encode(PackData.WithPayload(expected[16:26]))

	content := make([]byte, 13)
	d := NewDemuxer(Sideband64k, buf)
	n, err := io.ReadFull(d, content)
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 13)
	c.Assert(content, DeepEquals, expected[0:13])

	n, err = d.Read(content)
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 13)
	c.Assert(content, DeepEquals, expected[13:26])
}

func (s *SidebandSuite) TestDecodeErrMaxPacked(c *C) {
	buf := bytes.NewBuffer(nil)
	e := pktline.NewEncoder(buf)
	e.Encode(PackData.WithPayload(bytes.Repeat([]byte{'0'}, MaxPackedSize+1)))

	content := make([]byte, 13)
	d := NewDemuxer(Sideband, buf)
	n, err := io.ReadFull(d, content)
	c.Assert(err, Equals, ErrMaxPackedExceeded)
	c.Assert(n, Equals, 0)

}
