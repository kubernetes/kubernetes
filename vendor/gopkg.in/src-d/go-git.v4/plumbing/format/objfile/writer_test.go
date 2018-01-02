package objfile

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"

	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-git.v4/plumbing"
)

type SuiteWriter struct{}

var _ = Suite(&SuiteWriter{})

func (s *SuiteWriter) TestWriteObjfile(c *C) {
	for k, fixture := range objfileFixtures {
		buffer := bytes.NewBuffer(nil)

		com := fmt.Sprintf("test %d: ", k)
		hash := plumbing.NewHash(fixture.hash)
		content, _ := base64.StdEncoding.DecodeString(fixture.content)

		// Write the data out to the buffer
		testWriter(c, buffer, hash, fixture.t, content)

		// Read the data back in from the buffer to be sure it matches
		testReader(c, buffer, hash, fixture.t, content, com)
	}
}

func testWriter(c *C, dest io.Writer, hash plumbing.Hash, t plumbing.ObjectType, content []byte) {
	size := int64(len(content))
	w := NewWriter(dest)

	err := w.WriteHeader(t, size)
	c.Assert(err, IsNil)

	written, err := io.Copy(w, bytes.NewReader(content))
	c.Assert(err, IsNil)
	c.Assert(written, Equals, size)

	c.Assert(w.Hash(), Equals, hash)
	c.Assert(w.Close(), IsNil)
}

func (s *SuiteWriter) TestWriteOverflow(c *C) {
	buf := bytes.NewBuffer(nil)
	w := NewWriter(buf)

	err := w.WriteHeader(plumbing.BlobObject, 8)
	c.Assert(err, IsNil)

	n, err := w.Write([]byte("1234"))
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 4)

	n, err = w.Write([]byte("56789"))
	c.Assert(err, Equals, ErrOverflow)
	c.Assert(n, Equals, 4)
}

func (s *SuiteWriter) TestNewWriterInvalidType(c *C) {
	buf := bytes.NewBuffer(nil)
	w := NewWriter(buf)

	err := w.WriteHeader(plumbing.InvalidObject, 8)
	c.Assert(err, Equals, plumbing.ErrInvalidType)
}

func (s *SuiteWriter) TestNewWriterInvalidSize(c *C) {
	buf := bytes.NewBuffer(nil)
	w := NewWriter(buf)

	err := w.WriteHeader(plumbing.BlobObject, -1)
	c.Assert(err, Equals, ErrNegativeSize)
	err = w.WriteHeader(plumbing.BlobObject, -1651860)
	c.Assert(err, Equals, ErrNegativeSize)
}
