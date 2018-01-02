package packp

import (
	"bytes"
	"io"
	"testing"

	"gopkg.in/src-d/go-git.v4/plumbing/format/pktline"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

// returns a byte slice with the pkt-lines for the given payloads.
func pktlines(c *C, payloads ...string) []byte {
	var buf bytes.Buffer
	e := pktline.NewEncoder(&buf)

	err := e.EncodeString(payloads...)
	c.Assert(err, IsNil, Commentf("building pktlines for %v\n", payloads))

	return buf.Bytes()
}

func toPktLines(c *C, payloads []string) io.Reader {
	var buf bytes.Buffer
	e := pktline.NewEncoder(&buf)
	err := e.EncodeString(payloads...)
	c.Assert(err, IsNil)

	return &buf
}
