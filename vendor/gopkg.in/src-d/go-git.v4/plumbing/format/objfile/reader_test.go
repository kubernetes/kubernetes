package objfile

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"io/ioutil"

	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-git.v4/plumbing"
)

type SuiteReader struct{}

var _ = Suite(&SuiteReader{})

func (s *SuiteReader) TestReadObjfile(c *C) {
	for k, fixture := range objfileFixtures {
		com := fmt.Sprintf("test %d: ", k)
		hash := plumbing.NewHash(fixture.hash)
		content, _ := base64.StdEncoding.DecodeString(fixture.content)
		data, _ := base64.StdEncoding.DecodeString(fixture.data)

		testReader(c, bytes.NewReader(data), hash, fixture.t, content, com)
	}
}

func testReader(c *C, source io.Reader, hash plumbing.Hash, t plumbing.ObjectType, content []byte, com string) {
	r, err := NewReader(source)
	c.Assert(err, IsNil)

	typ, size, err := r.Header()
	c.Assert(err, IsNil)
	c.Assert(typ, Equals, t)
	c.Assert(content, HasLen, int(size))

	rc, err := ioutil.ReadAll(r)
	c.Assert(err, IsNil)
	c.Assert(rc, DeepEquals, content, Commentf("%scontent=%s, expected=%s", base64.StdEncoding.EncodeToString(rc), base64.StdEncoding.EncodeToString(content)))

	c.Assert(r.Hash(), Equals, hash) // Test Hash() before close
	c.Assert(r.Close(), IsNil)

}

func (s *SuiteReader) TestReadEmptyObjfile(c *C) {
	source := bytes.NewReader([]byte{})
	_, err := NewReader(source)
	c.Assert(err, NotNil)
}

func (s *SuiteReader) TestReadGarbage(c *C) {
	source := bytes.NewReader([]byte("!@#$RO!@NROSADfinq@o#irn@oirfn"))
	_, err := NewReader(source)
	c.Assert(err, NotNil)
}

func (s *SuiteReader) TestReadCorruptZLib(c *C) {
	data, _ := base64.StdEncoding.DecodeString("eAFLysaalPUjBgAAAJsAHw")
	source := bytes.NewReader(data)
	r, err := NewReader(source)
	c.Assert(err, IsNil)

	_, _, err = r.Header()
	c.Assert(err, NotNil)
}
