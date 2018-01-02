package temporal

import (
	"strings"
	"testing"

	"gopkg.in/src-d/go-billy.v3/memfs"
	"gopkg.in/src-d/go-billy.v3/test"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

var _ = Suite(&TemporalSuite{})

type TemporalSuite struct {
	test.TempFileSuite
}

func (s *TemporalSuite) SetUpTest(c *C) {
	s.FS = New(memfs.New(), "foo")
}

func (s *TemporalSuite) TestTempFileDefaultPath(c *C) {
	fs := New(memfs.New(), "foo")
	f, err := fs.TempFile("", "bar")
	c.Assert(err, IsNil)
	c.Assert(f.Close(), IsNil)

	c.Assert(strings.HasPrefix(f.Name(), fs.Join("foo", "bar")), Equals, true)
}
