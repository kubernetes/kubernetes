package gitignore

import (
	"os"

	"gopkg.in/src-d/go-billy.v3"
	"gopkg.in/src-d/go-billy.v3/memfs"

	. "gopkg.in/check.v1"
)

type MatcherSuite struct {
	FS billy.Filesystem
}

var _ = Suite(&MatcherSuite{})

func (s *MatcherSuite) SetUpTest(c *C) {
	fs := memfs.New()
	f, err := fs.Create(".gitignore")
	c.Assert(err, IsNil)
	_, err = f.Write([]byte("vendor/g*/\n"))
	c.Assert(err, IsNil)
	err = f.Close()
	c.Assert(err, IsNil)

	err = fs.MkdirAll("vendor", os.ModePerm)
	c.Assert(err, IsNil)
	f, err = fs.Create("vendor/.gitignore")
	c.Assert(err, IsNil)
	_, err = f.Write([]byte("!github.com/\n"))
	c.Assert(err, IsNil)
	err = f.Close()
	c.Assert(err, IsNil)

	fs.MkdirAll("another", os.ModePerm)
	fs.MkdirAll("vendor/github.com", os.ModePerm)
	fs.MkdirAll("vendor/gopkg.in", os.ModePerm)

	s.FS = fs
}

func (s *MatcherSuite) TestDir_ReadPatterns(c *C) {
	ps, err := ReadPatterns(s.FS, nil)
	c.Assert(err, IsNil)
	c.Assert(ps, HasLen, 2)

	m := NewMatcher(ps)
	c.Assert(m.Match([]string{"vendor", "gopkg.in"}, true), Equals, true)
	c.Assert(m.Match([]string{"vendor", "github.com"}, true), Equals, false)
}
