package test

import (
	"os"

	. "gopkg.in/check.v1"
	. "gopkg.in/src-d/go-billy.v3"
	"gopkg.in/src-d/go-billy.v3/util"
)

// ChrootSuite is a convenient test suite to validate any implementation of
// billy.Chroot
type ChrootSuite struct {
	FS interface {
		Basic
		Chroot
	}
}

func (s *ChrootSuite) TestCreateWithChroot(c *C) {
	fs, _ := s.FS.Chroot("foo")
	f, err := fs.Create("bar")
	c.Assert(err, IsNil)
	c.Assert(f.Close(), IsNil)
	c.Assert(f.Name(), Equals, "bar")

	f, err = s.FS.Open("foo/bar")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, s.FS.Join("foo", "bar"))
	c.Assert(f.Close(), IsNil)
}

func (s *ChrootSuite) TestOpenWithChroot(c *C) {
	fs, _ := s.FS.Chroot("foo")
	f, err := fs.Create("bar")
	c.Assert(err, IsNil)
	c.Assert(f.Close(), IsNil)
	c.Assert(f.Name(), Equals, "bar")

	f, err = fs.Open("bar")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "bar")
	c.Assert(f.Close(), IsNil)
}

func (s *ChrootSuite) TestOpenOutOffBoundary(c *C) {
	err := util.WriteFile(s.FS, "bar", nil, 0644)
	c.Assert(err, IsNil)

	fs, _ := s.FS.Chroot("foo")
	f, err := fs.Open("../bar")
	c.Assert(err, Equals, ErrCrossedBoundary)
	c.Assert(f, IsNil)
}

func (s *ChrootSuite) TestStatOutOffBoundary(c *C) {
	err := util.WriteFile(s.FS, "bar", nil, 0644)
	c.Assert(err, IsNil)

	fs, _ := s.FS.Chroot("foo")
	f, err := fs.Stat("../bar")
	c.Assert(err, Equals, ErrCrossedBoundary)
	c.Assert(f, IsNil)
}

func (s *ChrootSuite) TestStatWithChroot(c *C) {
	files := []string{"foo", "bar", "qux/baz", "qux/qux"}
	for _, name := range files {
		err := util.WriteFile(s.FS, name, nil, 0644)
		c.Assert(err, IsNil)
	}

	// Some implementations detect directories based on a prefix
	// for all files; it's easy to miss path separator handling there.
	fi, err := s.FS.Stat("qu")
	c.Assert(os.IsNotExist(err), Equals, true, Commentf("error: %s", err))
	c.Assert(fi, IsNil)

	fi, err = s.FS.Stat("qux")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "qux")
	c.Assert(fi.IsDir(), Equals, true)

	qux, _ := s.FS.Chroot("qux")

	fi, err = qux.Stat("baz")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "baz")
	c.Assert(fi.IsDir(), Equals, false)

	fi, err = qux.Stat("/baz")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "baz")
	c.Assert(fi.IsDir(), Equals, false)
}

func (s *ChrootSuite) TestRenameOutOffBoundary(c *C) {
	err := util.WriteFile(s.FS, "foo/foo", nil, 0644)
	c.Assert(err, IsNil)

	err = util.WriteFile(s.FS, "bar", nil, 0644)
	c.Assert(err, IsNil)

	fs, _ := s.FS.Chroot("foo")
	err = fs.Rename("../bar", "foo")
	c.Assert(err, Equals, ErrCrossedBoundary)

	err = fs.Rename("foo", "../bar")
	c.Assert(err, Equals, ErrCrossedBoundary)
}

func (s *ChrootSuite) TestRemoveOutOffBoundary(c *C) {
	err := util.WriteFile(s.FS, "bar", nil, 0644)
	c.Assert(err, IsNil)

	fs, _ := s.FS.Chroot("foo")
	err = fs.Remove("../bar")
	c.Assert(err, Equals, ErrCrossedBoundary)
}

func (s *FilesystemSuite) TestRoot(c *C) {
	c.Assert(s.FS.Root(), Not(Equals), "")
}
