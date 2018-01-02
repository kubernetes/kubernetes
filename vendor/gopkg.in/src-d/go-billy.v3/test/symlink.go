package test

import (
	"io/ioutil"
	"os"

	. "gopkg.in/check.v1"
	. "gopkg.in/src-d/go-billy.v3"
	"gopkg.in/src-d/go-billy.v3/util"
)

// SymlinkSuite is a convenient test suite to validate any implementation of
// billy.Symlink
type SymlinkSuite struct {
	FS interface {
		Basic
		Symlink
	}
}

func (s *SymlinkSuite) TestSymlink(c *C) {
	err := util.WriteFile(s.FS, "file", nil, 0644)
	c.Assert(err, IsNil)

	err = s.FS.Symlink("file", "link")
	c.Assert(err, IsNil)

	fi, err := s.FS.Stat("link")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "link")
}

func (s *SymlinkSuite) TestSymlinkCrossDirs(c *C) {
	err := util.WriteFile(s.FS, "foo/file", nil, 0644)
	c.Assert(err, IsNil)

	err = s.FS.Symlink("../foo/file", "bar/link")
	c.Assert(err, IsNil)

	fi, err := s.FS.Stat("bar/link")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "link")
}

func (s *SymlinkSuite) TestSymlinkNested(c *C) {
	err := util.WriteFile(s.FS, "file", []byte("hello world!"), 0644)
	c.Assert(err, IsNil)

	err = s.FS.Symlink("file", "linkA")
	c.Assert(err, IsNil)

	err = s.FS.Symlink("linkA", "linkB")
	c.Assert(err, IsNil)

	fi, err := s.FS.Stat("linkB")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "linkB")
	c.Assert(fi.Size(), Equals, int64(12))
}

func (s *SymlinkSuite) TestSymlinkWithNonExistentdTarget(c *C) {
	err := s.FS.Symlink("file", "link")
	c.Assert(err, IsNil)

	_, err = s.FS.Stat("link")
	c.Assert(os.IsNotExist(err), Equals, true)
}

func (s *SymlinkSuite) TestSymlinkWithExistingLink(c *C) {
	err := util.WriteFile(s.FS, "link", nil, 0644)
	c.Assert(err, IsNil)

	err = s.FS.Symlink("file", "link")
	c.Assert(err, Not(IsNil))
}

func (s *SymlinkSuite) TestOpenWithSymlinkToRelativePath(c *C) {
	err := util.WriteFile(s.FS, "dir/file", []byte("foo"), 0644)
	c.Assert(err, IsNil)

	err = s.FS.Symlink("file", "dir/link")
	c.Assert(err, IsNil)

	f, err := s.FS.Open("dir/link")
	c.Assert(err, IsNil)

	all, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)
	c.Assert(string(all), Equals, "foo")
	c.Assert(f.Close(), IsNil)
}

func (s *SymlinkSuite) TestOpenWithSymlinkToAbsolutePath(c *C) {
	err := util.WriteFile(s.FS, "dir/file", []byte("foo"), 0644)
	c.Assert(err, IsNil)

	err = s.FS.Symlink("/dir/file", "dir/link")
	c.Assert(err, IsNil)

	f, err := s.FS.Open("dir/link")
	c.Assert(err, IsNil)

	all, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)
	c.Assert(string(all), Equals, "foo")
	c.Assert(f.Close(), IsNil)
}

func (s *SymlinkSuite) TestReadlink(c *C) {
	err := util.WriteFile(s.FS, "file", nil, 0644)
	c.Assert(err, IsNil)

	_, err = s.FS.Readlink("file")
	c.Assert(err, Not(IsNil))
}

func (s *SymlinkSuite) TestReadlinkWithRelativePath(c *C) {
	err := util.WriteFile(s.FS, "dir/file", nil, 0644)
	c.Assert(err, IsNil)

	err = s.FS.Symlink("file", "dir/link")
	c.Assert(err, IsNil)

	oldname, err := s.FS.Readlink("dir/link")
	c.Assert(err, IsNil)
	c.Assert(oldname, Equals, "file")
}

func (s *SymlinkSuite) TestReadlinkWithAbsolutePath(c *C) {
	err := util.WriteFile(s.FS, "dir/file", nil, 0644)
	c.Assert(err, IsNil)

	err = s.FS.Symlink("/dir/file", "dir/link")
	c.Assert(err, IsNil)

	oldname, err := s.FS.Readlink("dir/link")
	c.Assert(err, IsNil)
	c.Assert(oldname, Equals, expectedSymlinkTarget)
}

func (s *SymlinkSuite) TestReadlinkWithNonExistentTarget(c *C) {
	err := s.FS.Symlink("file", "link")
	c.Assert(err, IsNil)

	oldname, err := s.FS.Readlink("link")
	c.Assert(err, IsNil)
	c.Assert(oldname, Equals, "file")
}

func (s *SymlinkSuite) TestReadlinkWithNonExistentLink(c *C) {
	_, err := s.FS.Readlink("link")
	c.Assert(os.IsNotExist(err), Equals, true)
}

func (s *SymlinkSuite) TestStatLink(c *C) {
	util.WriteFile(s.FS, "foo/bar", []byte("foo"), customMode)
	s.FS.Symlink("bar", "foo/qux")

	fi, err := s.FS.Stat("foo/qux")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "qux")
	c.Assert(fi.Size(), Equals, int64(3))
	c.Assert(fi.Mode(), Equals, customMode)
	c.Assert(fi.ModTime().IsZero(), Equals, false)
	c.Assert(fi.IsDir(), Equals, false)
}

func (s *SymlinkSuite) TestLstat(c *C) {
	util.WriteFile(s.FS, "foo/bar", []byte("foo"), customMode)

	fi, err := s.FS.Lstat("foo/bar")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "bar")
	c.Assert(fi.Size(), Equals, int64(3))
	c.Assert(fi.Mode()&os.ModeSymlink != 0, Equals, false)
	c.Assert(fi.ModTime().IsZero(), Equals, false)
	c.Assert(fi.IsDir(), Equals, false)
}

func (s *SymlinkSuite) TestLstatLink(c *C) {
	util.WriteFile(s.FS, "foo/bar", []byte("fosddddaaao"), customMode)
	s.FS.Symlink("bar", "foo/qux")

	fi, err := s.FS.Lstat("foo/qux")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "qux")
	c.Assert(fi.Mode()&os.ModeSymlink != 0, Equals, true)
	c.Assert(fi.ModTime().IsZero(), Equals, false)
	c.Assert(fi.IsDir(), Equals, false)
}

func (s *SymlinkSuite) TestRenameWithSymlink(c *C) {
	err := s.FS.Symlink("file", "link")
	c.Assert(err, IsNil)

	err = s.FS.Rename("link", "newlink")
	c.Assert(err, IsNil)

	_, err = s.FS.Readlink("newlink")
	c.Assert(err, IsNil)
}

func (s *SymlinkSuite) TestRemoveWithSymlink(c *C) {
	err := util.WriteFile(s.FS, "file", []byte("foo"), 0644)
	c.Assert(err, IsNil)

	err = s.FS.Symlink("file", "link")
	c.Assert(err, IsNil)

	err = s.FS.Remove("link")
	c.Assert(err, IsNil)

	_, err = s.FS.Readlink("link")
	c.Assert(os.IsNotExist(err), Equals, true)

	_, err = s.FS.Stat("link")
	c.Assert(os.IsNotExist(err), Equals, true)

	_, err = s.FS.Stat("file")
	c.Assert(err, IsNil)
}
