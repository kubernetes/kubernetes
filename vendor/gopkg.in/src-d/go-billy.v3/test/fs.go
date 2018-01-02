package test

import (
	"os"

	. "gopkg.in/check.v1"
	. "gopkg.in/src-d/go-billy.v3"
	"gopkg.in/src-d/go-billy.v3/util"
)

// FilesystemSuite is a convenient test suite to validate any implementation of
// billy.Filesystem
type FilesystemSuite struct {
	FS Filesystem

	BasicSuite
	DirSuite
	SymlinkSuite
	TempFileSuite
	ChrootSuite
}

// NewFilesystemSuite returns a new FilesystemSuite based on the given fs.
func NewFilesystemSuite(fs Filesystem) FilesystemSuite {
	s := FilesystemSuite{FS: fs}
	s.BasicSuite.FS = s.FS
	s.DirSuite.FS = s.FS
	s.SymlinkSuite.FS = s.FS
	s.TempFileSuite.FS = s.FS
	s.ChrootSuite.FS = s.FS

	return s
}

func (s *FilesystemSuite) TestSymlinkToDir(c *C) {
	err := s.FS.MkdirAll("dir", 0755)
	c.Assert(err, IsNil)

	err = s.FS.Symlink("dir", "link")
	c.Assert(err, IsNil)

	fi, err := s.FS.Stat("link")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "link")
	c.Assert(fi.IsDir(), Equals, true)
}

func (s *FilesystemSuite) TestSymlinkReadDir(c *C) {
	err := util.WriteFile(s.FS, "dir/file", []byte("foo"), 0644)
	c.Assert(err, IsNil)

	err = s.FS.Symlink("dir", "link")
	c.Assert(err, IsNil)

	info, err := s.FS.ReadDir("link")
	c.Assert(err, IsNil)
	c.Assert(info, HasLen, 1)

	c.Assert(info[0].Size(), Equals, int64(3))
	c.Assert(info[0].IsDir(), Equals, false)
	c.Assert(info[0].Name(), Equals, "file")
}

func (s *FilesystemSuite) TestCreateWithExistantDir(c *C) {
	err := s.FS.MkdirAll("foo", 0644)
	c.Assert(err, IsNil)

	f, err := s.FS.Create("foo")
	c.Assert(err, NotNil)
	c.Assert(f, IsNil)
}

func (s *ChrootSuite) TestReadDirWithChroot(c *C) {
	files := []string{"foo", "bar", "qux/baz", "qux/qux"}
	for _, name := range files {
		err := util.WriteFile(s.FS, name, nil, 0644)
		c.Assert(err, IsNil)
	}

	qux, _ := s.FS.Chroot("/qux")

	info, err := qux.(Filesystem).ReadDir("/")
	c.Assert(err, IsNil)
	c.Assert(info, HasLen, 2)
}

func (s *FilesystemSuite) TestSymlinkWithChrootBasic(c *C) {
	qux, _ := s.FS.Chroot("/qux")

	err := util.WriteFile(qux, "file", nil, 0644)
	c.Assert(err, IsNil)

	err = qux.(Filesystem).Symlink("file", "link")
	c.Assert(err, IsNil)

	fi, err := qux.Stat("link")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "link")

	fi, err = s.FS.Stat("qux/link")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "link")
}

func (s *FilesystemSuite) TestSymlinkWithChrootCrossBounders(c *C) {
	qux, _ := s.FS.Chroot("/qux")
	util.WriteFile(s.FS, "file", []byte("foo"), customMode)

	err := qux.Symlink("../../file", "qux/link")
	c.Assert(err, Equals, nil)

	fi, err := qux.Stat("qux/link")
	c.Assert(fi, NotNil)
	c.Assert(err, Equals, nil)
}

func (s *FilesystemSuite) TestReadDirWithLink(c *C) {
	util.WriteFile(s.FS, "foo/bar", []byte("foo"), customMode)
	s.FS.Symlink("bar", "foo/qux")

	info, err := s.FS.ReadDir("/foo")
	c.Assert(err, IsNil)
	c.Assert(info, HasLen, 2)
}

func (s *FilesystemSuite) TestRemoveAllNonExistent(c *C) {
	c.Assert(util.RemoveAll(s.FS, "non-existent"), IsNil)
}

func (s *FilesystemSuite) TestRemoveAllEmptyDir(c *C) {
	c.Assert(s.FS.MkdirAll("empty", os.FileMode(0755)), IsNil)
	c.Assert(util.RemoveAll(s.FS, "empty"), IsNil)
	_, err := s.FS.Stat("empty")
	c.Assert(err, NotNil)
	c.Assert(os.IsNotExist(err), Equals, true)
}

func (s *FilesystemSuite) TestRemoveAll(c *C) {
	fnames := []string{
		"foo/1",
		"foo/2",
		"foo/bar/1",
		"foo/bar/2",
		"foo/bar/baz/1",
		"foo/bar/baz/qux/1",
		"foo/bar/baz/qux/2",
		"foo/bar/baz/qux/3",
	}

	for _, fname := range fnames {
		err := util.WriteFile(s.FS, fname, nil, 0644)
		c.Assert(err, IsNil)
	}

	c.Assert(util.RemoveAll(s.FS, "foo"), IsNil)

	for _, fname := range fnames {
		_, err := s.FS.Stat(fname)
		comment := Commentf("not removed: %s %s", fname, err)
		c.Assert(os.IsNotExist(err), Equals, true, comment)
	}
}

func (s *FilesystemSuite) TestRemoveAllRelative(c *C) {
	fnames := []string{
		"foo/1",
		"foo/2",
		"foo/bar/1",
		"foo/bar/2",
		"foo/bar/baz/1",
		"foo/bar/baz/qux/1",
		"foo/bar/baz/qux/2",
		"foo/bar/baz/qux/3",
	}

	for _, fname := range fnames {
		err := util.WriteFile(s.FS, fname, nil, 0644)
		c.Assert(err, IsNil)
	}

	c.Assert(util.RemoveAll(s.FS, "foo/bar/.."), IsNil)

	for _, fname := range fnames {
		_, err := s.FS.Stat(fname)
		comment := Commentf("not removed: %s %s", fname, err)
		c.Assert(os.IsNotExist(err), Equals, true, comment)
	}
}

func (s *FilesystemSuite) TestReadDir(c *C) {
	files := []string{"foo", "bar", "qux/baz", "qux/qux"}
	for _, name := range files {
		err := util.WriteFile(s.FS, name, nil, 0644)
		c.Assert(err, IsNil)
	}

	info, err := s.FS.ReadDir("/")
	c.Assert(err, IsNil)
	c.Assert(info, HasLen, 3)

	info, err = s.FS.ReadDir("/qux")
	c.Assert(err, IsNil)
	c.Assert(info, HasLen, 2)
}
