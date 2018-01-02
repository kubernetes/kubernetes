package test

import (
	"os"

	. "gopkg.in/check.v1"
	. "gopkg.in/src-d/go-billy.v3"
	"gopkg.in/src-d/go-billy.v3/util"
)

// DirSuite is a convenient test suite to validate any implementation of
// billy.Dir
type DirSuite struct {
	FS interface {
		Basic
		Dir
	}
}

func (s *DirSuite) TestMkdirAll(c *C) {
	err := s.FS.MkdirAll("empty", os.FileMode(0755))
	c.Assert(err, IsNil)

	fi, err := s.FS.Stat("empty")
	c.Assert(err, IsNil)
	c.Assert(fi.IsDir(), Equals, true)
}

func (s *DirSuite) TestMkdirAllNested(c *C) {
	err := s.FS.MkdirAll("foo/bar/baz", os.FileMode(0755))
	c.Assert(err, IsNil)

	fi, err := s.FS.Stat("foo/bar/baz")
	c.Assert(err, IsNil)
	c.Assert(fi.IsDir(), Equals, true)

	fi, err = s.FS.Stat("foo/bar")
	c.Assert(err, IsNil)
	c.Assert(fi.IsDir(), Equals, true)

	fi, err = s.FS.Stat("foo")
	c.Assert(err, IsNil)
	c.Assert(fi.IsDir(), Equals, true)
}

func (s *DirSuite) TestMkdirAllIdempotent(c *C) {
	err := s.FS.MkdirAll("empty", 0755)
	c.Assert(err, IsNil)
	fi, err := s.FS.Stat("empty")
	c.Assert(err, IsNil)
	c.Assert(fi.IsDir(), Equals, true)

	// idempotent
	err = s.FS.MkdirAll("empty", 0755)
	c.Assert(err, IsNil)
	fi, err = s.FS.Stat("empty")
	c.Assert(err, IsNil)
	c.Assert(fi.IsDir(), Equals, true)
}

func (s *DirSuite) TestMkdirAllAndCreate(c *C) {
	err := s.FS.MkdirAll("dir", os.FileMode(0755))
	c.Assert(err, IsNil)

	f, err := s.FS.Create("dir/bar/foo")
	c.Assert(err, IsNil)
	c.Assert(f.Close(), IsNil)

	fi, err := s.FS.Stat("dir/bar/foo")
	c.Assert(err, IsNil)
	c.Assert(fi.IsDir(), Equals, false)
}

func (s *DirSuite) TestMkdirAllWithExistingFile(c *C) {
	f, err := s.FS.Create("dir/foo")
	c.Assert(err, IsNil)
	c.Assert(f.Close(), IsNil)

	err = s.FS.MkdirAll("dir/foo", os.FileMode(0755))
	c.Assert(err, NotNil)

	fi, err := s.FS.Stat("dir/foo")
	c.Assert(err, IsNil)
	c.Assert(fi.IsDir(), Equals, false)
}

func (s *DirSuite) TestStatDir(c *C) {
	s.FS.MkdirAll("foo/bar", 0644)

	fi, err := s.FS.Stat("foo/bar")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "bar")
	c.Assert(fi.Mode().IsDir(), Equals, true)
	c.Assert(fi.ModTime().IsZero(), Equals, false)
	c.Assert(fi.IsDir(), Equals, true)
}

func (s *BasicSuite) TestStatDeep(c *C) {
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

	fi, err = s.FS.Stat("qux/baz")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "baz")
	c.Assert(fi.IsDir(), Equals, false)
}

func (s *DirSuite) TestReadDir(c *C) {
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

func (s *DirSuite) TestReadDirWithMkDirAll(c *C) {
	err := s.FS.MkdirAll("qux", 0644)
	c.Assert(err, IsNil)

	files := []string{"qux/baz", "qux/qux"}
	for _, name := range files {
		err := util.WriteFile(s.FS, name, nil, 0644)
		c.Assert(err, IsNil)
	}

	info, err := s.FS.ReadDir("/")
	c.Assert(err, IsNil)
	c.Assert(info, HasLen, 1)
	c.Assert(info[0].IsDir(), Equals, true)

	info, err = s.FS.ReadDir("/qux")
	c.Assert(err, IsNil)
	c.Assert(info, HasLen, 2)
}

func (s *DirSuite) TestReadDirFileInfo(c *C) {
	err := util.WriteFile(s.FS, "foo", []byte{'F', 'O', 'O'}, 0644)
	c.Assert(err, IsNil)

	info, err := s.FS.ReadDir("/")
	c.Assert(err, IsNil)
	c.Assert(info, HasLen, 1)

	c.Assert(info[0].Size(), Equals, int64(3))
	c.Assert(info[0].IsDir(), Equals, false)
	c.Assert(info[0].Name(), Equals, "foo")
}

func (s *DirSuite) TestReadDirFileInfoDirs(c *C) {
	files := []string{"qux/baz/foo"}
	for _, name := range files {
		err := util.WriteFile(s.FS, name, []byte{'F', 'O', 'O'}, 0644)
		c.Assert(err, IsNil)
	}

	info, err := s.FS.ReadDir("qux")
	c.Assert(err, IsNil)
	c.Assert(info, HasLen, 1)
	c.Assert(info[0].IsDir(), Equals, true)
	c.Assert(info[0].Name(), Equals, "baz")

	info, err = s.FS.ReadDir("qux/baz")
	c.Assert(err, IsNil)
	c.Assert(info, HasLen, 1)
	c.Assert(info[0].Size(), Equals, int64(3))
	c.Assert(info[0].IsDir(), Equals, false)
	c.Assert(info[0].Name(), Equals, "foo")
	c.Assert(info[0].Mode(), Not(Equals), 0)
}

func (s *DirSuite) TestRenameToDir(c *C) {
	err := util.WriteFile(s.FS, "foo", nil, 0644)
	c.Assert(err, IsNil)

	err = s.FS.Rename("foo", "bar/qux")
	c.Assert(err, IsNil)

	old, err := s.FS.Stat("foo")
	c.Assert(old, IsNil)
	c.Assert(os.IsNotExist(err), Equals, true)

	dir, err := s.FS.Stat("bar")
	c.Assert(dir, NotNil)
	c.Assert(err, IsNil)

	file, err := s.FS.Stat("bar/qux")
	c.Assert(file.Name(), Equals, "qux")
	c.Assert(err, IsNil)
}

func (s *DirSuite) TestRenameDir(c *C) {
	err := s.FS.MkdirAll("foo", 0644)
	c.Assert(err, IsNil)

	err = util.WriteFile(s.FS, "foo/bar", nil, 0644)
	c.Assert(err, IsNil)

	err = s.FS.Rename("foo", "bar")
	c.Assert(err, IsNil)

	dirfoo, err := s.FS.Stat("foo")
	c.Assert(dirfoo, IsNil)
	c.Assert(os.IsNotExist(err), Equals, true)

	dirbar, err := s.FS.Stat("bar")
	c.Assert(err, IsNil)
	c.Assert(dirbar, NotNil)

	foo, err := s.FS.Stat("foo/bar")
	c.Assert(os.IsNotExist(err), Equals, true)
	c.Assert(foo, IsNil)

	bar, err := s.FS.Stat("bar/bar")
	c.Assert(err, IsNil)
	c.Assert(bar, NotNil)
}
