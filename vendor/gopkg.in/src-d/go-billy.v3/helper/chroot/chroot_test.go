package chroot

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/src-d/go-billy.v3"
	"gopkg.in/src-d/go-billy.v3/test"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

var _ = Suite(&ChrootSuite{})

type ChrootSuite struct{}

func (s *ChrootSuite) TestCreate(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	f, err := fs.Create("bar/qux")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, filepath.Join("bar", "qux"))

	c.Assert(m.CreateArgs, HasLen, 1)
	c.Assert(m.CreateArgs[0], Equals, "/foo/bar/qux")
}

func (s *ChrootSuite) TestCreateErrCrossedBoundary(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	_, err := fs.Create("../foo")
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestOpen(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	f, err := fs.Open("bar/qux")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, filepath.Join("bar", "qux"))

	c.Assert(m.OpenArgs, HasLen, 1)
	c.Assert(m.OpenArgs[0], Equals, "/foo/bar/qux")
}

func (s *ChrootSuite) TestChroot(c *C) {
	m := &test.BasicMock{}

	fs, _ := New(m, "/foo").Chroot("baz")
	f, err := fs.Open("bar/qux")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, filepath.Join("bar", "qux"))

	c.Assert(m.OpenArgs, HasLen, 1)
	c.Assert(m.OpenArgs[0], Equals, "/foo/baz/bar/qux")
}

func (s *ChrootSuite) TestChrootErrCrossedBoundary(c *C) {
	m := &test.BasicMock{}

	fs, err := New(m, "/foo").Chroot("../qux")
	c.Assert(fs, IsNil)
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestOpenErrCrossedBoundary(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	_, err := fs.Open("../foo")
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestOpenFile(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	f, err := fs.OpenFile("bar/qux", 42, 0777)
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, filepath.Join("bar", "qux"))

	c.Assert(m.OpenFileArgs, HasLen, 1)
	c.Assert(m.OpenFileArgs[0], Equals, [3]interface{}{"/foo/bar/qux", 42, os.FileMode(0777)})
}

func (s *ChrootSuite) TestOpenFileErrCrossedBoundary(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	_, err := fs.OpenFile("../foo", 42, 0777)
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestStat(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	_, err := fs.Stat("bar/qux")
	c.Assert(err, IsNil)

	c.Assert(m.StatArgs, HasLen, 1)
	c.Assert(m.StatArgs[0], Equals, "/foo/bar/qux")
}

func (s *ChrootSuite) TestStatErrCrossedBoundary(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	_, err := fs.Stat("../foo")
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestRename(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	err := fs.Rename("bar/qux", "qux/bar")
	c.Assert(err, IsNil)

	c.Assert(m.RenameArgs, HasLen, 1)
	c.Assert(m.RenameArgs[0], Equals, [2]string{"/foo/bar/qux", "/foo/qux/bar"})
}

func (s *ChrootSuite) TestRenameErrCrossedBoundary(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	err := fs.Rename("../foo", "bar")
	c.Assert(err, Equals, billy.ErrCrossedBoundary)

	err = fs.Rename("foo", "../bar")
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestRemove(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	err := fs.Remove("bar/qux")
	c.Assert(err, IsNil)

	c.Assert(m.RemoveArgs, HasLen, 1)
	c.Assert(m.RemoveArgs[0], Equals, "/foo/bar/qux")
}

func (s *ChrootSuite) TestRemoveErrCrossedBoundary(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	err := fs.Remove("../foo")
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestTempFile(c *C) {
	m := &test.TempFileMock{}

	fs := New(m, "/foo")
	_, err := fs.TempFile("bar", "qux")
	c.Assert(err, IsNil)

	c.Assert(m.TempFileArgs, HasLen, 1)
	c.Assert(m.TempFileArgs[0], Equals, [2]string{"/foo/bar", "qux"})
}

func (s *ChrootSuite) TestTempFileErrCrossedBoundary(c *C) {
	m := &test.TempFileMock{}

	fs := New(m, "/foo")
	_, err := fs.TempFile("../foo", "qux")
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestTempFileWithBasic(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	_, err := fs.TempFile("", "")
	c.Assert(err, Equals, billy.ErrNotSupported)
}

func (s *ChrootSuite) TestReadDir(c *C) {
	m := &test.DirMock{}

	fs := New(m, "/foo")
	_, err := fs.ReadDir("bar")
	c.Assert(err, IsNil)

	c.Assert(m.ReadDirArgs, HasLen, 1)
	c.Assert(m.ReadDirArgs[0], Equals, "/foo/bar")
}

func (s *ChrootSuite) TestReadDirErrCrossedBoundary(c *C) {
	m := &test.DirMock{}

	fs := New(m, "/foo")
	_, err := fs.ReadDir("../foo")
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestReadDirWithBasic(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	_, err := fs.ReadDir("")
	c.Assert(err, Equals, billy.ErrNotSupported)
}

func (s *ChrootSuite) TestMkDirAll(c *C) {
	m := &test.DirMock{}

	fs := New(m, "/foo")
	err := fs.MkdirAll("bar", 0777)
	c.Assert(err, IsNil)

	c.Assert(m.MkdirAllArgs, HasLen, 1)
	c.Assert(m.MkdirAllArgs[0], Equals, [2]interface{}{"/foo/bar", os.FileMode(0777)})
}

func (s *ChrootSuite) TestMkdirAllErrCrossedBoundary(c *C) {
	m := &test.DirMock{}

	fs := New(m, "/foo")
	err := fs.MkdirAll("../foo", 0777)
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestMkdirAllWithBasic(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	err := fs.MkdirAll("", 0)
	c.Assert(err, Equals, billy.ErrNotSupported)
}

func (s *ChrootSuite) TestLstat(c *C) {
	m := &test.SymlinkMock{}

	fs := New(m, "/foo")
	_, err := fs.Lstat("qux")
	c.Assert(err, IsNil)

	c.Assert(m.LstatArgs, HasLen, 1)
	c.Assert(m.LstatArgs[0], Equals, "/foo/qux")
}

func (s *ChrootSuite) TestLstatErrCrossedBoundary(c *C) {
	m := &test.SymlinkMock{}

	fs := New(m, "/foo")
	_, err := fs.Lstat("../qux")
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestLstatWithBasic(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	_, err := fs.Lstat("")
	c.Assert(err, Equals, billy.ErrNotSupported)
}

func (s *ChrootSuite) TestSymlink(c *C) {
	m := &test.SymlinkMock{}

	fs := New(m, "/foo")
	err := fs.Symlink("../baz", "qux/bar")
	c.Assert(err, IsNil)

	c.Assert(m.SymlinkArgs, HasLen, 1)
	c.Assert(m.SymlinkArgs[0], Equals, [2]string{filepath.FromSlash("../baz"), "/foo/qux/bar"})
}

func (s *ChrootSuite) TestSymlinkWithAbsoluteTarget(c *C) {
	m := &test.SymlinkMock{}

	fs := New(m, "/foo")
	err := fs.Symlink("/bar", "qux/baz")
	c.Assert(err, IsNil)

	c.Assert(m.SymlinkArgs, HasLen, 1)
	c.Assert(m.SymlinkArgs[0], Equals, [2]string{filepath.FromSlash("/foo/bar"), "/foo/qux/baz"})
}

func (s *ChrootSuite) TestSymlinkErrCrossedBoundary(c *C) {
	m := &test.SymlinkMock{}

	fs := New(m, "/foo")
	err := fs.Symlink("qux", "../foo")
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestSymlinkWithBasic(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	err := fs.Symlink("qux", "bar")
	c.Assert(err, Equals, billy.ErrNotSupported)
}

func (s *ChrootSuite) TestReadlink(c *C) {
	m := &test.SymlinkMock{}

	fs := New(m, "/foo")
	link, err := fs.Readlink("/qux")
	c.Assert(err, IsNil)
	c.Assert(link, Equals, filepath.FromSlash("/qux"))

	c.Assert(m.ReadlinkArgs, HasLen, 1)
	c.Assert(m.ReadlinkArgs[0], Equals, "/foo/qux")
}

func (s *ChrootSuite) TestReadlinkWithRelative(c *C) {
	m := &test.SymlinkMock{}

	fs := New(m, "/foo")
	link, err := fs.Readlink("qux/bar")
	c.Assert(err, IsNil)
	c.Assert(link, Equals, filepath.FromSlash("/qux/bar"))

	c.Assert(m.ReadlinkArgs, HasLen, 1)
	c.Assert(m.ReadlinkArgs[0], Equals, "/foo/qux/bar")
}

func (s *ChrootSuite) TestReadlinkErrCrossedBoundary(c *C) {
	m := &test.SymlinkMock{}

	fs := New(m, "/foo")
	_, err := fs.Readlink("../qux")
	c.Assert(err, Equals, billy.ErrCrossedBoundary)
}

func (s *ChrootSuite) TestReadlinkWithBasic(c *C) {
	m := &test.BasicMock{}

	fs := New(m, "/foo")
	_, err := fs.Readlink("")
	c.Assert(err, Equals, billy.ErrNotSupported)
}
