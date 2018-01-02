package mount

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/src-d/go-billy.v3"
	"gopkg.in/src-d/go-billy.v3/memfs"
	"gopkg.in/src-d/go-billy.v3/test"
	"gopkg.in/src-d/go-billy.v3/util"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

var _ = Suite(&MountSuite{})

type MountSuite struct {
	Helper     *Mount
	Underlying mock
	Source     mock
}

type mock struct {
	test.BasicMock
	test.DirMock
	test.SymlinkMock
}

func (s *MountSuite) SetUpTest(c *C) {
	s.Underlying.BasicMock = test.BasicMock{}
	s.Underlying.DirMock = test.DirMock{}
	s.Underlying.SymlinkMock = test.SymlinkMock{}
	s.Source.BasicMock = test.BasicMock{}
	s.Source.DirMock = test.DirMock{}
	s.Source.SymlinkMock = test.SymlinkMock{}

	s.Helper = New(&s.Underlying, "/foo", &s.Source)
}

func (s *MountSuite) TestCreate(c *C) {
	f, err := s.Helper.Create("bar/qux")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, filepath.Join("bar", "qux"))

	c.Assert(s.Underlying.CreateArgs, HasLen, 1)
	c.Assert(s.Underlying.CreateArgs[0], Equals, filepath.Join("bar", "qux"))
	c.Assert(s.Source.CreateArgs, HasLen, 0)
}

func (s *MountSuite) TestCreateMountPoint(c *C) {
	f, err := s.Helper.Create("foo")
	c.Assert(f, IsNil)
	c.Assert(err, Equals, os.ErrInvalid)
}

func (s *MountSuite) TestCreateInMount(c *C) {
	f, err := s.Helper.Create("foo/bar/qux")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, filepath.Join("foo", "bar", "qux"))

	c.Assert(s.Underlying.CreateArgs, HasLen, 0)
	c.Assert(s.Source.CreateArgs, HasLen, 1)
	c.Assert(s.Source.CreateArgs[0], Equals, filepath.Join("bar", "qux"))
}

func (s *MountSuite) TestOpen(c *C) {
	f, err := s.Helper.Open("bar/qux")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, filepath.Join("bar", "qux"))

	c.Assert(s.Underlying.OpenArgs, HasLen, 1)
	c.Assert(s.Underlying.OpenArgs[0], Equals, filepath.Join("bar", "qux"))
	c.Assert(s.Source.OpenArgs, HasLen, 0)
}

func (s *MountSuite) TestOpenMountPoint(c *C) {
	f, err := s.Helper.Open("foo")
	c.Assert(f, IsNil)
	c.Assert(err, Equals, os.ErrInvalid)
}

func (s *MountSuite) TestOpenInMount(c *C) {
	f, err := s.Helper.Open("foo/bar/qux")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, filepath.Join("foo", "bar", "qux"))

	c.Assert(s.Underlying.OpenArgs, HasLen, 0)
	c.Assert(s.Source.OpenArgs, HasLen, 1)
	c.Assert(s.Source.OpenArgs[0], Equals, filepath.Join("bar", "qux"))
}

func (s *MountSuite) TestOpenFile(c *C) {
	f, err := s.Helper.OpenFile("bar/qux", 42, 0777)
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, filepath.Join("bar", "qux"))

	c.Assert(s.Underlying.OpenFileArgs, HasLen, 1)
	c.Assert(s.Underlying.OpenFileArgs[0], Equals,
		[3]interface{}{filepath.Join("bar", "qux"), 42, os.FileMode(0777)})
	c.Assert(s.Source.OpenFileArgs, HasLen, 0)
}

func (s *MountSuite) TestOpenFileMountPoint(c *C) {
	f, err := s.Helper.OpenFile("foo", 42, 0777)
	c.Assert(f, IsNil)
	c.Assert(err, Equals, os.ErrInvalid)
}

func (s *MountSuite) TestOpenFileInMount(c *C) {
	f, err := s.Helper.OpenFile("foo/bar/qux", 42, 0777)
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, filepath.Join("foo", "bar", "qux"))

	c.Assert(s.Underlying.OpenFileArgs, HasLen, 0)
	c.Assert(s.Source.OpenFileArgs, HasLen, 1)
	c.Assert(s.Source.OpenFileArgs[0], Equals,
		[3]interface{}{filepath.Join("bar", "qux"), 42, os.FileMode(0777)})
}

func (s *MountSuite) TestStat(c *C) {
	_, err := s.Helper.Stat("bar/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.StatArgs, HasLen, 1)
	c.Assert(s.Underlying.StatArgs[0], Equals, filepath.Join("bar", "qux"))
	c.Assert(s.Source.StatArgs, HasLen, 0)
}

func (s *MountSuite) TestStatInMount(c *C) {
	_, err := s.Helper.Stat("foo/bar/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.StatArgs, HasLen, 0)
	c.Assert(s.Source.StatArgs, HasLen, 1)
	c.Assert(s.Source.StatArgs[0], Equals, filepath.Join("bar", "qux"))
}

func (s *MountSuite) TestRename(c *C) {
	err := s.Helper.Rename("bar/qux", "qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.RenameArgs, HasLen, 1)
	c.Assert(s.Underlying.RenameArgs[0], Equals, [2]string{"bar/qux", "qux"})
	c.Assert(s.Source.RenameArgs, HasLen, 0)
}

func (s *MountSuite) TestRenameInMount(c *C) {
	err := s.Helper.Rename("foo/bar/qux", "foo/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.RenameArgs, HasLen, 0)
	c.Assert(s.Source.RenameArgs, HasLen, 1)
	c.Assert(s.Source.RenameArgs[0], Equals,
		[2]string{filepath.Join("bar", "qux"), "qux"})
}

func (s *MountSuite) TestRenameCross(c *C) {
	underlying := memfs.New()
	source := memfs.New()

	util.WriteFile(underlying, "file", []byte("foo"), 0777)

	fs := New(underlying, "/foo", source)
	err := fs.Rename("file", "foo/file")
	c.Assert(err, IsNil)

	_, err = underlying.Stat("file")
	c.Assert(err, Equals, os.ErrNotExist)

	_, err = source.Stat("file")
	c.Assert(err, IsNil)

	err = fs.Rename("foo/file", "file")
	c.Assert(err, IsNil)

	_, err = underlying.Stat("file")
	c.Assert(err, IsNil)

	_, err = source.Stat("file")
	c.Assert(err, Equals, os.ErrNotExist)
}

func (s *MountSuite) TestRemove(c *C) {
	err := s.Helper.Remove("bar/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.RemoveArgs, HasLen, 1)
	c.Assert(s.Underlying.RemoveArgs[0], Equals, filepath.Join("bar", "qux"))
	c.Assert(s.Source.RemoveArgs, HasLen, 0)
}

func (s *MountSuite) TestRemoveMountPoint(c *C) {
	err := s.Helper.Remove("foo")
	c.Assert(err, Equals, os.ErrInvalid)
}

func (s *MountSuite) TestRemoveInMount(c *C) {
	err := s.Helper.Remove("foo/bar/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.RemoveArgs, HasLen, 0)
	c.Assert(s.Source.RemoveArgs, HasLen, 1)
	c.Assert(s.Source.RemoveArgs[0], Equals, filepath.Join("bar", "qux"))
}

func (s *MountSuite) TestReadDir(c *C) {
	_, err := s.Helper.ReadDir("bar/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.ReadDirArgs, HasLen, 1)
	c.Assert(s.Underlying.ReadDirArgs[0], Equals, filepath.Join("bar", "qux"))
	c.Assert(s.Source.ReadDirArgs, HasLen, 0)
}

func (s *MountSuite) TestJoin(c *C) {
	s.Helper.Join("foo", "bar")

	c.Assert(s.Underlying.JoinArgs, HasLen, 1)
	c.Assert(s.Underlying.JoinArgs[0], DeepEquals, []string{"foo", "bar"})
	c.Assert(s.Source.JoinArgs, HasLen, 0)
}

func (s *MountSuite) TestReadDirInMount(c *C) {
	_, err := s.Helper.ReadDir("foo/bar/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.ReadDirArgs, HasLen, 0)
	c.Assert(s.Source.ReadDirArgs, HasLen, 1)
	c.Assert(s.Source.ReadDirArgs[0], Equals, filepath.Join("bar", "qux"))
}

func (s *MountSuite) TestMkdirAll(c *C) {
	err := s.Helper.MkdirAll("bar/qux", 0777)
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.MkdirAllArgs, HasLen, 1)
	c.Assert(s.Underlying.MkdirAllArgs[0], Equals,
		[2]interface{}{filepath.Join("bar", "qux"), os.FileMode(0777)})
	c.Assert(s.Source.MkdirAllArgs, HasLen, 0)
}

func (s *MountSuite) TestMkdirAllInMount(c *C) {
	err := s.Helper.MkdirAll("foo/bar/qux", 0777)
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.MkdirAllArgs, HasLen, 0)
	c.Assert(s.Source.MkdirAllArgs, HasLen, 1)
	c.Assert(s.Source.MkdirAllArgs[0], Equals,
		[2]interface{}{filepath.Join("bar", "qux"), os.FileMode(0777)})
}

func (s *MountSuite) TestLstat(c *C) {
	_, err := s.Helper.Lstat("bar/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.LstatArgs, HasLen, 1)
	c.Assert(s.Underlying.LstatArgs[0], Equals, filepath.Join("bar", "qux"))
	c.Assert(s.Source.LstatArgs, HasLen, 0)
}

func (s *MountSuite) TestLstatInMount(c *C) {
	_, err := s.Helper.Lstat("foo/bar/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.LstatArgs, HasLen, 0)
	c.Assert(s.Source.LstatArgs, HasLen, 1)
	c.Assert(s.Source.LstatArgs[0], Equals, filepath.Join("bar", "qux"))
}

func (s *MountSuite) TestSymlink(c *C) {
	err := s.Helper.Symlink("../baz", "bar/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.SymlinkArgs, HasLen, 1)
	c.Assert(s.Underlying.SymlinkArgs[0], Equals,
		[2]string{"../baz", filepath.Join("bar", "qux")})
	c.Assert(s.Source.SymlinkArgs, HasLen, 0)
}

func (s *MountSuite) TestSymlinkCrossMount(c *C) {
	err := s.Helper.Symlink("../foo", "bar/qux")
	c.Assert(err, NotNil)

	err = s.Helper.Symlink("../foo/qux", "bar/qux")
	c.Assert(err, NotNil)

	err = s.Helper.Symlink("../baz", "foo")
	c.Assert(err, NotNil)

	err = s.Helper.Symlink("../../../foo", "foo/bar/qux")
	c.Assert(err, NotNil)
}

func (s *MountSuite) TestSymlinkInMount(c *C) {
	err := s.Helper.Symlink("../baz", "foo/bar/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.SymlinkArgs, HasLen, 0)
	c.Assert(s.Source.SymlinkArgs, HasLen, 1)
	c.Assert(s.Source.SymlinkArgs[0], Equals,
		[2]string{"../baz", filepath.Join("bar", "qux")})
}

func (s *MountSuite) TestRadlink(c *C) {
	_, err := s.Helper.Readlink("bar/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.ReadlinkArgs, HasLen, 1)
	c.Assert(s.Underlying.ReadlinkArgs[0], Equals, filepath.Join("bar", "qux"))
	c.Assert(s.Source.ReadlinkArgs, HasLen, 0)
}

func (s *MountSuite) TestReadlinkInMount(c *C) {
	_, err := s.Helper.Readlink("foo/bar/qux")
	c.Assert(err, IsNil)

	c.Assert(s.Underlying.ReadlinkArgs, HasLen, 0)
	c.Assert(s.Source.ReadlinkArgs, HasLen, 1)
	c.Assert(s.Source.ReadlinkArgs[0], Equals, filepath.Join("bar", "qux"))
}

func (s *MountSuite) TestUnderlyingNotSupported(c *C) {
	h := New(&test.BasicMock{}, "/foo", &test.BasicMock{})
	_, err := h.ReadDir("qux")
	c.Assert(err, Equals, billy.ErrNotSupported)
	_, err = h.Readlink("qux")
	c.Assert(err, Equals, billy.ErrNotSupported)
}

func (s *MountSuite) TestSourceNotSupported(c *C) {
	h := New(&s.Underlying, "/foo", &test.BasicMock{})
	_, err := h.ReadDir("foo")
	c.Assert(err, Equals, billy.ErrNotSupported)
	_, err = h.Readlink("foo")
	c.Assert(err, Equals, billy.ErrNotSupported)
}
