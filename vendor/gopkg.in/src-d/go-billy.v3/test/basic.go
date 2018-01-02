package test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"

	. "gopkg.in/check.v1"
	. "gopkg.in/src-d/go-billy.v3"
	"gopkg.in/src-d/go-billy.v3/util"
)

// BasicSuite is a convenient test suite to validate any implementation of
// billy.Basic
type BasicSuite struct {
	FS Basic
}

func (s *BasicSuite) TestCreate(c *C) {
	f, err := s.FS.Create("foo")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "foo")
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestCreateDepth(c *C) {
	f, err := s.FS.Create("bar/foo")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, s.FS.Join("bar", "foo"))
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestCreateDepthAbsolute(c *C) {
	f, err := s.FS.Create("/bar/foo")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, s.FS.Join("bar", "foo"))
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestCreateOverwrite(c *C) {
	for i := 0; i < 3; i++ {
		f, err := s.FS.Create("foo")
		c.Assert(err, IsNil)

		l, err := f.Write([]byte(fmt.Sprintf("foo%d", i)))
		c.Assert(err, IsNil)
		c.Assert(l, Equals, 4)

		err = f.Close()
		c.Assert(err, IsNil)
	}

	f, err := s.FS.Open("foo")
	c.Assert(err, IsNil)

	wrote, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)
	c.Assert(string(wrote), DeepEquals, "foo2")
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestCreateAndClose(c *C) {
	f, err := s.FS.Create("foo")
	c.Assert(err, IsNil)

	_, err = f.Write([]byte("foo"))
	c.Assert(err, IsNil)
	c.Assert(f.Close(), IsNil)

	f, err = s.FS.Open(f.Name())
	c.Assert(err, IsNil)

	wrote, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)
	c.Assert(string(wrote), DeepEquals, "foo")
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestOpen(c *C) {
	f, err := s.FS.Create("foo")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "foo")
	c.Assert(f.Close(), IsNil)

	f, err = s.FS.Open("foo")
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "foo")
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestOpenNotExists(c *C) {
	f, err := s.FS.Open("not-exists")
	c.Assert(err, NotNil)
	c.Assert(f, IsNil)
}

func (s *BasicSuite) TestOpenFile(c *C) {
	defaultMode := os.FileMode(0666)

	f, err := s.FS.OpenFile("foo1", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, defaultMode)
	c.Assert(err, IsNil)
	s.testWriteClose(c, f, "foo1")

	// Truncate if it exists
	f, err = s.FS.OpenFile("foo1", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, defaultMode)
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "foo1")
	s.testWriteClose(c, f, "foo1overwritten")

	// Read-only if it exists
	f, err = s.FS.OpenFile("foo1", os.O_RDONLY, defaultMode)
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "foo1")
	s.testReadClose(c, f, "foo1overwritten")

	// Create when it does exist
	f, err = s.FS.OpenFile("foo1", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, defaultMode)
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "foo1")
	s.testWriteClose(c, f, "bar")

	f, err = s.FS.OpenFile("foo1", os.O_RDONLY, defaultMode)
	c.Assert(err, IsNil)
	s.testReadClose(c, f, "bar")
}

func (s *BasicSuite) TestOpenFileNoTruncate(c *C) {
	defaultMode := os.FileMode(0666)

	// Create when it does not exist
	f, err := s.FS.OpenFile("foo1", os.O_CREATE|os.O_WRONLY, defaultMode)
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "foo1")
	s.testWriteClose(c, f, "foo1")

	f, err = s.FS.OpenFile("foo1", os.O_RDONLY, defaultMode)
	c.Assert(err, IsNil)
	s.testReadClose(c, f, "foo1")

	// Create when it does exist
	f, err = s.FS.OpenFile("foo1", os.O_CREATE|os.O_WRONLY, defaultMode)
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "foo1")
	s.testWriteClose(c, f, "bar")

	f, err = s.FS.OpenFile("foo1", os.O_RDONLY, defaultMode)
	c.Assert(err, IsNil)
	s.testReadClose(c, f, "bar1")
}

func (s *BasicSuite) TestOpenFileAppend(c *C) {
	defaultMode := os.FileMode(0666)

	f, err := s.FS.OpenFile("foo1", os.O_CREATE|os.O_WRONLY|os.O_APPEND, defaultMode)
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "foo1")
	s.testWriteClose(c, f, "foo1")

	f, err = s.FS.OpenFile("foo1", os.O_WRONLY|os.O_APPEND, defaultMode)
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "foo1")
	s.testWriteClose(c, f, "bar1")

	f, err = s.FS.OpenFile("foo1", os.O_RDONLY, defaultMode)
	c.Assert(err, IsNil)
	s.testReadClose(c, f, "foo1bar1")
}

func (s *BasicSuite) TestOpenFileReadWrite(c *C) {
	defaultMode := os.FileMode(0666)

	f, err := s.FS.OpenFile("foo1", os.O_CREATE|os.O_TRUNC|os.O_RDWR, defaultMode)
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "foo1")

	written, err := f.Write([]byte("foobar"))
	c.Assert(written, Equals, 6)
	c.Assert(err, IsNil)

	_, err = f.Seek(0, os.SEEK_SET)
	c.Assert(err, IsNil)

	written, err = f.Write([]byte("qux"))
	c.Assert(written, Equals, 3)
	c.Assert(err, IsNil)

	_, err = f.Seek(0, os.SEEK_SET)
	c.Assert(err, IsNil)

	s.testReadClose(c, f, "quxbar")
}

func (s *BasicSuite) TestOpenFileWithModes(c *C) {
	f, err := s.FS.OpenFile("foo", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, customMode)
	c.Assert(err, IsNil)
	c.Assert(f.Close(), IsNil)

	fi, err := s.FS.Stat("foo")
	c.Assert(err, IsNil)
	c.Assert(fi.Mode(), Equals, os.FileMode(customMode))
}

func (s *BasicSuite) testWriteClose(c *C, f File, content string) {
	written, err := f.Write([]byte(content))
	c.Assert(written, Equals, len(content))
	c.Assert(err, IsNil)
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) testReadClose(c *C, f File, content string) {
	read, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)
	c.Assert(string(read), Equals, content)
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestFileWrite(c *C) {
	f, err := s.FS.Create("foo")
	c.Assert(err, IsNil)

	n, err := f.Write([]byte("foo"))
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 3)

	f.Seek(0, io.SeekStart)
	all, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)
	c.Assert(string(all), Equals, "foo")
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestFileWriteClose(c *C) {
	f, err := s.FS.Create("foo")
	c.Assert(err, IsNil)

	c.Assert(f.Close(), IsNil)

	_, err = f.Write([]byte("foo"))
	c.Assert(err, NotNil)
}

func (s *BasicSuite) TestFileRead(c *C) {
	err := util.WriteFile(s.FS, "foo", []byte("foo"), 0644)
	c.Assert(err, IsNil)

	f, err := s.FS.Open("foo")
	c.Assert(err, IsNil)

	all, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)
	c.Assert(string(all), Equals, "foo")
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestFileClosed(c *C) {
	err := util.WriteFile(s.FS, "foo", []byte("foo"), 0644)
	c.Assert(err, IsNil)

	f, err := s.FS.Open("foo")
	c.Assert(err, IsNil)
	c.Assert(f.Close(), IsNil)

	_, err = ioutil.ReadAll(f)
	c.Assert(err, NotNil)
}

func (s *BasicSuite) TestFileNonRead(c *C) {
	err := util.WriteFile(s.FS, "foo", []byte("foo"), 0644)
	c.Assert(err, IsNil)

	f, err := s.FS.OpenFile("foo", os.O_WRONLY, 0)
	c.Assert(err, IsNil)

	_, err = ioutil.ReadAll(f)
	c.Assert(err, NotNil)

	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestFileSeekstart(c *C) {
	s.testFileSeek(c, 10, io.SeekStart)
}

func (s *BasicSuite) TestFileSeekCurrent(c *C) {
	s.testFileSeek(c, 5, io.SeekCurrent)
}

func (s *BasicSuite) TestFileSeekEnd(c *C) {
	s.testFileSeek(c, -26, io.SeekEnd)
}

func (s *BasicSuite) testFileSeek(c *C, offset int64, whence int) {
	err := util.WriteFile(s.FS, "foo", []byte("0123456789abcdefghijklmnopqrstuvwxyz"), 0644)
	c.Assert(err, IsNil)

	f, err := s.FS.Open("foo")
	c.Assert(err, IsNil)

	some := make([]byte, 5)
	_, err = f.Read(some)
	c.Assert(err, IsNil)
	c.Assert(string(some), Equals, "01234")

	p, err := f.Seek(offset, whence)
	c.Assert(err, IsNil)
	c.Assert(int(p), Equals, 10)

	all, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)
	c.Assert(all, HasLen, 26)
	c.Assert(string(all), Equals, "abcdefghijklmnopqrstuvwxyz")
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestSeekToEndAndWrite(c *C) {
	defaultMode := os.FileMode(0666)

	f, err := s.FS.OpenFile("foo1", os.O_CREATE|os.O_TRUNC|os.O_RDWR, defaultMode)
	c.Assert(err, IsNil)
	c.Assert(f.Name(), Equals, "foo1")

	_, err = f.Seek(10, io.SeekEnd)
	c.Assert(err, IsNil)

	n, err := f.Write([]byte(`TEST`))
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 4)

	_, err = f.Seek(0, io.SeekStart)
	c.Assert(err, IsNil)

	s.testReadClose(c, f, "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00TEST")
}

func (s *BasicSuite) TestFileSeekClosed(c *C) {
	err := util.WriteFile(s.FS, "foo", []byte("foo"), 0644)
	c.Assert(err, IsNil)

	f, err := s.FS.Open("foo")
	c.Assert(err, IsNil)
	c.Assert(f.Close(), IsNil)

	_, err = f.Seek(0, 0)
	c.Assert(err, NotNil)
}

func (s *BasicSuite) TestFileCloseTwice(c *C) {
	f, err := s.FS.Create("foo")
	c.Assert(err, IsNil)

	c.Assert(f.Close(), IsNil)
	c.Assert(f.Close(), NotNil)
}

func (s *BasicSuite) TestStat(c *C) {
	util.WriteFile(s.FS, "foo/bar", []byte("foo"), customMode)

	fi, err := s.FS.Stat("foo/bar")
	c.Assert(err, IsNil)
	c.Assert(fi.Name(), Equals, "bar")
	c.Assert(fi.Size(), Equals, int64(3))
	c.Assert(fi.Mode(), Equals, customMode)
	c.Assert(fi.ModTime().IsZero(), Equals, false)
	c.Assert(fi.IsDir(), Equals, false)
}

func (s *BasicSuite) TestStatNonExistent(c *C) {
	fi, err := s.FS.Stat("non-existent")
	comment := Commentf("error: %s", err)
	c.Assert(os.IsNotExist(err), Equals, true, comment)
	c.Assert(fi, IsNil)
}

func (s *BasicSuite) TestRename(c *C) {
	err := util.WriteFile(s.FS, "foo", nil, 0644)
	c.Assert(err, IsNil)

	err = s.FS.Rename("foo", "bar")
	c.Assert(err, IsNil)

	foo, err := s.FS.Stat("foo")
	c.Assert(foo, IsNil)
	c.Assert(os.IsNotExist(err), Equals, true)

	bar, err := s.FS.Stat("bar")
	c.Assert(err, IsNil)
	c.Assert(bar, NotNil)
}

func (s *BasicSuite) TestOpenAndWrite(c *C) {
	err := util.WriteFile(s.FS, "foo", nil, 0644)
	c.Assert(err, IsNil)

	foo, err := s.FS.Open("foo")
	c.Assert(foo, NotNil)
	c.Assert(err, IsNil)

	n, err := foo.Write([]byte("foo"))
	c.Assert(err, NotNil)
	c.Assert(n, Equals, 0)

	c.Assert(foo.Close(), IsNil)
}

func (s *BasicSuite) TestOpenAndStat(c *C) {
	err := util.WriteFile(s.FS, "foo", []byte("foo"), 0644)
	c.Assert(err, IsNil)

	foo, err := s.FS.Open("foo")
	c.Assert(foo, NotNil)
	c.Assert(foo.Name(), Equals, "foo")
	c.Assert(err, IsNil)
	c.Assert(foo.Close(), IsNil)

	stat, err := s.FS.Stat("foo")
	c.Assert(stat, NotNil)
	c.Assert(err, IsNil)
	c.Assert(stat.Name(), Equals, "foo")
	c.Assert(stat.Size(), Equals, int64(3))
}

func (s *BasicSuite) TestRemove(c *C) {
	f, err := s.FS.Create("foo")
	c.Assert(err, IsNil)
	c.Assert(f.Close(), IsNil)

	err = s.FS.Remove("foo")
	c.Assert(err, IsNil)
}

func (s *BasicSuite) TestRemoveNonExisting(c *C) {
	err := s.FS.Remove("NON-EXISTING")
	c.Assert(err, NotNil)
	c.Assert(os.IsNotExist(err), Equals, true)
}

func (s *BasicSuite) TestRemoveNotEmptyDir(c *C) {
	err := util.WriteFile(s.FS, "foo", nil, 0644)
	c.Assert(err, IsNil)

	err = s.FS.Remove("no-exists")
	c.Assert(err, NotNil)
}

func (s *BasicSuite) TestJoin(c *C) {
	c.Assert(s.FS.Join("foo", "bar"), Equals, fmt.Sprintf("foo%cbar", filepath.Separator))
}

func (s *BasicSuite) TestReadAtOnReadWrite(c *C) {
	f, err := s.FS.Create("foo")
	c.Assert(err, IsNil)
	_, err = f.Write([]byte("abcdefg"))
	c.Assert(err, IsNil)

	rf, ok := f.(io.ReaderAt)
	c.Assert(ok, Equals, true)

	b := make([]byte, 3)
	n, err := rf.ReadAt(b, 2)
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 3)
	c.Assert(string(b), Equals, "cde")
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestReadAtOnReadOnly(c *C) {
	err := util.WriteFile(s.FS, "foo", []byte("abcdefg"), 0644)
	c.Assert(err, IsNil)

	f, err := s.FS.Open("foo")
	c.Assert(err, IsNil)

	rf, ok := f.(io.ReaderAt)
	c.Assert(ok, Equals, true)

	b := make([]byte, 3)
	n, err := rf.ReadAt(b, 2)
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 3)
	c.Assert(string(b), Equals, "cde")
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestReadAtEOF(c *C) {
	err := util.WriteFile(s.FS, "foo", []byte("TEST"), 0644)
	c.Assert(err, IsNil)

	f, err := s.FS.Open("foo")
	c.Assert(err, IsNil)

	b := make([]byte, 5)
	n, err := f.ReadAt(b, 0)
	c.Assert(err, Equals, io.EOF)
	c.Assert(n, Equals, 4)
	c.Assert(string(b), Equals, "TEST\x00")

	err = f.Close()
	c.Assert(err, IsNil)
}

func (s *BasicSuite) TestReadAtOffset(c *C) {
	err := util.WriteFile(s.FS, "foo", []byte("TEST"), 0644)
	c.Assert(err, IsNil)

	f, err := s.FS.Open("foo")
	c.Assert(err, IsNil)

	rf, ok := f.(io.ReaderAt)
	c.Assert(ok, Equals, true)

	o, err := f.Seek(0, io.SeekCurrent)
	c.Assert(err, IsNil)
	c.Assert(o, Equals, int64(0))

	b := make([]byte, 4)
	n, err := rf.ReadAt(b, 0)
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 4)
	c.Assert(string(b), Equals, "TEST")

	o, err = f.Seek(0, io.SeekCurrent)
	c.Assert(err, IsNil)
	c.Assert(o, Equals, int64(0))

	err = f.Close()
	c.Assert(err, IsNil)
}

func (s *BasicSuite) TestReadWriteLargeFile(c *C) {
	f, err := s.FS.Create("foo")
	c.Assert(err, IsNil)

	size := 1 << 20

	n, err := f.Write(bytes.Repeat([]byte("F"), size))
	c.Assert(err, IsNil)
	c.Assert(n, Equals, size)

	c.Assert(f.Close(), IsNil)

	f, err = s.FS.Open("foo")
	c.Assert(err, IsNil)
	b, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)
	c.Assert(len(b), Equals, size)
	c.Assert(f.Close(), IsNil)
}

func (s *BasicSuite) TestWriteFile(c *C) {
	err := util.WriteFile(s.FS, "foo", []byte("bar"), 0777)
	c.Assert(err, IsNil)

	f, err := s.FS.Open("foo")
	c.Assert(err, IsNil)

	wrote, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)
	c.Assert(string(wrote), DeepEquals, "bar")

	c.Assert(f.Close(), IsNil)
}
