package fsnoder

import (
	"testing"

	"gopkg.in/src-d/go-git.v4/utils/merkletrie/noder"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type FileSuite struct{}

var _ = Suite(&FileSuite{})

var (
	HashOfEmptyFile = []byte{0xcb, 0xf2, 0x9c, 0xe4, 0x84, 0x22, 0x23, 0x25} // fnv64 basis offset
	HashOfContents  = []byte{0xee, 0x7e, 0xf3, 0xd0, 0xc2, 0xb5, 0xef, 0x83} // hash of "contents"
)

func (s *FileSuite) TestNewFileEmpty(c *C) {
	f, err := newFile("name", "")
	c.Assert(err, IsNil)

	c.Assert(f.Hash(), DeepEquals, HashOfEmptyFile)
	c.Assert(f.Name(), Equals, "name")
	c.Assert(f.IsDir(), Equals, false)
	assertChildren(c, f, noder.NoChildren)
	c.Assert(f.String(), Equals, "name<>")
}

func (s *FileSuite) TestNewFileWithContents(c *C) {
	f, err := newFile("name", "contents")
	c.Assert(err, IsNil)

	c.Assert(f.Hash(), DeepEquals, HashOfContents)
	c.Assert(f.Name(), Equals, "name")
	c.Assert(f.IsDir(), Equals, false)
	assertChildren(c, f, noder.NoChildren)
	c.Assert(f.String(), Equals, "name<contents>")
}

func (s *FileSuite) TestNewfileErrorEmptyName(c *C) {
	_, err := newFile("", "contents")
	c.Assert(err, Not(IsNil))
}

func (s *FileSuite) TestDifferentContentsHaveDifferentHash(c *C) {
	f1, err := newFile("name", "contents")
	c.Assert(err, IsNil)

	f2, err := newFile("name", "foo")
	c.Assert(err, IsNil)

	c.Assert(f1.Hash(), Not(DeepEquals), f2.Hash())
}

func (s *FileSuite) TestSameContentsHaveSameHash(c *C) {
	f1, err := newFile("name1", "contents")
	c.Assert(err, IsNil)

	f2, err := newFile("name2", "contents")
	c.Assert(err, IsNil)

	c.Assert(f1.Hash(), DeepEquals, f2.Hash())
}
