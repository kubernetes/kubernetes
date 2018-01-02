package packfile

import (
	"strconv"
	"strings"

	"gopkg.in/src-d/go-git.v4/plumbing"

	. "gopkg.in/check.v1"
)

type IndexSuite struct{}

var _ = Suite(&IndexSuite{})

func (s *IndexSuite) TestLookupOffset(c *C) {
	idx := NewIndex(0)

	for o1 := 0; o1 < 10000; o1 += 100 {
		for o2 := 0; o2 < 10000; o2 += 100 {
			if o2 >= o1 {
				e, ok := idx.LookupOffset(uint64(o2))
				c.Assert(ok, Equals, false)
				c.Assert(e, IsNil)
			} else {
				e, ok := idx.LookupOffset(uint64(o2))
				c.Assert(ok, Equals, true)
				c.Assert(e, NotNil)
				c.Assert(e.Hash, Equals, s.toHash(o2))
				c.Assert(e.Offset, Equals, uint64(o2))
			}
		}

		h1 := s.toHash(o1)
		idx.Add(h1, uint64(o1), 0)

		for o2 := 0; o2 < 10000; o2 += 100 {
			if o2 > o1 {
				e, ok := idx.LookupOffset(uint64(o2))
				c.Assert(ok, Equals, false)
				c.Assert(e, IsNil)
			} else {
				e, ok := idx.LookupOffset(uint64(o2))
				c.Assert(ok, Equals, true)
				c.Assert(e, NotNil)
				c.Assert(e.Hash, Equals, s.toHash(o2))
				c.Assert(e.Offset, Equals, uint64(o2))
			}
		}
	}
}

func (s *IndexSuite) TestLookupHash(c *C) {
	idx := NewIndex(0)

	for o1 := 0; o1 < 10000; o1 += 100 {
		for o2 := 0; o2 < 10000; o2 += 100 {
			if o2 >= o1 {
				e, ok := idx.LookupHash(s.toHash(o2))
				c.Assert(ok, Equals, false)
				c.Assert(e, IsNil)
			} else {
				e, ok := idx.LookupHash(s.toHash(o2))
				c.Assert(ok, Equals, true)
				c.Assert(e, NotNil)
				c.Assert(e.Hash, Equals, s.toHash(o2))
				c.Assert(e.Offset, Equals, uint64(o2))
			}
		}

		h1 := s.toHash(o1)
		idx.Add(h1, uint64(o1), 0)

		for o2 := 0; o2 < 10000; o2 += 100 {
			if o2 > o1 {
				e, ok := idx.LookupHash(s.toHash(o2))
				c.Assert(ok, Equals, false)
				c.Assert(e, IsNil)
			} else {
				e, ok := idx.LookupHash(s.toHash(o2))
				c.Assert(ok, Equals, true)
				c.Assert(e, NotNil)
				c.Assert(e.Hash, Equals, s.toHash(o2))
				c.Assert(e.Offset, Equals, uint64(o2))
			}
		}
	}
}

func (s *IndexSuite) TestSize(c *C) {
	idx := NewIndex(0)

	for o1 := 0; o1 < 1000; o1++ {
		c.Assert(idx.Size(), Equals, o1)
		h1 := s.toHash(o1)
		idx.Add(h1, uint64(o1), 0)
	}
}

func (s *IndexSuite) TestIdxFileEmpty(c *C) {
	idx := NewIndex(0)
	idxf := idx.ToIdxFile()
	idx2 := NewIndexFromIdxFile(idxf)
	c.Assert(idx, DeepEquals, idx2)
}

func (s *IndexSuite) TestIdxFile(c *C) {
	idx := NewIndex(0)
	for o1 := 0; o1 < 1000; o1++ {
		h1 := s.toHash(o1)
		idx.Add(h1, uint64(o1), 0)
	}

	idx2 := NewIndexFromIdxFile(idx.ToIdxFile())
	c.Assert(idx, DeepEquals, idx2)
}

func (s *IndexSuite) toHash(i int) plumbing.Hash {
	is := strconv.Itoa(i)
	padding := strings.Repeat("a", 40-len(is))
	return plumbing.NewHash(padding + is)
}
