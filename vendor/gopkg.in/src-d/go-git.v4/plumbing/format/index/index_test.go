package index

import (
	. "gopkg.in/check.v1"
)

func (s *IndexSuite) TestIndexEntry(c *C) {
	idx := &Index{
		Entries: []*Entry{
			{Name: "foo", Size: 42},
			{Name: "bar", Size: 82},
		},
	}

	e, err := idx.Entry("foo")
	c.Assert(err, IsNil)
	c.Assert(e.Name, Equals, "foo")

	e, err = idx.Entry("missing")
	c.Assert(e, IsNil)
	c.Assert(err, Equals, ErrEntryNotFound)
}

func (s *IndexSuite) TestIndexRemove(c *C) {
	idx := &Index{
		Entries: []*Entry{
			{Name: "foo", Size: 42},
			{Name: "bar", Size: 82},
		},
	}

	e, err := idx.Remove("foo")
	c.Assert(err, IsNil)
	c.Assert(e.Name, Equals, "foo")

	e, err = idx.Remove("foo")
	c.Assert(e, IsNil)
	c.Assert(err, Equals, ErrEntryNotFound)
}
