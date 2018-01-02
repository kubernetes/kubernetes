package idxfile_test

import (
	"bytes"
	"io/ioutil"

	"github.com/src-d/go-git-fixtures"
	"gopkg.in/src-d/go-git.v4/plumbing"
	. "gopkg.in/src-d/go-git.v4/plumbing/format/idxfile"

	. "gopkg.in/check.v1"
)

func (s *IdxfileSuite) TestEncode(c *C) {
	expected := &Idxfile{}
	expected.Add(plumbing.NewHash("4bfc730165c370df4a012afbb45ba3f9c332c0d4"), 82, 82)
	expected.Add(plumbing.NewHash("8fa2238efdae08d83c12ee176fae65ff7c99af46"), 42, 42)

	buf := bytes.NewBuffer(nil)
	e := NewEncoder(buf)
	_, err := e.Encode(expected)
	c.Assert(err, IsNil)

	idx := &Idxfile{}
	d := NewDecoder(buf)
	err = d.Decode(idx)
	c.Assert(err, IsNil)

	c.Assert(idx.Entries, DeepEquals, expected.Entries)
}

func (s *IdxfileSuite) TestDecodeEncode(c *C) {
	fixtures.ByTag("packfile").Test(c, func(f *fixtures.Fixture) {
		expected, err := ioutil.ReadAll(f.Idx())
		c.Assert(err, IsNil)

		idx := &Idxfile{}
		d := NewDecoder(bytes.NewBuffer(expected))
		err = d.Decode(idx)
		c.Assert(err, IsNil)

		result := bytes.NewBuffer(nil)
		e := NewEncoder(result)
		size, err := e.Encode(idx)
		c.Assert(err, IsNil)

		c.Assert(size, Equals, len(expected))
		c.Assert(result.Bytes(), DeepEquals, expected)
	})
}
