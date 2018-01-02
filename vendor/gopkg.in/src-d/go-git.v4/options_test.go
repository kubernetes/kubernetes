package git

import (
	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-git.v4/plumbing/object"
)

type OptionsSuite struct {
	BaseSuite
}

var _ = Suite(&OptionsSuite{})

func (s *OptionsSuite) TestCommitOptionsParentsFromHEAD(c *C) {
	o := CommitOptions{Author: &object.Signature{}}
	err := o.Validate(s.Repository)
	c.Assert(err, IsNil)
	c.Assert(o.Parents, HasLen, 1)
}

func (s *OptionsSuite) TestCommitOptionsMissingAuthor(c *C) {
	o := CommitOptions{}
	err := o.Validate(s.Repository)
	c.Assert(err, Equals, ErrMissingAuthor)
}

func (s *OptionsSuite) TestCommitOptionsCommitter(c *C) {
	sig := &object.Signature{}

	o := CommitOptions{Author: sig}
	err := o.Validate(s.Repository)
	c.Assert(err, IsNil)

	c.Assert(o.Committer, Equals, o.Author)
}
