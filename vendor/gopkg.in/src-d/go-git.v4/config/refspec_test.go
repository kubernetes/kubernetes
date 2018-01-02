package config

import (
	"testing"

	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-git.v4/plumbing"
)

type RefSpecSuite struct{}

var _ = Suite(&RefSpecSuite{})

func Test(t *testing.T) { TestingT(t) }

func (s *RefSpecSuite) TestRefSpecIsValid(c *C) {
	spec := RefSpec("+refs/heads/*:refs/remotes/origin/*")
	c.Assert(spec.Validate(), Equals, nil)

	spec = RefSpec("refs/heads/*:refs/remotes/origin/")
	c.Assert(spec.Validate(), Equals, ErrRefSpecMalformedWildcard)

	spec = RefSpec("refs/heads/master:refs/remotes/origin/master")
	c.Assert(spec.Validate(), Equals, nil)

	spec = RefSpec(":refs/heads/master")
	c.Assert(spec.Validate(), Equals, nil)

	spec = RefSpec(":refs/heads/*")
	c.Assert(spec.Validate(), Equals, ErrRefSpecMalformedWildcard)

	spec = RefSpec(":*")
	c.Assert(spec.Validate(), Equals, ErrRefSpecMalformedWildcard)

	spec = RefSpec("refs/heads/*")
	c.Assert(spec.Validate(), Equals, ErrRefSpecMalformedSeparator)

	spec = RefSpec("refs/heads:")
	c.Assert(spec.Validate(), Equals, ErrRefSpecMalformedSeparator)
}

func (s *RefSpecSuite) TestRefSpecIsForceUpdate(c *C) {
	spec := RefSpec("+refs/heads/*:refs/remotes/origin/*")
	c.Assert(spec.IsForceUpdate(), Equals, true)

	spec = RefSpec("refs/heads/*:refs/remotes/origin/*")
	c.Assert(spec.IsForceUpdate(), Equals, false)
}

func (s *RefSpecSuite) TestRefSpecIsDelete(c *C) {
	spec := RefSpec(":refs/heads/master")
	c.Assert(spec.IsDelete(), Equals, true)

	spec = RefSpec("+refs/heads/*:refs/remotes/origin/*")
	c.Assert(spec.IsDelete(), Equals, false)

	spec = RefSpec("refs/heads/*:refs/remotes/origin/*")
	c.Assert(spec.IsDelete(), Equals, false)
}

func (s *RefSpecSuite) TestRefSpecSrc(c *C) {
	spec := RefSpec("refs/heads/*:refs/remotes/origin/*")
	c.Assert(spec.Src(), Equals, "refs/heads/*")

	spec = RefSpec(":refs/heads/master")
	c.Assert(spec.Src(), Equals, "")
}

func (s *RefSpecSuite) TestRefSpecMatch(c *C) {
	spec := RefSpec("refs/heads/master:refs/remotes/origin/master")
	c.Assert(spec.Match(plumbing.ReferenceName("refs/heads/foo")), Equals, false)
	c.Assert(spec.Match(plumbing.ReferenceName("refs/heads/master")), Equals, true)

	spec = RefSpec(":refs/heads/master")
	c.Assert(spec.Match(plumbing.ReferenceName("")), Equals, true)
	c.Assert(spec.Match(plumbing.ReferenceName("refs/heads/master")), Equals, false)
}

func (s *RefSpecSuite) TestRefSpecMatchGlob(c *C) {
	spec := RefSpec("refs/heads/*:refs/remotes/origin/*")
	c.Assert(spec.Match(plumbing.ReferenceName("refs/tag/foo")), Equals, false)
	c.Assert(spec.Match(plumbing.ReferenceName("refs/heads/foo")), Equals, true)
}

func (s *RefSpecSuite) TestRefSpecDst(c *C) {
	spec := RefSpec("refs/heads/master:refs/remotes/origin/master")
	c.Assert(
		spec.Dst(plumbing.ReferenceName("refs/heads/master")).String(), Equals,
		"refs/remotes/origin/master",
	)
}

func (s *RefSpecSuite) TestRefSpecDstBlob(c *C) {
	spec := RefSpec("refs/heads/*:refs/remotes/origin/*")
	c.Assert(
		spec.Dst(plumbing.ReferenceName("refs/heads/foo")).String(), Equals,
		"refs/remotes/origin/foo",
	)
}
func (s *RefSpecSuite) TestMatchAny(c *C) {
	specs := []RefSpec{
		"refs/heads/bar:refs/remotes/origin/foo",
		"refs/heads/foo:refs/remotes/origin/bar",
	}

	c.Assert(MatchAny(specs, plumbing.ReferenceName("refs/heads/foo")), Equals, true)
	c.Assert(MatchAny(specs, plumbing.ReferenceName("refs/heads/bar")), Equals, true)
	c.Assert(MatchAny(specs, plumbing.ReferenceName("refs/heads/master")), Equals, false)
}
