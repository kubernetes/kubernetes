package plumbing

import . "gopkg.in/check.v1"

type ReferenceSuite struct{}

var _ = Suite(&ReferenceSuite{})

const (
	ExampleReferenceName ReferenceName = "refs/heads/v4"
)

func (s *ReferenceSuite) TestReferenceTypeString(c *C) {
	c.Assert(SymbolicReference.String(), Equals, "symbolic-reference")
}

func (s *ReferenceSuite) TestReferenceNameShort(c *C) {
	c.Assert(ExampleReferenceName.Short(), Equals, "v4")
}

func (s *ReferenceSuite) TestReferenceNameWithSlash(c *C) {
	r := ReferenceName("refs/remotes/origin/feature/AllowSlashes")
	c.Assert(r.Short(), Equals, "origin/feature/AllowSlashes")
}

func (s *ReferenceSuite) TestReferenceNameNote(c *C) {
	r := ReferenceName("refs/notes/foo")
	c.Assert(r.Short(), Equals, "notes/foo")
}

func (s *ReferenceSuite) TestNewReferenceFromStrings(c *C) {
	r := NewReferenceFromStrings("refs/heads/v4", "6ecf0ef2c2dffb796033e5a02219af86ec6584e5")
	c.Assert(r.Type(), Equals, HashReference)
	c.Assert(r.Name(), Equals, ExampleReferenceName)
	c.Assert(r.Hash(), Equals, NewHash("6ecf0ef2c2dffb796033e5a02219af86ec6584e5"))

	r = NewReferenceFromStrings("HEAD", "ref: refs/heads/v4")
	c.Assert(r.Type(), Equals, SymbolicReference)
	c.Assert(r.Name(), Equals, HEAD)
	c.Assert(r.Target(), Equals, ExampleReferenceName)
}

func (s *ReferenceSuite) TestNewSymbolicReference(c *C) {
	r := NewSymbolicReference(HEAD, ExampleReferenceName)
	c.Assert(r.Type(), Equals, SymbolicReference)
	c.Assert(r.Name(), Equals, HEAD)
	c.Assert(r.Target(), Equals, ExampleReferenceName)
}

func (s *ReferenceSuite) TestNewHashReference(c *C) {
	r := NewHashReference(ExampleReferenceName, NewHash("6ecf0ef2c2dffb796033e5a02219af86ec6584e5"))
	c.Assert(r.Type(), Equals, HashReference)
	c.Assert(r.Name(), Equals, ExampleReferenceName)
	c.Assert(r.Hash(), Equals, NewHash("6ecf0ef2c2dffb796033e5a02219af86ec6584e5"))
}

func (s *ReferenceSuite) TestIsBranch(c *C) {
	r := ExampleReferenceName
	c.Assert(r.IsBranch(), Equals, true)
}

func (s *ReferenceSuite) TestIsNote(c *C) {
	r := ReferenceName("refs/notes/foo")
	c.Assert(r.IsNote(), Equals, true)
}

func (s *ReferenceSuite) TestIsRemote(c *C) {
	r := ReferenceName("refs/remotes/origin/master")
	c.Assert(r.IsRemote(), Equals, true)
}

func (s *ReferenceSuite) TestIsTag(c *C) {
	r := ReferenceName("refs/tags/v3.1.")
	c.Assert(r.IsTag(), Equals, true)
}
