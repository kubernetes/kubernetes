package noder

import (
	"testing"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type NoderSuite struct{}

var _ = Suite(&NoderSuite{})

type noderMock struct {
	name     string
	hash     []byte
	isDir    bool
	children []Noder
}

func (n noderMock) String() string             { return n.Name() }
func (n noderMock) Hash() []byte               { return n.hash }
func (n noderMock) Name() string               { return n.name }
func (n noderMock) IsDir() bool                { return n.isDir }
func (n noderMock) Children() ([]Noder, error) { return n.children, nil }
func (n noderMock) NumChildren() (int, error)  { return len(n.children), nil }

// Returns a sequence with the noders 3, 2, and 1 from the
// following diagram:
//
//   3
//   |
//   2
//   |
//   1
//  / \
// c1  c2
//
// This is also the path of "1".
func nodersFixture() []Noder {
	n1 := &noderMock{
		name:     "1",
		hash:     []byte{0x00, 0x01, 0x02},
		isDir:    true,
		children: childrenFixture(),
	}
	n2 := &noderMock{name: "2"}
	n3 := &noderMock{name: "3"}
	return []Noder{n3, n2, n1}
}

// Returns a collection of 2 noders: c1, c2.
func childrenFixture() []Noder {
	c1 := &noderMock{name: "c1"}
	c2 := &noderMock{name: "c2"}
	return []Noder{c1, c2}
}

// Returns the same as nodersFixture but sorted by name, this is: "1",
// "2" and then "3".
func sortedNodersFixture() []Noder {
	n1 := &noderMock{
		name:     "1",
		hash:     []byte{0x00, 0x01, 0x02},
		isDir:    true,
		children: childrenFixture(),
	}
	n2 := &noderMock{name: "2"}
	n3 := &noderMock{name: "3"}
	return []Noder{n1, n2, n3} // the same as nodersFixture but sorted by name
}

// returns nodersFixture as the path of "1".
func pathFixture() Path {
	return Path(nodersFixture())
}

func (s *NoderSuite) TestString(c *C) {
	c.Assert(pathFixture().String(), Equals, "3/2/1")
}

func (s *NoderSuite) TestLast(c *C) {
	c.Assert(pathFixture().Last().Name(), Equals, "1")
}

func (s *NoderSuite) TestPathImplementsNoder(c *C) {
	p := pathFixture()
	c.Assert(p.Name(), Equals, "1")
	c.Assert(p.Hash(), DeepEquals, []byte{0x00, 0x01, 0x02})
	c.Assert(p.IsDir(), Equals, true)

	children, err := p.Children()
	c.Assert(err, IsNil)
	c.Assert(children, DeepEquals, childrenFixture())

	numChildren, err := p.NumChildren()
	c.Assert(err, IsNil)
	c.Assert(numChildren, Equals, 2)
}
