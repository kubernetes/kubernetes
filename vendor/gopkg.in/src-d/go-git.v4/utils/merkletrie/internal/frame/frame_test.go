package frame

import (
	"fmt"
	"testing"

	"gopkg.in/src-d/go-git.v4/utils/merkletrie/internal/fsnoder"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie/noder"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type FrameSuite struct{}

var _ = Suite(&FrameSuite{})

func (s *FrameSuite) TestNewFrameFromEmptyDir(c *C) {
	A, err := fsnoder.New("A()")
	c.Assert(err, IsNil)

	frame, err := New(A)
	c.Assert(err, IsNil)

	expectedString := `[]`
	c.Assert(frame.String(), Equals, expectedString)

	first, ok := frame.First()
	c.Assert(first, IsNil)
	c.Assert(ok, Equals, false)

	first, ok = frame.First()
	c.Assert(first, IsNil)
	c.Assert(ok, Equals, false)

	l := frame.Len()
	c.Assert(l, Equals, 0)
}

func (s *FrameSuite) TestNewFrameFromNonEmpty(c *C) {
	//        _______A/________
	//        |     /  \       |
	//        x    y    B/     C/
	//                         |
	//                         z
	root, err := fsnoder.New("A(x<> y<> B() C(z<>))")
	c.Assert(err, IsNil)
	frame, err := New(root)
	c.Assert(err, IsNil)

	expectedString := `["B", "C", "x", "y"]`
	c.Assert(frame.String(), Equals, expectedString)

	l := frame.Len()
	c.Assert(l, Equals, 4)

	checkFirstAndDrop(c, frame, "B", true)
	l = frame.Len()
	c.Assert(l, Equals, 3)

	checkFirstAndDrop(c, frame, "C", true)
	l = frame.Len()
	c.Assert(l, Equals, 2)

	checkFirstAndDrop(c, frame, "x", true)
	l = frame.Len()
	c.Assert(l, Equals, 1)

	checkFirstAndDrop(c, frame, "y", true)
	l = frame.Len()
	c.Assert(l, Equals, 0)

	checkFirstAndDrop(c, frame, "", false)
	l = frame.Len()
	c.Assert(l, Equals, 0)

	checkFirstAndDrop(c, frame, "", false)
}

func checkFirstAndDrop(c *C, f *Frame, expectedNodeName string, expectedOK bool) {
	first, ok := f.First()
	c.Assert(ok, Equals, expectedOK)
	if expectedOK {
		c.Assert(first.Name(), Equals, expectedNodeName)
	}

	f.Drop()
}

// a mock noder that returns error when Children() is called
type errorNoder struct{ noder.Noder }

func (e *errorNoder) Children() ([]noder.Noder, error) {
	return nil, fmt.Errorf("mock error")
}

func (s *FrameSuite) TestNewFrameErrors(c *C) {
	_, err := New(&errorNoder{})
	c.Assert(err, ErrorMatches, "mock error")
}
