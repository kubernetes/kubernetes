package packfile

import (
	"io"

	"gopkg.in/src-d/go-git.v4/plumbing"

	. "gopkg.in/check.v1"
)

type ObjectToPackSuite struct{}

var _ = Suite(&ObjectToPackSuite{})

func (s *ObjectToPackSuite) TestObjectToPack(c *C) {
	obj := &dummyObject{}
	otp := newObjectToPack(obj)
	c.Assert(obj, Equals, otp.Object)
	c.Assert(obj, Equals, otp.Original)
	c.Assert(otp.Base, IsNil)
	c.Assert(otp.IsDelta(), Equals, false)

	original := &dummyObject{}
	delta := &dummyObject{}
	deltaToPack := newDeltaObjectToPack(otp, original, delta)
	c.Assert(obj, Equals, deltaToPack.Object)
	c.Assert(original, Equals, deltaToPack.Original)
	c.Assert(otp, Equals, deltaToPack.Base)
	c.Assert(deltaToPack.IsDelta(), Equals, true)
}

type dummyObject struct{}

func (*dummyObject) Hash() plumbing.Hash             { return plumbing.ZeroHash }
func (*dummyObject) Type() plumbing.ObjectType       { return plumbing.InvalidObject }
func (*dummyObject) SetType(plumbing.ObjectType)     {}
func (*dummyObject) Size() int64                     { return 0 }
func (*dummyObject) SetSize(s int64)                 {}
func (*dummyObject) Reader() (io.ReadCloser, error)  { return nil, nil }
func (*dummyObject) Writer() (io.WriteCloser, error) { return nil, nil }
