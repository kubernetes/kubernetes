// Copyright (C) 2013-2018 by Maxim Bublis <b@codemonkey.ru>
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package uuid

import (
	. "gopkg.in/check.v1"
)

type sqlTestSuite struct{}

var _ = Suite(&sqlTestSuite{})

func (s *sqlTestSuite) TestValue(c *C) {
	u, err := FromString("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
	c.Assert(err, IsNil)

	val, err := u.Value()
	c.Assert(err, IsNil)
	c.Assert(val, Equals, u.String())
}

func (s *sqlTestSuite) TestValueNil(c *C) {
	u := UUID{}

	val, err := u.Value()
	c.Assert(err, IsNil)
	c.Assert(val, Equals, Nil.String())
}

func (s *sqlTestSuite) TestNullUUIDValueNil(c *C) {
	u := NullUUID{}

	val, err := u.Value()
	c.Assert(err, IsNil)
	c.Assert(val, IsNil)
}

func (s *sqlTestSuite) TestScanBinary(c *C) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	u1 := UUID{}
	err := u1.Scan(b1)
	c.Assert(err, IsNil)
	c.Assert(u, Equals, u1)

	b2 := []byte{}
	u2 := UUID{}

	err = u2.Scan(b2)
	c.Assert(err, NotNil)
}

func (s *sqlTestSuite) TestScanString(c *C) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	s1 := "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

	u1 := UUID{}
	err := u1.Scan(s1)
	c.Assert(err, IsNil)
	c.Assert(u, Equals, u1)

	s2 := ""
	u2 := UUID{}

	err = u2.Scan(s2)
	c.Assert(err, NotNil)
}

func (s *sqlTestSuite) TestScanText(c *C) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

	u1 := UUID{}
	err := u1.Scan(b1)
	c.Assert(err, IsNil)
	c.Assert(u, Equals, u1)

	b2 := []byte("")
	u2 := UUID{}
	err = u2.Scan(b2)
	c.Assert(err, NotNil)
}

func (s *sqlTestSuite) TestScanUnsupported(c *C) {
	u := UUID{}

	err := u.Scan(true)
	c.Assert(err, NotNil)
}

func (s *sqlTestSuite) TestScanNil(c *C) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	err := u.Scan(nil)
	c.Assert(err, NotNil)
}

func (s *sqlTestSuite) TestNullUUIDScanValid(c *C) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	s1 := "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

	u1 := NullUUID{}
	err := u1.Scan(s1)
	c.Assert(err, IsNil)
	c.Assert(u1.Valid, Equals, true)
	c.Assert(u1.UUID, Equals, u)
}

func (s *sqlTestSuite) TestNullUUIDScanNil(c *C) {
	u := NullUUID{UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}, true}

	err := u.Scan(nil)
	c.Assert(err, IsNil)
	c.Assert(u.Valid, Equals, false)
	c.Assert(u.UUID, Equals, Nil)
}
