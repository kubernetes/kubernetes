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
	"bytes"

	. "gopkg.in/check.v1"
)

type codecTestSuite struct{}

var _ = Suite(&codecTestSuite{})

func (s *codecTestSuite) TestFromBytes(c *C) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	u1, err := FromBytes(b1)
	c.Assert(err, IsNil)
	c.Assert(u1, Equals, u)

	b2 := []byte{}
	_, err = FromBytes(b2)
	c.Assert(err, NotNil)
}

func (s *codecTestSuite) BenchmarkFromBytes(c *C) {
	bytes := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	for i := 0; i < c.N; i++ {
		FromBytes(bytes)
	}
}

func (s *codecTestSuite) TestMarshalBinary(c *C) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	b2, err := u.MarshalBinary()
	c.Assert(err, IsNil)
	c.Assert(bytes.Equal(b1, b2), Equals, true)
}

func (s *codecTestSuite) BenchmarkMarshalBinary(c *C) {
	u := NewV4()
	for i := 0; i < c.N; i++ {
		u.MarshalBinary()
	}
}

func (s *codecTestSuite) TestUnmarshalBinary(c *C) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	u1 := UUID{}
	err := u1.UnmarshalBinary(b1)
	c.Assert(err, IsNil)
	c.Assert(u1, Equals, u)

	b2 := []byte{}
	u2 := UUID{}
	err = u2.UnmarshalBinary(b2)
	c.Assert(err, NotNil)
}

func (s *codecTestSuite) TestFromString(c *C) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	s1 := "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
	s2 := "{6ba7b810-9dad-11d1-80b4-00c04fd430c8}"
	s3 := "urn:uuid:6ba7b810-9dad-11d1-80b4-00c04fd430c8"
	s4 := "6ba7b8109dad11d180b400c04fd430c8"
	s5 := "urn:uuid:6ba7b8109dad11d180b400c04fd430c8"

	_, err := FromString("")
	c.Assert(err, NotNil)

	u1, err := FromString(s1)
	c.Assert(err, IsNil)
	c.Assert(u1, Equals, u)

	u2, err := FromString(s2)
	c.Assert(err, IsNil)
	c.Assert(u2, Equals, u)

	u3, err := FromString(s3)
	c.Assert(err, IsNil)
	c.Assert(u3, Equals, u)

	u4, err := FromString(s4)
	c.Assert(err, IsNil)
	c.Assert(u4, Equals, u)

	u5, err := FromString(s5)
	c.Assert(err, IsNil)
	c.Assert(u5, Equals, u)
}

func (s *codecTestSuite) BenchmarkFromString(c *C) {
	str := "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
	for i := 0; i < c.N; i++ {
		FromString(str)
	}
}

func (s *codecTestSuite) BenchmarkFromStringUrn(c *C) {
	str := "urn:uuid:6ba7b810-9dad-11d1-80b4-00c04fd430c8"
	for i := 0; i < c.N; i++ {
		FromString(str)
	}
}

func (s *codecTestSuite) BenchmarkFromStringWithBrackets(c *C) {
	str := "{6ba7b810-9dad-11d1-80b4-00c04fd430c8}"
	for i := 0; i < c.N; i++ {
		FromString(str)
	}
}

func (s *codecTestSuite) TestFromStringShort(c *C) {
	// Invalid 35-character UUID string
	s1 := "6ba7b810-9dad-11d1-80b4-00c04fd430c"

	for i := len(s1); i >= 0; i-- {
		_, err := FromString(s1[:i])
		c.Assert(err, NotNil)
	}
}

func (s *codecTestSuite) TestFromStringLong(c *C) {
	// Invalid 37+ character UUID string
	strings := []string{
		"6ba7b810-9dad-11d1-80b4-00c04fd430c8=",
		"6ba7b810-9dad-11d1-80b4-00c04fd430c8}",
		"{6ba7b810-9dad-11d1-80b4-00c04fd430c8}f",
		"6ba7b810-9dad-11d1-80b4-00c04fd430c800c04fd430c8",
	}

	for _, str := range strings {
		_, err := FromString(str)
		c.Assert(err, NotNil)
	}
}

func (s *codecTestSuite) TestFromStringInvalid(c *C) {
	// Invalid UUID string formats
	strings := []string{
		"6ba7b8109dad11d180b400c04fd430c86ba7b8109dad11d180b400c04fd430c8",
		"urn:uuid:{6ba7b810-9dad-11d1-80b4-00c04fd430c8}",
		"uuid:urn:6ba7b810-9dad-11d1-80b4-00c04fd430c8",
		"uuid:urn:6ba7b8109dad11d180b400c04fd430c8",
		"6ba7b8109-dad-11d1-80b4-00c04fd430c8",
		"6ba7b810-9dad1-1d1-80b4-00c04fd430c8",
		"6ba7b810-9dad-11d18-0b4-00c04fd430c8",
		"6ba7b810-9dad-11d1-80b40-0c04fd430c8",
		"6ba7b810+9dad+11d1+80b4+00c04fd430c8",
		"(6ba7b810-9dad-11d1-80b4-00c04fd430c8}",
		"{6ba7b810-9dad-11d1-80b4-00c04fd430c8>",
		"zba7b810-9dad-11d1-80b4-00c04fd430c8",
		"6ba7b810-9dad11d180b400c04fd430c8",
		"6ba7b8109dad-11d180b400c04fd430c8",
		"6ba7b8109dad11d1-80b400c04fd430c8",
		"6ba7b8109dad11d180b4-00c04fd430c8",
	}

	for _, str := range strings {
		_, err := FromString(str)
		c.Assert(err, NotNil)
	}
}

func (s *codecTestSuite) TestFromStringOrNil(c *C) {
	u := FromStringOrNil("")
	c.Assert(u, Equals, Nil)
}

func (s *codecTestSuite) TestFromBytesOrNil(c *C) {
	b := []byte{}
	u := FromBytesOrNil(b)
	c.Assert(u, Equals, Nil)
}

func (s *codecTestSuite) TestMarshalText(c *C) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

	b2, err := u.MarshalText()
	c.Assert(err, IsNil)
	c.Assert(bytes.Equal(b1, b2), Equals, true)
}

func (s *codecTestSuite) BenchmarkMarshalText(c *C) {
	u := NewV4()
	for i := 0; i < c.N; i++ {
		u.MarshalText()
	}
}

func (s *codecTestSuite) TestUnmarshalText(c *C) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

	u1 := UUID{}
	err := u1.UnmarshalText(b1)
	c.Assert(err, IsNil)
	c.Assert(u1, Equals, u)

	b2 := []byte("")
	u2 := UUID{}
	err = u2.UnmarshalText(b2)
	c.Assert(err, NotNil)
}

func (s *codecTestSuite) BenchmarkUnmarshalText(c *C) {
	bytes := []byte("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
	u := UUID{}
	for i := 0; i < c.N; i++ {
		u.UnmarshalText(bytes)
	}
}

var sink string

func (s *codecTestSuite) BenchmarkMarshalToString(c *C) {
	u := NewV4()
	for i := 0; i < c.N; i++ {
		sink = u.String()
	}
}
