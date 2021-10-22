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
	"testing"

	. "gopkg.in/check.v1"
)

// Hook up gocheck into the "go test" runner.
func TestUUID(t *testing.T) { TestingT(t) }

type testSuite struct{}

var _ = Suite(&testSuite{})

func (s *testSuite) TestBytes(c *C) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	bytes1 := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	c.Assert(bytes.Equal(u.Bytes(), bytes1), Equals, true)
}

func (s *testSuite) TestString(c *C) {
	c.Assert(NamespaceDNS.String(), Equals, "6ba7b810-9dad-11d1-80b4-00c04fd430c8")
}

func (s *testSuite) TestEqual(c *C) {
	c.Assert(Equal(NamespaceDNS, NamespaceDNS), Equals, true)
	c.Assert(Equal(NamespaceDNS, NamespaceURL), Equals, false)
}

func (s *testSuite) TestVersion(c *C) {
	u := UUID{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
	c.Assert(u.Version(), Equals, V1)
}

func (s *testSuite) TestSetVersion(c *C) {
	u := UUID{}
	u.SetVersion(4)
	c.Assert(u.Version(), Equals, V4)
}

func (s *testSuite) TestVariant(c *C) {
	u1 := UUID{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
	c.Assert(u1.Variant(), Equals, VariantNCS)

	u2 := UUID{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
	c.Assert(u2.Variant(), Equals, VariantRFC4122)

	u3 := UUID{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
	c.Assert(u3.Variant(), Equals, VariantMicrosoft)

	u4 := UUID{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
	c.Assert(u4.Variant(), Equals, VariantFuture)
}

func (s *testSuite) TestSetVariant(c *C) {
	u := UUID{}
	u.SetVariant(VariantNCS)
	c.Assert(u.Variant(), Equals, VariantNCS)
	u.SetVariant(VariantRFC4122)
	c.Assert(u.Variant(), Equals, VariantRFC4122)
	u.SetVariant(VariantMicrosoft)
	c.Assert(u.Variant(), Equals, VariantMicrosoft)
	u.SetVariant(VariantFuture)
	c.Assert(u.Variant(), Equals, VariantFuture)
}
