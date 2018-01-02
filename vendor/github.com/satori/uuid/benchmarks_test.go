// Copyright (C) 2013-2015 by Maxim Bublis <b@codemonkey.ru>
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
	"testing"
)

func BenchmarkFromBytes(b *testing.B) {
	bytes := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	for i := 0; i < b.N; i++ {
		FromBytes(bytes)
	}
}

func BenchmarkFromString(b *testing.B) {
	s := "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
	for i := 0; i < b.N; i++ {
		FromString(s)
	}
}

func BenchmarkFromStringUrn(b *testing.B) {
	s := "urn:uuid:6ba7b810-9dad-11d1-80b4-00c04fd430c8"
	for i := 0; i < b.N; i++ {
		FromString(s)
	}
}

func BenchmarkFromStringWithBrackets(b *testing.B) {
	s := "{6ba7b810-9dad-11d1-80b4-00c04fd430c8}"
	for i := 0; i < b.N; i++ {
		FromString(s)
	}
}

func BenchmarkNewV1(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewV1()
	}
}

func BenchmarkNewV2(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewV2(DomainPerson)
	}
}

func BenchmarkNewV3(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewV3(NamespaceDNS, "www.example.com")
	}
}

func BenchmarkNewV4(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewV4()
	}
}

func BenchmarkNewV5(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewV5(NamespaceDNS, "www.example.com")
	}
}

func BenchmarkMarshalBinary(b *testing.B) {
	u := NewV4()
	for i := 0; i < b.N; i++ {
		u.MarshalBinary()
	}
}

func BenchmarkMarshalText(b *testing.B) {
	u := NewV4()
	for i := 0; i < b.N; i++ {
		u.MarshalText()
	}
}

func BenchmarkUnmarshalBinary(b *testing.B) {
	bytes := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	u := UUID{}
	for i := 0; i < b.N; i++ {
		u.UnmarshalBinary(bytes)
	}
}

func BenchmarkUnmarshalText(b *testing.B) {
	bytes := []byte("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
	u := UUID{}
	for i := 0; i < b.N; i++ {
		u.UnmarshalText(bytes)
	}
}

var sink string

func BenchmarkMarshalToString(b *testing.B) {
	u := NewV4()
	for i := 0; i < b.N; i++ {
		sink = u.String()
	}
}
