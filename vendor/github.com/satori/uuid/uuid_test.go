// Copyright (C) 2013, 2015 by Maxim Bublis <b@codemonkey.ru>
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
)

func TestBytes(t *testing.T) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	bytes1 := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	if !bytes.Equal(u.Bytes(), bytes1) {
		t.Errorf("Incorrect bytes representation for UUID: %s", u)
	}
}

func TestString(t *testing.T) {
	if NamespaceDNS.String() != "6ba7b810-9dad-11d1-80b4-00c04fd430c8" {
		t.Errorf("Incorrect string representation for UUID: %s", NamespaceDNS.String())
	}
}

func TestEqual(t *testing.T) {
	if !Equal(NamespaceDNS, NamespaceDNS) {
		t.Errorf("Incorrect comparison of %s and %s", NamespaceDNS, NamespaceDNS)
	}

	if Equal(NamespaceDNS, NamespaceURL) {
		t.Errorf("Incorrect comparison of %s and %s", NamespaceDNS, NamespaceURL)
	}
}

func TestOr(t *testing.T) {
	u1 := UUID{0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff}
	u2 := UUID{0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00}

	u := UUID{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}

	if !Equal(u, Or(u1, u2)) {
		t.Errorf("Incorrect bitwise OR result %s", Or(u1, u2))
	}
}

func TestAnd(t *testing.T) {
	u1 := UUID{0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff}
	u2 := UUID{0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00}

	u := UUID{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}

	if !Equal(u, And(u1, u2)) {
		t.Errorf("Incorrect bitwise AND result %s", And(u1, u2))
	}
}

func TestVersion(t *testing.T) {
	u := UUID{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}

	if u.Version() != 1 {
		t.Errorf("Incorrect version for UUID: %d", u.Version())
	}
}

func TestSetVersion(t *testing.T) {
	u := UUID{}
	u.SetVersion(4)

	if u.Version() != 4 {
		t.Errorf("Incorrect version for UUID after u.setVersion(4): %d", u.Version())
	}
}

func TestVariant(t *testing.T) {
	u1 := UUID{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}

	if u1.Variant() != VariantNCS {
		t.Errorf("Incorrect variant for UUID variant %d: %d", VariantNCS, u1.Variant())
	}

	u2 := UUID{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}

	if u2.Variant() != VariantRFC4122 {
		t.Errorf("Incorrect variant for UUID variant %d: %d", VariantRFC4122, u2.Variant())
	}

	u3 := UUID{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}

	if u3.Variant() != VariantMicrosoft {
		t.Errorf("Incorrect variant for UUID variant %d: %d", VariantMicrosoft, u3.Variant())
	}

	u4 := UUID{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}

	if u4.Variant() != VariantFuture {
		t.Errorf("Incorrect variant for UUID variant %d: %d", VariantFuture, u4.Variant())
	}
}

func TestSetVariant(t *testing.T) {
	u := new(UUID)
	u.SetVariant()

	if u.Variant() != VariantRFC4122 {
		t.Errorf("Incorrect variant for UUID after u.setVariant(): %d", u.Variant())
	}
}

func TestFromBytes(t *testing.T) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	u1, err := FromBytes(b1)
	if err != nil {
		t.Errorf("Error parsing UUID from bytes: %s", err)
	}

	if !Equal(u, u1) {
		t.Errorf("UUIDs should be equal: %s and %s", u, u1)
	}

	b2 := []byte{}

	_, err = FromBytes(b2)
	if err == nil {
		t.Errorf("Should return error parsing from empty byte slice, got %s", err)
	}
}

func TestMarshalBinary(t *testing.T) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	b2, err := u.MarshalBinary()
	if err != nil {
		t.Errorf("Error marshaling UUID: %s", err)
	}

	if !bytes.Equal(b1, b2) {
		t.Errorf("Marshaled UUID should be %s, got %s", b1, b2)
	}
}

func TestUnmarshalBinary(t *testing.T) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	u1 := UUID{}
	err := u1.UnmarshalBinary(b1)
	if err != nil {
		t.Errorf("Error unmarshaling UUID: %s", err)
	}

	if !Equal(u, u1) {
		t.Errorf("UUIDs should be equal: %s and %s", u, u1)
	}

	b2 := []byte{}
	u2 := UUID{}

	err = u2.UnmarshalBinary(b2)
	if err == nil {
		t.Errorf("Should return error unmarshalling from empty byte slice, got %s", err)
	}
}

func TestFromString(t *testing.T) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	s1 := "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
	s2 := "{6ba7b810-9dad-11d1-80b4-00c04fd430c8}"
	s3 := "urn:uuid:6ba7b810-9dad-11d1-80b4-00c04fd430c8"

	_, err := FromString("")
	if err == nil {
		t.Errorf("Should return error trying to parse empty string, got %s", err)
	}

	u1, err := FromString(s1)
	if err != nil {
		t.Errorf("Error parsing UUID from string: %s", err)
	}

	if !Equal(u, u1) {
		t.Errorf("UUIDs should be equal: %s and %s", u, u1)
	}

	u2, err := FromString(s2)
	if err != nil {
		t.Errorf("Error parsing UUID from string: %s", err)
	}

	if !Equal(u, u2) {
		t.Errorf("UUIDs should be equal: %s and %s", u, u2)
	}

	u3, err := FromString(s3)
	if err != nil {
		t.Errorf("Error parsing UUID from string: %s", err)
	}

	if !Equal(u, u3) {
		t.Errorf("UUIDs should be equal: %s and %s", u, u3)
	}
}

func TestFromStringShort(t *testing.T) {
	// Invalid 35-character UUID string
	s1 := "6ba7b810-9dad-11d1-80b4-00c04fd430c"

	for i := len(s1); i >= 0; i-- {
		_, err := FromString(s1[:i])
		if err == nil {
			t.Errorf("Should return error trying to parse too short string, got %s", err)
		}
	}
}

func TestFromStringLong(t *testing.T) {
	// Invalid 37+ character UUID string
	s := []string{
		"6ba7b810-9dad-11d1-80b4-00c04fd430c8=",
		"6ba7b810-9dad-11d1-80b4-00c04fd430c8}",
		"{6ba7b810-9dad-11d1-80b4-00c04fd430c8}f",
		"6ba7b810-9dad-11d1-80b4-00c04fd430c800c04fd430c8",
	}

	for _, str := range s {
		_, err := FromString(str)
		if err == nil {
			t.Errorf("Should return error trying to parse too long string, passed %s", str)
		}
	}
}

func TestFromStringInvalid(t *testing.T) {
	// Invalid UUID string formats
	s := []string{
		"6ba7b8109dad11d180b400c04fd430c8",
		"6ba7b8109dad11d180b400c04fd430c86ba7b8109dad11d180b400c04fd430c8",
		"urn:uuid:{6ba7b810-9dad-11d1-80b4-00c04fd430c8}",
		"6ba7b8109-dad-11d1-80b4-00c04fd430c8",
		"6ba7b810-9dad1-1d1-80b4-00c04fd430c8",
		"6ba7b810-9dad-11d18-0b4-00c04fd430c8",
		"6ba7b810-9dad-11d1-80b40-0c04fd430c8",
		"6ba7b810+9dad+11d1+80b4+00c04fd430c8",
		"6ba7b810-9dad11d180b400c04fd430c8",
		"6ba7b8109dad-11d180b400c04fd430c8",
		"6ba7b8109dad11d1-80b400c04fd430c8",
		"6ba7b8109dad11d180b4-00c04fd430c8",
	}

	for _, str := range s {
		_, err := FromString(str)
		if err == nil {
			t.Errorf("Should return error trying to parse invalid string, passed %s", str)
		}
	}
}

func TestFromStringOrNil(t *testing.T) {
	u := FromStringOrNil("")
	if u != Nil {
		t.Errorf("Should return Nil UUID on parse failure, got %s", u)
	}
}

func TestFromBytesOrNil(t *testing.T) {
	b := []byte{}
	u := FromBytesOrNil(b)
	if u != Nil {
		t.Errorf("Should return Nil UUID on parse failure, got %s", u)
	}
}

func TestMarshalText(t *testing.T) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

	b2, err := u.MarshalText()
	if err != nil {
		t.Errorf("Error marshaling UUID: %s", err)
	}

	if !bytes.Equal(b1, b2) {
		t.Errorf("Marshaled UUID should be %s, got %s", b1, b2)
	}
}

func TestUnmarshalText(t *testing.T) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

	u1 := UUID{}
	err := u1.UnmarshalText(b1)
	if err != nil {
		t.Errorf("Error unmarshaling UUID: %s", err)
	}

	if !Equal(u, u1) {
		t.Errorf("UUIDs should be equal: %s and %s", u, u1)
	}

	b2 := []byte("")
	u2 := UUID{}

	err = u2.UnmarshalText(b2)
	if err == nil {
		t.Errorf("Should return error trying to unmarshal from empty string")
	}
}

func TestValue(t *testing.T) {
	u, err := FromString("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
	if err != nil {
		t.Errorf("Error parsing UUID from string: %s", err)
	}

	val, err := u.Value()
	if err != nil {
		t.Errorf("Error getting UUID value: %s", err)
	}

	if val != u.String() {
		t.Errorf("Wrong value returned, should be equal: %s and %s", val, u)
	}
}

func TestValueNil(t *testing.T) {
	u := UUID{}

	val, err := u.Value()
	if err != nil {
		t.Errorf("Error getting UUID value: %s", err)
	}

	if val != Nil.String() {
		t.Errorf("Wrong value returned, should be equal to UUID.Nil: %s", val)
	}
}

func TestNullUUIDValueNil(t *testing.T) {
	u := NullUUID{}

	val, err := u.Value()
	if err != nil {
		t.Errorf("Error getting UUID value: %s", err)
	}

	if val != nil {
		t.Errorf("Wrong value returned, should be nil: %s", val)
	}
}

func TestScanBinary(t *testing.T) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	u1 := UUID{}
	err := u1.Scan(b1)
	if err != nil {
		t.Errorf("Error unmarshaling UUID: %s", err)
	}

	if !Equal(u, u1) {
		t.Errorf("UUIDs should be equal: %s and %s", u, u1)
	}

	b2 := []byte{}
	u2 := UUID{}

	err = u2.Scan(b2)
	if err == nil {
		t.Errorf("Should return error unmarshalling from empty byte slice, got %s", err)
	}
}

func TestScanString(t *testing.T) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	s1 := "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

	u1 := UUID{}
	err := u1.Scan(s1)
	if err != nil {
		t.Errorf("Error unmarshaling UUID: %s", err)
	}

	if !Equal(u, u1) {
		t.Errorf("UUIDs should be equal: %s and %s", u, u1)
	}

	s2 := ""
	u2 := UUID{}

	err = u2.Scan(s2)
	if err == nil {
		t.Errorf("Should return error trying to unmarshal from empty string")
	}
}

func TestScanText(t *testing.T) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	b1 := []byte("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

	u1 := UUID{}
	err := u1.Scan(b1)
	if err != nil {
		t.Errorf("Error unmarshaling UUID: %s", err)
	}

	if !Equal(u, u1) {
		t.Errorf("UUIDs should be equal: %s and %s", u, u1)
	}

	b2 := []byte("")
	u2 := UUID{}

	err = u2.Scan(b2)
	if err == nil {
		t.Errorf("Should return error trying to unmarshal from empty string")
	}
}

func TestScanUnsupported(t *testing.T) {
	u := UUID{}

	err := u.Scan(true)
	if err == nil {
		t.Errorf("Should return error trying to unmarshal from bool")
	}
}

func TestScanNil(t *testing.T) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}

	err := u.Scan(nil)
	if err == nil {
		t.Errorf("Error UUID shouldn't allow unmarshalling from nil")
	}
}

func TestNullUUIDScanValid(t *testing.T) {
	u := UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}
	s1 := "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

	u1 := NullUUID{}
	err := u1.Scan(s1)
	if err != nil {
		t.Errorf("Error unmarshaling NullUUID: %s", err)
	}

	if !u1.Valid {
		t.Errorf("NullUUID should be valid")
	}

	if !Equal(u, u1.UUID) {
		t.Errorf("UUIDs should be equal: %s and %s", u, u1.UUID)
	}
}

func TestNullUUIDScanNil(t *testing.T) {
	u := NullUUID{UUID{0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8}, true}

	err := u.Scan(nil)
	if err != nil {
		t.Errorf("Error unmarshaling NullUUID: %s", err)
	}

	if u.Valid {
		t.Errorf("NullUUID should not be valid")
	}

	if !Equal(u.UUID, Nil) {
		t.Errorf("NullUUID value should be equal to Nil: %v", u)
	}
}

func TestNewV1(t *testing.T) {
	u := NewV1()

	if u.Version() != 1 {
		t.Errorf("UUIDv1 generated with incorrect version: %d", u.Version())
	}

	if u.Variant() != VariantRFC4122 {
		t.Errorf("UUIDv1 generated with incorrect variant: %d", u.Variant())
	}

	u1 := NewV1()
	u2 := NewV1()

	if Equal(u1, u2) {
		t.Errorf("UUIDv1 generated two equal UUIDs: %s and %s", u1, u2)
	}

	oldFunc := epochFunc
	epochFunc = func() uint64 { return 0 }

	u3 := NewV1()
	u4 := NewV1()

	if Equal(u3, u4) {
		t.Errorf("UUIDv1 generated two equal UUIDs: %s and %s", u3, u4)
	}

	epochFunc = oldFunc
}

func TestNewV2(t *testing.T) {
	u1 := NewV2(DomainPerson)

	if u1.Version() != 2 {
		t.Errorf("UUIDv2 generated with incorrect version: %d", u1.Version())
	}

	if u1.Variant() != VariantRFC4122 {
		t.Errorf("UUIDv2 generated with incorrect variant: %d", u1.Variant())
	}

	u2 := NewV2(DomainGroup)

	if u2.Version() != 2 {
		t.Errorf("UUIDv2 generated with incorrect version: %d", u2.Version())
	}

	if u2.Variant() != VariantRFC4122 {
		t.Errorf("UUIDv2 generated with incorrect variant: %d", u2.Variant())
	}
}

func TestNewV3(t *testing.T) {
	u := NewV3(NamespaceDNS, "www.example.com")

	if u.Version() != 3 {
		t.Errorf("UUIDv3 generated with incorrect version: %d", u.Version())
	}

	if u.Variant() != VariantRFC4122 {
		t.Errorf("UUIDv3 generated with incorrect variant: %d", u.Variant())
	}

	if u.String() != "5df41881-3aed-3515-88a7-2f4a814cf09e" {
		t.Errorf("UUIDv3 generated incorrectly: %s", u.String())
	}

	u = NewV3(NamespaceDNS, "python.org")

	if u.String() != "6fa459ea-ee8a-3ca4-894e-db77e160355e" {
		t.Errorf("UUIDv3 generated incorrectly: %s", u.String())
	}

	u1 := NewV3(NamespaceDNS, "golang.org")
	u2 := NewV3(NamespaceDNS, "golang.org")
	if !Equal(u1, u2) {
		t.Errorf("UUIDv3 generated different UUIDs for same namespace and name: %s and %s", u1, u2)
	}

	u3 := NewV3(NamespaceDNS, "example.com")
	if Equal(u1, u3) {
		t.Errorf("UUIDv3 generated same UUIDs for different names in same namespace: %s and %s", u1, u2)
	}

	u4 := NewV3(NamespaceURL, "golang.org")
	if Equal(u1, u4) {
		t.Errorf("UUIDv3 generated same UUIDs for sane names in different namespaces: %s and %s", u1, u4)
	}
}

func TestNewV4(t *testing.T) {
	u := NewV4()

	if u.Version() != 4 {
		t.Errorf("UUIDv4 generated with incorrect version: %d", u.Version())
	}

	if u.Variant() != VariantRFC4122 {
		t.Errorf("UUIDv4 generated with incorrect variant: %d", u.Variant())
	}
}

func TestNewV5(t *testing.T) {
	u := NewV5(NamespaceDNS, "www.example.com")

	if u.Version() != 5 {
		t.Errorf("UUIDv5 generated with incorrect version: %d", u.Version())
	}

	if u.Variant() != VariantRFC4122 {
		t.Errorf("UUIDv5 generated with incorrect variant: %d", u.Variant())
	}

	u = NewV5(NamespaceDNS, "python.org")

	if u.String() != "886313e1-3b8a-5372-9b90-0c9aee199e5d" {
		t.Errorf("UUIDv5 generated incorrectly: %s", u.String())
	}

	u1 := NewV5(NamespaceDNS, "golang.org")
	u2 := NewV5(NamespaceDNS, "golang.org")
	if !Equal(u1, u2) {
		t.Errorf("UUIDv5 generated different UUIDs for same namespace and name: %s and %s", u1, u2)
	}

	u3 := NewV5(NamespaceDNS, "example.com")
	if Equal(u1, u3) {
		t.Errorf("UUIDv5 generated same UUIDs for different names in same namespace: %s and %s", u1, u2)
	}

	u4 := NewV5(NamespaceURL, "golang.org")
	if Equal(u1, u4) {
		t.Errorf("UUIDv3 generated same UUIDs for sane names in different namespaces: %s and %s", u1, u4)
	}
}
