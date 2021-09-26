// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package strfmt

import (
	"encoding"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
)

func TestFormatURI(t *testing.T) {
	uri := URI("http://somewhere.com")
	str := "http://somewhereelse.com"
	testStringFormat(t, &uri, "uri", str, []string{}, []string{"somewhere.com"})
}

func TestFormatEmail(t *testing.T) {
	email := Email("somebody@somewhere.com")
	str := string("somebodyelse@somewhere.com")
	validEmails := []string{
		"blah@gmail.com",
		"test@d.verylongtoplevel",
		"email+tag@gmail.com",
		`" "@example.com`,
		`"Abc\@def"@example.com`,
		`"Fred Bloggs"@example.com`,
		`"Joe\\Blow"@example.com`,
		`"Abc@def"@example.com`,
		"customer/department=shipping@example.com",
		"$A12345@example.com",
		"!def!xyz%abc@example.com",
		"_somename@example.com",
		"!#$%&'*+-/=?^_`{}|~@example.com",
		"Miles.O'Brian@example.com",
		"postmaster@☁→❄→☃→☀→☺→☂→☹→✝.ws",
		"root@localhost",
		"john@com",
		"api@piston.ninja",
	}

	testStringFormat(t, &email, "email", str, validEmails, []string{"somebody@somewhere@com"})
}

func TestFormatHostname(t *testing.T) {
	hostname := Hostname("somewhere.com")
	str := string("somewhere.com")
	veryLongStr := strings.Repeat("a", 256)
	longStr := strings.Repeat("a", 64)
	longAddrSegment := strings.Join([]string{"x", "y", longStr}, ".")
	invalidHostnames := []string{
		"somewhere.com!",
		"user@email.domain",
		"1.1.1.1",
		veryLongStr,
		longAddrSegment,
		// dashes
		"www.example-.org",
		"www.--example.org",
		"-www.example.org",
		"www-.example.org",
		"www.d-.org",
		"www.-d.org",
		"www-",
		"-www",
		// other characters (not in symbols)
		"www.ex ample.org",
		"_www.example.org",
		"www.ex;ample.org",
		"www.example_underscored.org",
		// short top-level domains
		"www.詹姆斯.x",
		"a.b.c.d",
		"-xyz",
		"xyz-",
		"x.",
		"a.b.c.dot-",
		"a.b.c.é;ö",
	}
	validHostnames := []string{
		"somewhere.com",
		"888.com",
		"a.com",
		"a.b.com",
		"a.b.c.com",
		"a.b.c.d.com",
		"a.b.c.d.e.com",
		"1.com",
		"1.2.com",
		"1.2.3.com",
		"1.2.3.4.com",
		"99.domain.com",
		"99.99.domain.com",
		"1wwworg.example.com", // valid, per RFC1123
		"1000wwworg.example.com",
		"xn--bcher-kva.example.com", // puny encoded
		"xn-80ak6aa92e.co",
		"xn-80ak6aa92e.com",
		"xn--ls8h.la",
		"☁→❄→☃→☀→☺→☂→☹→✝.ws",
		"www.example.onion",
		"www.example.ôlà",
		"ôlà.ôlà",
		"ôlà.ôlà.ôlà",
		"ex$ample",
		"localhost",
		"example",
		"x",
		"x-y",
		"a.b.c.dot",
		"www.example.org",
		"a.b.c.d.e.f.g.dot",
		// extended symbol alphabet
		"ex=ample.com",
		"<foo>",
		"www.example-hyphenated.org",
		// localized hostnames
		"www.詹姆斯.org",
		"www.élégigôö.org",
		// long top-level domains
		"www.詹姆斯.london",
	}

	testStringFormat(t, &hostname, "hostname", str, []string{}, invalidHostnames)
	testStringFormat(t, &hostname, "hostname", str, validHostnames, []string{})
}

func TestFormatIPv4(t *testing.T) {
	ipv4 := IPv4("192.168.254.1")
	str := string("192.168.254.2")
	testStringFormat(t, &ipv4, "ipv4", str, []string{}, []string{"198.168.254.2.2"})
}

func TestFormatIPv6(t *testing.T) {
	ipv6 := IPv6("::1")
	str := string("::2")
	// TODO: test ipv6 zones
	testStringFormat(t, &ipv6, "ipv6", str, []string{}, []string{"127.0.0.1"})
}

func TestFormatCIDR(t *testing.T) {
	cidr := CIDR("192.168.254.1/24")
	str := string("192.168.254.2/24")
	testStringFormat(t, &cidr, "cidr", str, []string{"192.0.2.1/24", "2001:db8:a0b:12f0::1/32"}, []string{"198.168.254.2", "2001:db8:a0b:12f0::1"})
}

func TestFormatMAC(t *testing.T) {
	mac := MAC("01:02:03:04:05:06")
	str := string("06:05:04:03:02:01")
	testStringFormat(t, &mac, "mac", str, []string{}, []string{"01:02:03:04:05"})
}

func TestFormatUUID3(t *testing.T) {
	first3 := uuid.NewMD5(uuid.NameSpaceURL, []byte("somewhere.com"))
	other3 := uuid.NewMD5(uuid.NameSpaceURL, []byte("somewhereelse.com"))
	uuid3 := UUID3(first3.String())
	str := other3.String()
	testStringFormat(t, &uuid3, "uuid3", str, []string{}, []string{"not-a-uuid"})

	// special case for zero UUID
	var uuidZero UUID3
	err := uuidZero.UnmarshalJSON([]byte(jsonNull))
	assert.NoError(t, err)
	assert.EqualValues(t, UUID3(""), uuidZero)
}

func TestFormatUUID4(t *testing.T) {
	first4 := uuid.Must(uuid.NewRandom())
	other4 := uuid.Must(uuid.NewRandom())
	uuid4 := UUID4(first4.String())
	str := other4.String()
	testStringFormat(t, &uuid4, "uuid4", str, []string{}, []string{"not-a-uuid"})

	// special case for zero UUID
	var uuidZero UUID4
	err := uuidZero.UnmarshalJSON([]byte(jsonNull))
	assert.NoError(t, err)
	assert.EqualValues(t, UUID4(""), uuidZero)
}

func TestFormatUUID5(t *testing.T) {
	first5 := uuid.NewSHA1(uuid.NameSpaceURL, []byte("somewhere.com"))
	other5 := uuid.NewSHA1(uuid.NameSpaceURL, []byte("somewhereelse.com"))
	uuid5 := UUID5(first5.String())
	str := other5.String()
	testStringFormat(t, &uuid5, "uuid5", str, []string{}, []string{"not-a-uuid"})

	// special case for zero UUID
	var uuidZero UUID5
	err := uuidZero.UnmarshalJSON([]byte(jsonNull))
	assert.NoError(t, err)
	assert.EqualValues(t, UUID5(""), uuidZero)
}

func TestFormatUUID(t *testing.T) {
	first5 := uuid.NewSHA1(uuid.NameSpaceURL, []byte("somewhere.com"))
	other5 := uuid.NewSHA1(uuid.NameSpaceURL, []byte("somewhereelse.com"))
	uuid := UUID(first5.String())
	str := other5.String()
	testStringFormat(t, &uuid, "uuid", str, []string{}, []string{"not-a-uuid"})

	// special case for zero UUID
	var uuidZero UUID
	err := uuidZero.UnmarshalJSON([]byte(jsonNull))
	assert.NoError(t, err)
	assert.EqualValues(t, UUID(""), uuidZero)
}

func TestFormatISBN(t *testing.T) {
	isbn := ISBN("0321751043")
	str := string("0321751043")
	testStringFormat(t, &isbn, "isbn", str, []string{}, []string{"836217463"}) // bad checksum
}

func TestFormatISBN10(t *testing.T) {
	isbn10 := ISBN10("0321751043")
	str := string("0321751043")
	testStringFormat(t, &isbn10, "isbn10", str, []string{}, []string{"836217463"}) // bad checksum
}

func TestFormatISBN13(t *testing.T) {
	isbn13 := ISBN13("978-0321751041")
	str := string("978-0321751041")
	testStringFormat(t, &isbn13, "isbn13", str, []string{}, []string{"978-0321751042"}) // bad checksum
}

func TestFormatHexColor(t *testing.T) {
	hexColor := HexColor("#FFFFFF")
	str := string("#000000")
	testStringFormat(t, &hexColor, "hexcolor", str, []string{}, []string{"#fffffffz"})
}

func TestFormatRGBColor(t *testing.T) {
	rgbColor := RGBColor("rgb(255,255,255)")
	str := string("rgb(0,0,0)")
	testStringFormat(t, &rgbColor, "rgbcolor", str, []string{}, []string{"rgb(300,0,0)"})
}

func TestFormatSSN(t *testing.T) {
	ssn := SSN("111-11-1111")
	str := string("999 99 9999")
	testStringFormat(t, &ssn, "ssn", str, []string{}, []string{"999 99 999"})
}

func TestFormatCreditCard(t *testing.T) {
	creditCard := CreditCard("4111-1111-1111-1111")
	str := string("4012-8888-8888-1881")
	testStringFormat(t, &creditCard, "creditcard", str, []string{}, []string{"9999-9999-9999-999"})
}

func TestFormatPassword(t *testing.T) {
	password := Password("super secret stuff here")
	testStringFormat(t, &password, "password", "super secret!!!", []string{"even more secret"}, []string{})
}

func TestFormatBase64(t *testing.T) {
	const b64 string = "This is a byte array with unprintable chars, but it also isn"
	str := base64.URLEncoding.EncodeToString([]byte(b64))
	b := []byte(b64)
	expected := Base64(b)
	bj := []byte("\"" + str + "\"")

	var subj Base64
	err := subj.UnmarshalText([]byte(str))
	assert.NoError(t, err)
	assert.EqualValues(t, expected, subj)

	b, err = subj.MarshalText()
	assert.NoError(t, err)
	assert.Equal(t, []byte(str), b)

	var subj2 Base64
	err = subj2.UnmarshalJSON(bj)
	assert.NoError(t, err)
	assert.EqualValues(t, expected, subj2)

	b, err = subj2.MarshalJSON()
	assert.NoError(t, err)
	assert.Equal(t, bj, b)

	testValid(t, "byte", str)
	testInvalid(t, "byte", "ZWxpemFiZXRocG9zZXk") // missing pad char
}

type testableFormat interface {
	encoding.TextMarshaler
	encoding.TextUnmarshaler
	json.Marshaler
	json.Unmarshaler
	fmt.Stringer
}

func testStringFormat(t *testing.T, what testableFormat, format, with string, validSamples, invalidSamples []string) {
	// text encoding interface
	b := []byte(with)
	err := what.UnmarshalText(b)
	assert.NoError(t, err)

	val := reflect.Indirect(reflect.ValueOf(what))
	strVal := val.String()
	assert.Equalf(t, with, strVal, "[%s]UnmarshalText: expected %v and %v to be value equal", format, strVal, with)

	b, err = what.MarshalText()
	assert.NoError(t, err)
	assert.Equalf(t, []byte(with), b, "[%s]MarshalText: expected %v and %v to be value equal as []byte", format, string(b), with)

	// Stringer
	strVal = what.String()
	assert.Equalf(t, []byte(with), b, "[%s]String: expected %v and %v to be equal", strVal, with)

	// JSON encoding interface
	bj := []byte("\"" + with + "\"")
	err = what.UnmarshalJSON(bj)
	assert.NoError(t, err)
	val = reflect.Indirect(reflect.ValueOf(what))
	strVal = val.String()
	assert.EqualValuesf(t, with, strVal, "[%s]UnmarshalJSON: expected %v and %v to be value equal", format, strVal, with)

	b, err = what.MarshalJSON()
	assert.NoError(t, err)
	assert.Equalf(t, bj, b, "[%s]MarshalJSON: expected %v and %v to be value equal as []byte", format, string(b), with)

	// validation with Registry
	for _, valid := range append(validSamples, with) {
		testValid(t, format, valid)
	}

	for _, invalid := range invalidSamples {
		testInvalid(t, format, invalid)
	}
}

func testValid(t *testing.T, name, value string) {
	ok := Default.Validates(name, value)
	if !ok {
		t.Errorf("expected %q of type %s to be valid", value, name)
	}
}

func testInvalid(t *testing.T, name, value string) {
	ok := Default.Validates(name, value)
	if ok {
		t.Errorf("expected %q of type %s to be invalid", value, name)
	}
}

func TestDeepCopyBase64(t *testing.T) {
	b64 := Base64("ZWxpemFiZXRocG9zZXk=")
	in := &b64

	out := new(Base64)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *Base64
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyURI(t *testing.T) {
	uri := URI("http://somewhere.com")
	in := &uri

	out := new(URI)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *URI
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyEmail(t *testing.T) {
	email := Email("somebody@somewhere.com")
	in := &email

	out := new(Email)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *Email
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyHostname(t *testing.T) {
	hostname := Hostname("somewhere.com")
	in := &hostname

	out := new(Hostname)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *Hostname
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyIPv4(t *testing.T) {
	ipv4 := IPv4("192.168.254.1")
	in := &ipv4

	out := new(IPv4)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *IPv4
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyIPv6(t *testing.T) {
	ipv6 := IPv6("::1")
	in := &ipv6

	out := new(IPv6)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *IPv6
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyCIDR(t *testing.T) {
	cidr := CIDR("192.0.2.1/24")
	in := &cidr

	out := new(CIDR)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *CIDR
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyMAC(t *testing.T) {
	mac := MAC("01:02:03:04:05:06")
	in := &mac

	out := new(MAC)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *MAC
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyUUID(t *testing.T) {
	first5 := uuid.NewSHA1(uuid.NameSpaceURL, []byte("somewhere.com"))
	uuid := UUID(first5.String())
	in := &uuid

	out := new(UUID)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *UUID
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyUUID3(t *testing.T) {
	first3 := uuid.NewMD5(uuid.NameSpaceURL, []byte("somewhere.com"))
	uuid3 := UUID3(first3.String())
	in := &uuid3

	out := new(UUID3)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *UUID3
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyUUID4(t *testing.T) {
	first4 := uuid.Must(uuid.NewRandom())
	uuid4 := UUID4(first4.String())
	in := &uuid4

	out := new(UUID4)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *UUID4
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyUUID5(t *testing.T) {
	first5 := uuid.NewSHA1(uuid.NameSpaceURL, []byte("somewhere.com"))
	uuid5 := UUID5(first5.String())
	in := &uuid5

	out := new(UUID5)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *UUID5
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyISBN(t *testing.T) {
	isbn := ISBN("0321751043")
	in := &isbn

	out := new(ISBN)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *ISBN
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyISBN10(t *testing.T) {
	isbn10 := ISBN10("0321751043")
	in := &isbn10

	out := new(ISBN10)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *ISBN10
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyISBN13(t *testing.T) {
	isbn13 := ISBN13("978-0321751041")
	in := &isbn13

	out := new(ISBN13)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *ISBN13
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyCreditCard(t *testing.T) {
	creditCard := CreditCard("4111-1111-1111-1111")
	in := &creditCard

	out := new(CreditCard)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *CreditCard
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopySSN(t *testing.T) {
	ssn := SSN("111-11-1111")
	in := &ssn

	out := new(SSN)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *SSN
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyHexColor(t *testing.T) {
	hexColor := HexColor("#FFFFFF")
	in := &hexColor

	out := new(HexColor)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *HexColor
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyRGBColor(t *testing.T) {
	rgbColor := RGBColor("rgb(255,255,255)")
	in := &rgbColor

	out := new(RGBColor)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *RGBColor
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}

func TestDeepCopyPassword(t *testing.T) {
	password := Password("super secret stuff here")
	in := &password

	out := new(Password)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *Password
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}
