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
	"database/sql/driver"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/mail"
	"regexp"
	"strings"

	"github.com/asaskevich/govalidator"
	"go.mongodb.org/mongo-driver/bson"
)

const (
	// HostnamePattern http://json-schema.org/latest/json-schema-validation.html#anchor114
	//  A string instance is valid against this attribute if it is a valid
	//  representation for an Internet host name, as defined by RFC 1034, section 3.1 [RFC1034].
	//  http://tools.ietf.org/html/rfc1034#section-3.5
	//  <digit> ::= any one of the ten digits 0 through 9
	//  var digit = /[0-9]/;
	//  <letter> ::= any one of the 52 alphabetic characters A through Z in upper case and a through z in lower case
	//  var letter = /[a-zA-Z]/;
	//  <let-dig> ::= <letter> | <digit>
	//  var letDig = /[0-9a-zA-Z]/;
	//  <let-dig-hyp> ::= <let-dig> | "-"
	//  var letDigHyp = /[-0-9a-zA-Z]/;
	//  <ldh-str> ::= <let-dig-hyp> | <let-dig-hyp> <ldh-str>
	//  var ldhStr = /[-0-9a-zA-Z]+/;
	//  <label> ::= <letter> [ [ <ldh-str> ] <let-dig> ]
	//  var label = /[a-zA-Z](([-0-9a-zA-Z]+)?[0-9a-zA-Z])?/;
	//  <subdomain> ::= <label> | <subdomain> "." <label>
	//  var subdomain = /^[a-zA-Z](([-0-9a-zA-Z]+)?[0-9a-zA-Z])?(\.[a-zA-Z](([-0-9a-zA-Z]+)?[0-9a-zA-Z])?)*$/;
	//  <domain> ::= <subdomain> | " "
	//
	// Additional validations:
	//   - for FDQNs, top-level domain (e.g. ".com"), is at least to letters long (no special characters here)
	//   - hostnames may start with a digit [RFC1123]
	//   - special registered names with an underscore ('_') are not allowed in this context
	//   - dashes are permitted, but not at the start or the end of a segment
	//   - long top-level domain names (e.g. example.london) are permitted
	//   - symbol unicode points are permitted (e.g. emoji) (not for top-level domain)
	HostnamePattern = `^([a-zA-Z0-9\p{S}\p{L}]((-?[a-zA-Z0-9\p{S}\p{L}]{0,62})?)|([a-zA-Z0-9\p{S}\p{L}](([a-zA-Z0-9-\p{S}\p{L}]{0,61}[a-zA-Z0-9\p{S}\p{L}])?)(\.)){1,}([a-zA-Z\p{L}]){2,63})$`
	// UUIDPattern Regex for UUID that allows uppercase
	UUIDPattern = `(?i)^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$`
	// UUID3Pattern Regex for UUID3 that allows uppercase
	UUID3Pattern = `(?i)^[0-9a-f]{8}-?[0-9a-f]{4}-?3[0-9a-f]{3}-?[0-9a-f]{4}-?[0-9a-f]{12}$`
	// UUID4Pattern Regex for UUID4 that allows uppercase
	UUID4Pattern = `(?i)^[0-9a-f]{8}-?[0-9a-f]{4}-?4[0-9a-f]{3}-?[89ab][0-9a-f]{3}-?[0-9a-f]{12}$`
	// UUID5Pattern Regex for UUID5 that allows uppercase
	UUID5Pattern = `(?i)^[0-9a-f]{8}-?[0-9a-f]{4}-?5[0-9a-f]{3}-?[89ab][0-9a-f]{3}-?[0-9a-f]{12}$`
	// json null type
	jsonNull = "null"
)

var (
	rxHostname = regexp.MustCompile(HostnamePattern)
	rxUUID     = regexp.MustCompile(UUIDPattern)
	rxUUID3    = regexp.MustCompile(UUID3Pattern)
	rxUUID4    = regexp.MustCompile(UUID4Pattern)
	rxUUID5    = regexp.MustCompile(UUID5Pattern)
)

// IsHostname returns true when the string is a valid hostname
func IsHostname(str string) bool {
	if !rxHostname.MatchString(str) {
		return false
	}

	// the sum of all label octets and label lengths is limited to 255.
	if len(str) > 255 {
		return false
	}

	// Each node has a label, which is zero to 63 octets in length
	parts := strings.Split(str, ".")
	valid := true
	for _, p := range parts {
		if len(p) > 63 {
			valid = false
		}
	}
	return valid
}

// IsUUID returns true is the string matches a UUID, upper case is allowed
func IsUUID(str string) bool {
	return rxUUID.MatchString(str)
}

// IsUUID3 returns true is the string matches a UUID, upper case is allowed
func IsUUID3(str string) bool {
	return rxUUID3.MatchString(str)
}

// IsUUID4 returns true is the string matches a UUID, upper case is allowed
func IsUUID4(str string) bool {
	return rxUUID4.MatchString(str)
}

// IsUUID5 returns true is the string matches a UUID, upper case is allowed
func IsUUID5(str string) bool {
	return rxUUID5.MatchString(str)
}

// IsEmail validates an email address.
func IsEmail(str string) bool {
	addr, e := mail.ParseAddress(str)
	return e == nil && addr.Address != ""
}

func init() {
	// register formats in the default registry:
	//   - byte
	//   - creditcard
	//   - email
	//   - hexcolor
	//   - hostname
	//   - ipv4
	//   - ipv6
	//   - cidr
	//   - isbn
	//   - isbn10
	//   - isbn13
	//   - mac
	//   - password
	//   - rgbcolor
	//   - ssn
	//   - uri
	//   - uuid
	//   - uuid3
	//   - uuid4
	//   - uuid5
	u := URI("")
	Default.Add("uri", &u, govalidator.IsRequestURI)

	eml := Email("")
	Default.Add("email", &eml, IsEmail)

	hn := Hostname("")
	Default.Add("hostname", &hn, IsHostname)

	ip4 := IPv4("")
	Default.Add("ipv4", &ip4, govalidator.IsIPv4)

	ip6 := IPv6("")
	Default.Add("ipv6", &ip6, govalidator.IsIPv6)

	cidr := CIDR("")
	Default.Add("cidr", &cidr, govalidator.IsCIDR)

	mac := MAC("")
	Default.Add("mac", &mac, govalidator.IsMAC)

	uid := UUID("")
	Default.Add("uuid", &uid, IsUUID)

	uid3 := UUID3("")
	Default.Add("uuid3", &uid3, IsUUID3)

	uid4 := UUID4("")
	Default.Add("uuid4", &uid4, IsUUID4)

	uid5 := UUID5("")
	Default.Add("uuid5", &uid5, IsUUID5)

	isbn := ISBN("")
	Default.Add("isbn", &isbn, func(str string) bool { return govalidator.IsISBN10(str) || govalidator.IsISBN13(str) })

	isbn10 := ISBN10("")
	Default.Add("isbn10", &isbn10, govalidator.IsISBN10)

	isbn13 := ISBN13("")
	Default.Add("isbn13", &isbn13, govalidator.IsISBN13)

	cc := CreditCard("")
	Default.Add("creditcard", &cc, govalidator.IsCreditCard)

	ssn := SSN("")
	Default.Add("ssn", &ssn, govalidator.IsSSN)

	hc := HexColor("")
	Default.Add("hexcolor", &hc, govalidator.IsHexcolor)

	rc := RGBColor("")
	Default.Add("rgbcolor", &rc, govalidator.IsRGBcolor)

	b64 := Base64([]byte(nil))
	Default.Add("byte", &b64, govalidator.IsBase64)

	pw := Password("")
	Default.Add("password", &pw, func(_ string) bool { return true })
}

// Base64 represents a base64 encoded string, using URLEncoding alphabet
//
// swagger:strfmt byte
type Base64 []byte

// MarshalText turns this instance into text
func (b Base64) MarshalText() ([]byte, error) {
	enc := base64.URLEncoding
	src := []byte(b)
	buf := make([]byte, enc.EncodedLen(len(src)))
	enc.Encode(buf, src)
	return buf, nil
}

// UnmarshalText hydrates this instance from text
func (b *Base64) UnmarshalText(data []byte) error { // validation is performed later on
	enc := base64.URLEncoding
	dbuf := make([]byte, enc.DecodedLen(len(data)))

	n, err := enc.Decode(dbuf, data)
	if err != nil {
		return err
	}

	*b = dbuf[:n]
	return nil
}

// Scan read a value from a database driver
func (b *Base64) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		dbuf := make([]byte, base64.StdEncoding.DecodedLen(len(v)))
		n, err := base64.StdEncoding.Decode(dbuf, v)
		if err != nil {
			return err
		}
		*b = dbuf[:n]
	case string:
		vv, err := base64.StdEncoding.DecodeString(v)
		if err != nil {
			return err
		}
		*b = Base64(vv)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.Base64 from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (b Base64) Value() (driver.Value, error) {
	return driver.Value(b.String()), nil
}

func (b Base64) String() string {
	return base64.StdEncoding.EncodeToString([]byte(b))
}

// MarshalJSON returns the Base64 as JSON
func (b Base64) MarshalJSON() ([]byte, error) {
	return json.Marshal(b.String())
}

// UnmarshalJSON sets the Base64 from JSON
func (b *Base64) UnmarshalJSON(data []byte) error {
	var b64str string
	if err := json.Unmarshal(data, &b64str); err != nil {
		return err
	}
	vb, err := base64.StdEncoding.DecodeString(b64str)
	if err != nil {
		return err
	}
	*b = Base64(vb)
	return nil
}

// MarshalBSON document from this value
func (b Base64) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": b.String()})
}

// UnmarshalBSON document into this value
func (b *Base64) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if bd, ok := m["data"].(string); ok {
		vb, err := base64.StdEncoding.DecodeString(bd)
		if err != nil {
			return err
		}
		*b = Base64(vb)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as base64")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (b *Base64) DeepCopyInto(out *Base64) {
	*out = *b
}

// DeepCopy copies the receiver into a new Base64.
func (b *Base64) DeepCopy() *Base64 {
	if b == nil {
		return nil
	}
	out := new(Base64)
	b.DeepCopyInto(out)
	return out
}

// URI represents the uri string format as specified by the json schema spec
//
// swagger:strfmt uri
type URI string

// MarshalText turns this instance into text
func (u URI) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *URI) UnmarshalText(data []byte) error { // validation is performed later on
	*u = URI(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *URI) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = URI(string(v))
	case string:
		*u = URI(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.URI from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u URI) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u URI) String() string {
	return string(u)
}

// MarshalJSON returns the URI as JSON
func (u URI) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the URI from JSON
func (u *URI) UnmarshalJSON(data []byte) error {
	var uristr string
	if err := json.Unmarshal(data, &uristr); err != nil {
		return err
	}
	*u = URI(uristr)
	return nil
}

// MarshalBSON document from this value
func (u URI) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *URI) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = URI(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as uri")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *URI) DeepCopyInto(out *URI) {
	*out = *u
}

// DeepCopy copies the receiver into a new URI.
func (u *URI) DeepCopy() *URI {
	if u == nil {
		return nil
	}
	out := new(URI)
	u.DeepCopyInto(out)
	return out
}

// Email represents the email string format as specified by the json schema spec
//
// swagger:strfmt email
type Email string

// MarshalText turns this instance into text
func (e Email) MarshalText() ([]byte, error) {
	return []byte(string(e)), nil
}

// UnmarshalText hydrates this instance from text
func (e *Email) UnmarshalText(data []byte) error { // validation is performed later on
	*e = Email(string(data))
	return nil
}

// Scan read a value from a database driver
func (e *Email) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*e = Email(string(v))
	case string:
		*e = Email(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.Email from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (e Email) Value() (driver.Value, error) {
	return driver.Value(string(e)), nil
}

func (e Email) String() string {
	return string(e)
}

// MarshalJSON returns the Email as JSON
func (e Email) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(e))
}

// UnmarshalJSON sets the Email from JSON
func (e *Email) UnmarshalJSON(data []byte) error {
	var estr string
	if err := json.Unmarshal(data, &estr); err != nil {
		return err
	}
	*e = Email(estr)
	return nil
}

// MarshalBSON document from this value
func (e Email) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": e.String()})
}

// UnmarshalBSON document into this value
func (e *Email) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*e = Email(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as email")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (e *Email) DeepCopyInto(out *Email) {
	*out = *e
}

// DeepCopy copies the receiver into a new Email.
func (e *Email) DeepCopy() *Email {
	if e == nil {
		return nil
	}
	out := new(Email)
	e.DeepCopyInto(out)
	return out
}

// Hostname represents the hostname string format as specified by the json schema spec
//
// swagger:strfmt hostname
type Hostname string

// MarshalText turns this instance into text
func (h Hostname) MarshalText() ([]byte, error) {
	return []byte(string(h)), nil
}

// UnmarshalText hydrates this instance from text
func (h *Hostname) UnmarshalText(data []byte) error { // validation is performed later on
	*h = Hostname(string(data))
	return nil
}

// Scan read a value from a database driver
func (h *Hostname) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*h = Hostname(string(v))
	case string:
		*h = Hostname(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.Hostname from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (h Hostname) Value() (driver.Value, error) {
	return driver.Value(string(h)), nil
}

func (h Hostname) String() string {
	return string(h)
}

// MarshalJSON returns the Hostname as JSON
func (h Hostname) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(h))
}

// UnmarshalJSON sets the Hostname from JSON
func (h *Hostname) UnmarshalJSON(data []byte) error {
	var hstr string
	if err := json.Unmarshal(data, &hstr); err != nil {
		return err
	}
	*h = Hostname(hstr)
	return nil
}

// MarshalBSON document from this value
func (h Hostname) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": h.String()})
}

// UnmarshalBSON document into this value
func (h *Hostname) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*h = Hostname(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as hostname")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (h *Hostname) DeepCopyInto(out *Hostname) {
	*out = *h
}

// DeepCopy copies the receiver into a new Hostname.
func (h *Hostname) DeepCopy() *Hostname {
	if h == nil {
		return nil
	}
	out := new(Hostname)
	h.DeepCopyInto(out)
	return out
}

// IPv4 represents an IP v4 address
//
// swagger:strfmt ipv4
type IPv4 string

// MarshalText turns this instance into text
func (u IPv4) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *IPv4) UnmarshalText(data []byte) error { // validation is performed later on
	*u = IPv4(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *IPv4) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = IPv4(string(v))
	case string:
		*u = IPv4(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.IPv4 from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u IPv4) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u IPv4) String() string {
	return string(u)
}

// MarshalJSON returns the IPv4 as JSON
func (u IPv4) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the IPv4 from JSON
func (u *IPv4) UnmarshalJSON(data []byte) error {
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = IPv4(ustr)
	return nil
}

// MarshalBSON document from this value
func (u IPv4) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *IPv4) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = IPv4(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as ipv4")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *IPv4) DeepCopyInto(out *IPv4) {
	*out = *u
}

// DeepCopy copies the receiver into a new IPv4.
func (u *IPv4) DeepCopy() *IPv4 {
	if u == nil {
		return nil
	}
	out := new(IPv4)
	u.DeepCopyInto(out)
	return out
}

// IPv6 represents an IP v6 address
//
// swagger:strfmt ipv6
type IPv6 string

// MarshalText turns this instance into text
func (u IPv6) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *IPv6) UnmarshalText(data []byte) error { // validation is performed later on
	*u = IPv6(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *IPv6) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = IPv6(string(v))
	case string:
		*u = IPv6(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.IPv6 from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u IPv6) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u IPv6) String() string {
	return string(u)
}

// MarshalJSON returns the IPv6 as JSON
func (u IPv6) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the IPv6 from JSON
func (u *IPv6) UnmarshalJSON(data []byte) error {
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = IPv6(ustr)
	return nil
}

// MarshalBSON document from this value
func (u IPv6) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *IPv6) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = IPv6(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as ipv6")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *IPv6) DeepCopyInto(out *IPv6) {
	*out = *u
}

// DeepCopy copies the receiver into a new IPv6.
func (u *IPv6) DeepCopy() *IPv6 {
	if u == nil {
		return nil
	}
	out := new(IPv6)
	u.DeepCopyInto(out)
	return out
}

// CIDR represents a Classless Inter-Domain Routing notation
//
// swagger:strfmt cidr
type CIDR string

// MarshalText turns this instance into text
func (u CIDR) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *CIDR) UnmarshalText(data []byte) error { // validation is performed later on
	*u = CIDR(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *CIDR) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = CIDR(string(v))
	case string:
		*u = CIDR(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.CIDR from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u CIDR) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u CIDR) String() string {
	return string(u)
}

// MarshalJSON returns the CIDR as JSON
func (u CIDR) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the CIDR from JSON
func (u *CIDR) UnmarshalJSON(data []byte) error {
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = CIDR(ustr)
	return nil
}

// MarshalBSON document from this value
func (u CIDR) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *CIDR) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = CIDR(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as CIDR")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *CIDR) DeepCopyInto(out *CIDR) {
	*out = *u
}

// DeepCopy copies the receiver into a new CIDR.
func (u *CIDR) DeepCopy() *CIDR {
	if u == nil {
		return nil
	}
	out := new(CIDR)
	u.DeepCopyInto(out)
	return out
}

// MAC represents a 48 bit MAC address
//
// swagger:strfmt mac
type MAC string

// MarshalText turns this instance into text
func (u MAC) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *MAC) UnmarshalText(data []byte) error { // validation is performed later on
	*u = MAC(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *MAC) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = MAC(string(v))
	case string:
		*u = MAC(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.IPv4 from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u MAC) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u MAC) String() string {
	return string(u)
}

// MarshalJSON returns the MAC as JSON
func (u MAC) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the MAC from JSON
func (u *MAC) UnmarshalJSON(data []byte) error {
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = MAC(ustr)
	return nil
}

// MarshalBSON document from this value
func (u MAC) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *MAC) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = MAC(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as MAC")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *MAC) DeepCopyInto(out *MAC) {
	*out = *u
}

// DeepCopy copies the receiver into a new MAC.
func (u *MAC) DeepCopy() *MAC {
	if u == nil {
		return nil
	}
	out := new(MAC)
	u.DeepCopyInto(out)
	return out
}

// UUID represents a uuid string format
//
// swagger:strfmt uuid
type UUID string

// MarshalText turns this instance into text
func (u UUID) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *UUID) UnmarshalText(data []byte) error { // validation is performed later on
	*u = UUID(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *UUID) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = UUID(string(v))
	case string:
		*u = UUID(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.UUID from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u UUID) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u UUID) String() string {
	return string(u)
}

// MarshalJSON returns the UUID as JSON
func (u UUID) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the UUID from JSON
func (u *UUID) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = UUID(ustr)
	return nil
}

// MarshalBSON document from this value
func (u UUID) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *UUID) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = UUID(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as UUID")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *UUID) DeepCopyInto(out *UUID) {
	*out = *u
}

// DeepCopy copies the receiver into a new UUID.
func (u *UUID) DeepCopy() *UUID {
	if u == nil {
		return nil
	}
	out := new(UUID)
	u.DeepCopyInto(out)
	return out
}

// UUID3 represents a uuid3 string format
//
// swagger:strfmt uuid3
type UUID3 string

// MarshalText turns this instance into text
func (u UUID3) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *UUID3) UnmarshalText(data []byte) error { // validation is performed later on
	*u = UUID3(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *UUID3) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = UUID3(string(v))
	case string:
		*u = UUID3(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.UUID3 from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u UUID3) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u UUID3) String() string {
	return string(u)
}

// MarshalJSON returns the UUID as JSON
func (u UUID3) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the UUID from JSON
func (u *UUID3) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = UUID3(ustr)
	return nil
}

// MarshalBSON document from this value
func (u UUID3) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *UUID3) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = UUID3(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as UUID3")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *UUID3) DeepCopyInto(out *UUID3) {
	*out = *u
}

// DeepCopy copies the receiver into a new UUID3.
func (u *UUID3) DeepCopy() *UUID3 {
	if u == nil {
		return nil
	}
	out := new(UUID3)
	u.DeepCopyInto(out)
	return out
}

// UUID4 represents a uuid4 string format
//
// swagger:strfmt uuid4
type UUID4 string

// MarshalText turns this instance into text
func (u UUID4) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *UUID4) UnmarshalText(data []byte) error { // validation is performed later on
	*u = UUID4(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *UUID4) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = UUID4(string(v))
	case string:
		*u = UUID4(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.UUID4 from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u UUID4) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u UUID4) String() string {
	return string(u)
}

// MarshalJSON returns the UUID as JSON
func (u UUID4) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the UUID from JSON
func (u *UUID4) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = UUID4(ustr)
	return nil
}

// MarshalBSON document from this value
func (u UUID4) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *UUID4) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = UUID4(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as UUID4")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *UUID4) DeepCopyInto(out *UUID4) {
	*out = *u
}

// DeepCopy copies the receiver into a new UUID4.
func (u *UUID4) DeepCopy() *UUID4 {
	if u == nil {
		return nil
	}
	out := new(UUID4)
	u.DeepCopyInto(out)
	return out
}

// UUID5 represents a uuid5 string format
//
// swagger:strfmt uuid5
type UUID5 string

// MarshalText turns this instance into text
func (u UUID5) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *UUID5) UnmarshalText(data []byte) error { // validation is performed later on
	*u = UUID5(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *UUID5) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = UUID5(string(v))
	case string:
		*u = UUID5(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.UUID5 from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u UUID5) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u UUID5) String() string {
	return string(u)
}

// MarshalJSON returns the UUID as JSON
func (u UUID5) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the UUID from JSON
func (u *UUID5) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = UUID5(ustr)
	return nil
}

// MarshalBSON document from this value
func (u UUID5) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *UUID5) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = UUID5(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as UUID5")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *UUID5) DeepCopyInto(out *UUID5) {
	*out = *u
}

// DeepCopy copies the receiver into a new UUID5.
func (u *UUID5) DeepCopy() *UUID5 {
	if u == nil {
		return nil
	}
	out := new(UUID5)
	u.DeepCopyInto(out)
	return out
}

// ISBN represents an isbn string format
//
// swagger:strfmt isbn
type ISBN string

// MarshalText turns this instance into text
func (u ISBN) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *ISBN) UnmarshalText(data []byte) error { // validation is performed later on
	*u = ISBN(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *ISBN) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = ISBN(string(v))
	case string:
		*u = ISBN(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.ISBN from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u ISBN) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u ISBN) String() string {
	return string(u)
}

// MarshalJSON returns the ISBN as JSON
func (u ISBN) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the ISBN from JSON
func (u *ISBN) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = ISBN(ustr)
	return nil
}

// MarshalBSON document from this value
func (u ISBN) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *ISBN) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = ISBN(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as ISBN")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *ISBN) DeepCopyInto(out *ISBN) {
	*out = *u
}

// DeepCopy copies the receiver into a new ISBN.
func (u *ISBN) DeepCopy() *ISBN {
	if u == nil {
		return nil
	}
	out := new(ISBN)
	u.DeepCopyInto(out)
	return out
}

// ISBN10 represents an isbn 10 string format
//
// swagger:strfmt isbn10
type ISBN10 string

// MarshalText turns this instance into text
func (u ISBN10) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *ISBN10) UnmarshalText(data []byte) error { // validation is performed later on
	*u = ISBN10(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *ISBN10) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = ISBN10(string(v))
	case string:
		*u = ISBN10(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.ISBN10 from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u ISBN10) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u ISBN10) String() string {
	return string(u)
}

// MarshalJSON returns the ISBN10 as JSON
func (u ISBN10) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the ISBN10 from JSON
func (u *ISBN10) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = ISBN10(ustr)
	return nil
}

// MarshalBSON document from this value
func (u ISBN10) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *ISBN10) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = ISBN10(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as ISBN10")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *ISBN10) DeepCopyInto(out *ISBN10) {
	*out = *u
}

// DeepCopy copies the receiver into a new ISBN10.
func (u *ISBN10) DeepCopy() *ISBN10 {
	if u == nil {
		return nil
	}
	out := new(ISBN10)
	u.DeepCopyInto(out)
	return out
}

// ISBN13 represents an isbn 13 string format
//
// swagger:strfmt isbn13
type ISBN13 string

// MarshalText turns this instance into text
func (u ISBN13) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *ISBN13) UnmarshalText(data []byte) error { // validation is performed later on
	*u = ISBN13(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *ISBN13) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = ISBN13(string(v))
	case string:
		*u = ISBN13(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.ISBN13 from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u ISBN13) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u ISBN13) String() string {
	return string(u)
}

// MarshalJSON returns the ISBN13 as JSON
func (u ISBN13) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the ISBN13 from JSON
func (u *ISBN13) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = ISBN13(ustr)
	return nil
}

// MarshalBSON document from this value
func (u ISBN13) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *ISBN13) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = ISBN13(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as ISBN13")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *ISBN13) DeepCopyInto(out *ISBN13) {
	*out = *u
}

// DeepCopy copies the receiver into a new ISBN13.
func (u *ISBN13) DeepCopy() *ISBN13 {
	if u == nil {
		return nil
	}
	out := new(ISBN13)
	u.DeepCopyInto(out)
	return out
}

// CreditCard represents a credit card string format
//
// swagger:strfmt creditcard
type CreditCard string

// MarshalText turns this instance into text
func (u CreditCard) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *CreditCard) UnmarshalText(data []byte) error { // validation is performed later on
	*u = CreditCard(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *CreditCard) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = CreditCard(string(v))
	case string:
		*u = CreditCard(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.CreditCard from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u CreditCard) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u CreditCard) String() string {
	return string(u)
}

// MarshalJSON returns the CreditCard as JSON
func (u CreditCard) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the CreditCard from JSON
func (u *CreditCard) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = CreditCard(ustr)
	return nil
}

// MarshalBSON document from this value
func (u CreditCard) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *CreditCard) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = CreditCard(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as CreditCard")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *CreditCard) DeepCopyInto(out *CreditCard) {
	*out = *u
}

// DeepCopy copies the receiver into a new CreditCard.
func (u *CreditCard) DeepCopy() *CreditCard {
	if u == nil {
		return nil
	}
	out := new(CreditCard)
	u.DeepCopyInto(out)
	return out
}

// SSN represents a social security string format
//
// swagger:strfmt ssn
type SSN string

// MarshalText turns this instance into text
func (u SSN) MarshalText() ([]byte, error) {
	return []byte(string(u)), nil
}

// UnmarshalText hydrates this instance from text
func (u *SSN) UnmarshalText(data []byte) error { // validation is performed later on
	*u = SSN(string(data))
	return nil
}

// Scan read a value from a database driver
func (u *SSN) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*u = SSN(string(v))
	case string:
		*u = SSN(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.SSN from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (u SSN) Value() (driver.Value, error) {
	return driver.Value(string(u)), nil
}

func (u SSN) String() string {
	return string(u)
}

// MarshalJSON returns the SSN as JSON
func (u SSN) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(u))
}

// UnmarshalJSON sets the SSN from JSON
func (u *SSN) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*u = SSN(ustr)
	return nil
}

// MarshalBSON document from this value
func (u SSN) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": u.String()})
}

// UnmarshalBSON document into this value
func (u *SSN) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*u = SSN(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as SSN")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (u *SSN) DeepCopyInto(out *SSN) {
	*out = *u
}

// DeepCopy copies the receiver into a new SSN.
func (u *SSN) DeepCopy() *SSN {
	if u == nil {
		return nil
	}
	out := new(SSN)
	u.DeepCopyInto(out)
	return out
}

// HexColor represents a hex color string format
//
// swagger:strfmt hexcolor
type HexColor string

// MarshalText turns this instance into text
func (h HexColor) MarshalText() ([]byte, error) {
	return []byte(string(h)), nil
}

// UnmarshalText hydrates this instance from text
func (h *HexColor) UnmarshalText(data []byte) error { // validation is performed later on
	*h = HexColor(string(data))
	return nil
}

// Scan read a value from a database driver
func (h *HexColor) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*h = HexColor(string(v))
	case string:
		*h = HexColor(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.HexColor from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (h HexColor) Value() (driver.Value, error) {
	return driver.Value(string(h)), nil
}

func (h HexColor) String() string {
	return string(h)
}

// MarshalJSON returns the HexColor as JSON
func (h HexColor) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(h))
}

// UnmarshalJSON sets the HexColor from JSON
func (h *HexColor) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*h = HexColor(ustr)
	return nil
}

// MarshalBSON document from this value
func (h HexColor) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": h.String()})
}

// UnmarshalBSON document into this value
func (h *HexColor) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*h = HexColor(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as HexColor")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (h *HexColor) DeepCopyInto(out *HexColor) {
	*out = *h
}

// DeepCopy copies the receiver into a new HexColor.
func (h *HexColor) DeepCopy() *HexColor {
	if h == nil {
		return nil
	}
	out := new(HexColor)
	h.DeepCopyInto(out)
	return out
}

// RGBColor represents a RGB color string format
//
// swagger:strfmt rgbcolor
type RGBColor string

// MarshalText turns this instance into text
func (r RGBColor) MarshalText() ([]byte, error) {
	return []byte(string(r)), nil
}

// UnmarshalText hydrates this instance from text
func (r *RGBColor) UnmarshalText(data []byte) error { // validation is performed later on
	*r = RGBColor(string(data))
	return nil
}

// Scan read a value from a database driver
func (r *RGBColor) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*r = RGBColor(string(v))
	case string:
		*r = RGBColor(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.RGBColor from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (r RGBColor) Value() (driver.Value, error) {
	return driver.Value(string(r)), nil
}

func (r RGBColor) String() string {
	return string(r)
}

// MarshalJSON returns the RGBColor as JSON
func (r RGBColor) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(r))
}

// UnmarshalJSON sets the RGBColor from JSON
func (r *RGBColor) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*r = RGBColor(ustr)
	return nil
}

// MarshalBSON document from this value
func (r RGBColor) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": r.String()})
}

// UnmarshalBSON document into this value
func (r *RGBColor) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*r = RGBColor(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as RGBColor")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (r *RGBColor) DeepCopyInto(out *RGBColor) {
	*out = *r
}

// DeepCopy copies the receiver into a new RGBColor.
func (r *RGBColor) DeepCopy() *RGBColor {
	if r == nil {
		return nil
	}
	out := new(RGBColor)
	r.DeepCopyInto(out)
	return out
}

// Password represents a password.
// This has no validations and is mainly used as a marker for UI components.
//
// swagger:strfmt password
type Password string

// MarshalText turns this instance into text
func (r Password) MarshalText() ([]byte, error) {
	return []byte(string(r)), nil
}

// UnmarshalText hydrates this instance from text
func (r *Password) UnmarshalText(data []byte) error { // validation is performed later on
	*r = Password(string(data))
	return nil
}

// Scan read a value from a database driver
func (r *Password) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		*r = Password(string(v))
	case string:
		*r = Password(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.Password from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (r Password) Value() (driver.Value, error) {
	return driver.Value(string(r)), nil
}

func (r Password) String() string {
	return string(r)
}

// MarshalJSON returns the Password as JSON
func (r Password) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(r))
}

// UnmarshalJSON sets the Password from JSON
func (r *Password) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*r = Password(ustr)
	return nil
}

// MarshalBSON document from this value
func (r Password) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": r.String()})
}

// UnmarshalBSON document into this value
func (r *Password) UnmarshalBSON(data []byte) error {
	var m bson.M
	if err := bson.Unmarshal(data, &m); err != nil {
		return err
	}

	if ud, ok := m["data"].(string); ok {
		*r = Password(ud)
		return nil
	}
	return errors.New("couldn't unmarshal bson bytes as Password")
}

// DeepCopyInto copies the receiver and writes its value into out.
func (r *Password) DeepCopyInto(out *Password) {
	*out = *r
}

// DeepCopy copies the receiver into a new Password.
func (r *Password) DeepCopy() *Password {
	if r == nil {
		return nil
	}
	out := new(Password)
	r.DeepCopyInto(out)
	return out
}
