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
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/asaskevich/govalidator"
	"github.com/globalsign/mgo/bson"
	"github.com/mailru/easyjson/jlexer"
	"github.com/mailru/easyjson/jwriter"
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
	HostnamePattern = `^[a-zA-Z](([-0-9a-zA-Z]+)?[0-9a-zA-Z])?(\.[a-zA-Z](([-0-9a-zA-Z]+)?[0-9a-zA-Z])?)*$`
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

func init() {
	// register formats in the default registry:
	//   - byte
	//   - creditcard
	//   - email
	//   - hexcolor
	//   - hostname
	//   - ipv4
	//   - ipv6
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
	Default.Add("email", &eml, govalidator.IsEmail)

	hn := Hostname("")
	Default.Add("hostname", &hn, IsHostname)

	ip4 := IPv4("")
	Default.Add("ipv4", &ip4, govalidator.IsIPv4)

	ip6 := IPv6("")
	Default.Add("ipv6", &ip6, govalidator.IsIPv6)

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

/* unused:
var formatCheckers = map[string]Validator{
	"byte": govalidator.IsBase64,
}
*/

// Base64 represents a base64 encoded string
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
		*b = Base64(string(v))
	case string:
		*b = Base64(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.Base64 from: %#v", v)
	}

	return nil
}

// Value converts a value to a database driver value
func (b Base64) Value() (driver.Value, error) {
	return driver.Value(string(b)), nil
}

func (b Base64) String() string {
	return string(b)
}

// MarshalJSON returns the Base64 as JSON
func (b Base64) MarshalJSON() ([]byte, error) {
	var w jwriter.Writer
	b.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the Base64 to a easyjson.Writer
func (b Base64) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(base64.StdEncoding.EncodeToString([]byte(b)))
}

// UnmarshalJSON sets the Base64 from JSON
func (b *Base64) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	b.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the Base64 from a easyjson.Lexer
func (b *Base64) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		enc := base64.StdEncoding
		dbuf := make([]byte, enc.DecodedLen(len(data)))

		n, err := enc.Decode(dbuf, []byte(data))
		if err != nil {
			in.AddError(err)
			return
		}

		*b = dbuf[:n]
	}
}

// GetBSON returns the Base64 as a bson.M{} map.
func (b *Base64) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*b)}, nil
}

// SetBSON sets the Base64 from raw bson data
func (b *Base64) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*b = Base64(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as Base64")
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
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the URI to a easyjson.Writer
func (u URI) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the URI from JSON
func (u *URI) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the URI from a easyjson.Lexer
func (u *URI) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = URI(data)
	}
}

// GetBSON returns the URI as a bson.M{} map.
func (u *URI) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the URI from raw bson data
func (u *URI) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = URI(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as URI")
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
	var w jwriter.Writer
	e.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the Email to a easyjson.Writer
func (e Email) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(e))
}

// UnmarshalJSON sets the Email from JSON
func (e *Email) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	e.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the Email from a easyjson.Lexer
func (e *Email) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*e = Email(data)
	}
}

// GetBSON returns the Email as a bson.M{} map.
func (e *Email) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*e)}, nil
}

// SetBSON sets the Email from raw bson data
func (e *Email) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*e = Email(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as Email")
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
	var w jwriter.Writer
	h.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the Hostname to a easyjson.Writer
func (h Hostname) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(h))
}

// UnmarshalJSON sets the Hostname from JSON
func (h *Hostname) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	h.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the Hostname from a easyjson.Lexer
func (h *Hostname) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*h = Hostname(data)
	}
}

// GetBSON returns the Hostname as a bson.M{} map.
func (h *Hostname) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*h)}, nil
}

// SetBSON sets the Hostname from raw bson data
func (h *Hostname) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*h = Hostname(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as Hostname")
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
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the IPv4 to a easyjson.Writer
func (u IPv4) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the IPv4 from JSON
func (u *IPv4) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the IPv4 from a easyjson.Lexer
func (u *IPv4) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = IPv4(data)
	}
}

// GetBSON returns the IPv4 as a bson.M{} map.
func (u *IPv4) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the IPv4 from raw bson data
func (u *IPv4) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = IPv4(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as IPv4")
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
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the IPv6 to a easyjson.Writer
func (u IPv6) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the IPv6 from JSON
func (u *IPv6) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the IPv6 from a easyjson.Lexer
func (u *IPv6) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = IPv6(data)
	}
}

// GetBSON returns the IPv6 as a bson.M{} map.
func (u *IPv6) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the IPv6 from raw bson data
func (u *IPv6) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = IPv6(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as IPv6")
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
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the MAC to a easyjson.Writer
func (u MAC) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the MAC from JSON
func (u *MAC) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the MAC from a easyjson.Lexer
func (u *MAC) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = MAC(data)
	}
}

// GetBSON returns the MAC as a bson.M{} map.
func (u *MAC) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the MAC from raw bson data
func (u *MAC) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = MAC(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as MAC")
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
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the UUID to a easyjson.Writer
func (u UUID) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the UUID from JSON
func (u *UUID) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the UUID from a easyjson.Lexer
func (u *UUID) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = UUID(data)
	}
}

// GetBSON returns the UUID as a bson.M{} map.
func (u *UUID) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the UUID from raw bson data
func (u *UUID) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = UUID(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as UUID")
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

// MarshalJSON returns the UUID3 as JSON
func (u UUID3) MarshalJSON() ([]byte, error) {
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the UUID3 to a easyjson.Writer
func (u UUID3) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the UUID3 from JSON
func (u *UUID3) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the UUID3 from a easyjson.Lexer
func (u *UUID3) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = UUID3(data)
	}
}

// GetBSON returns the UUID3 as a bson.M{} map.
func (u *UUID3) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the UUID3 from raw bson data
func (u *UUID3) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = UUID3(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as UUID3")
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

// MarshalJSON returns the UUID4 as JSON
func (u UUID4) MarshalJSON() ([]byte, error) {
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the UUID4 to a easyjson.Writer
func (u UUID4) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the UUID4 from JSON
func (u *UUID4) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the UUID4 from a easyjson.Lexer
func (u *UUID4) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = UUID4(data)
	}
}

// GetBSON returns the UUID4 as a bson.M{} map.
func (u *UUID4) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the UUID4 from raw bson data
func (u *UUID4) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = UUID4(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as UUID4")
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

// MarshalJSON returns the UUID5 as JSON
func (u UUID5) MarshalJSON() ([]byte, error) {
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the UUID5 to a easyjson.Writer
func (u UUID5) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the UUID5 from JSON
func (u *UUID5) UnmarshalJSON(data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the UUID5 from a easyjson.Lexer
func (u *UUID5) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = UUID5(data)
	}
}

// GetBSON returns the UUID5 as a bson.M{} map.
func (u *UUID5) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the UUID5 from raw bson data
func (u *UUID5) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = UUID5(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as UUID5")
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
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the ISBN to a easyjson.Writer
func (u ISBN) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the ISBN from JSON
func (u *ISBN) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the ISBN from a easyjson.Lexer
func (u *ISBN) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = ISBN(data)
	}
}

// GetBSON returns the ISBN as a bson.M{} map.
func (u *ISBN) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the ISBN from raw bson data
func (u *ISBN) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = ISBN(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as ISBN")
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
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the ISBN10 to a easyjson.Writer
func (u ISBN10) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the ISBN10 from JSON
func (u *ISBN10) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the ISBN10 from a easyjson.Lexer
func (u *ISBN10) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = ISBN10(data)
	}
}

// GetBSON returns the ISBN10 as a bson.M{} map.
func (u *ISBN10) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the ISBN10 from raw bson data
func (u *ISBN10) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = ISBN10(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as ISBN10")
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
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the ISBN13 to a easyjson.Writer
func (u ISBN13) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the ISBN13 from JSON
func (u *ISBN13) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the ISBN13 from a easyjson.Lexer
func (u *ISBN13) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = ISBN13(data)
	}
}

// GetBSON returns the ISBN13 as a bson.M{} map.
func (u *ISBN13) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the ISBN13 from raw bson data
func (u *ISBN13) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = ISBN13(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as ISBN13")
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
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the CreditCard to a easyjson.Writer
func (u CreditCard) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the CreditCard from JSON
func (u *CreditCard) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the CreditCard from a easyjson.Lexer
func (u *CreditCard) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = CreditCard(data)
	}
}

// GetBSON returns the CreditCard as a bson.M{} map.
func (u *CreditCard) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the CreditCard from raw bson data
func (u *CreditCard) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = CreditCard(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as CreditCard")
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
	var w jwriter.Writer
	u.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the SSN to a easyjson.Writer
func (u SSN) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(u))
}

// UnmarshalJSON sets the SSN from JSON
func (u *SSN) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	u.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the SSN from a easyjson.Lexer
func (u *SSN) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*u = SSN(data)
	}
}

// GetBSON returns the SSN as a bson.M{} map.
func (u *SSN) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*u)}, nil
}

// SetBSON sets the SSN from raw bson data
func (u *SSN) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*u = SSN(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as SSN")
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
	var w jwriter.Writer
	h.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the HexColor to a easyjson.Writer
func (h HexColor) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(h))
}

// UnmarshalJSON sets the HexColor from JSON
func (h *HexColor) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	h.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the HexColor from a easyjson.Lexer
func (h *HexColor) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*h = HexColor(data)
	}
}

// GetBSON returns the HexColor as a bson.M{} map.
func (h *HexColor) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*h)}, nil
}

// SetBSON sets the HexColor from raw bson data
func (h *HexColor) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*h = HexColor(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as HexColor")
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
	var w jwriter.Writer
	r.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the RGBColor to a easyjson.Writer
func (r RGBColor) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(r))
}

// UnmarshalJSON sets the RGBColor from JSON
func (r *RGBColor) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	r.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the RGBColor from a easyjson.Lexer
func (r *RGBColor) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*r = RGBColor(data)
	}
}

// GetBSON returns the RGBColor as a bson.M{} map.
func (r *RGBColor) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*r)}, nil
}

// SetBSON sets the RGBColor from raw bson data
func (r *RGBColor) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*r = RGBColor(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as RGBColor")
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
	var w jwriter.Writer
	r.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the Password to a easyjson.Writer
func (r Password) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(string(r))
}

// UnmarshalJSON sets the Password from JSON
func (r *Password) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	r.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the Password from a easyjson.Lexer
func (r *Password) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		*r = Password(data)
	}
}

// GetBSON returns the Password as a bson.M{} map.
func (r *Password) GetBSON() (interface{}, error) {
	return bson.M{"data": string(*r)}, nil
}

// SetBSON sets the Password from raw bson data
func (r *Password) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(string); ok {
		*r = Password(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as Password")
}
