// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package baggage // import "go.opentelemetry.io/otel/baggage"

import (
	"errors"
	"fmt"
	"net/url"
	"strings"

	"go.opentelemetry.io/otel/internal/baggage"
)

const (
	maxMembers               = 180
	maxBytesPerMembers       = 4096
	maxBytesPerBaggageString = 8192

	listDelimiter     = ","
	keyValueDelimiter = "="
	propertyDelimiter = ";"
)

var (
	errInvalidKey      = errors.New("invalid key")
	errInvalidValue    = errors.New("invalid value")
	errInvalidProperty = errors.New("invalid baggage list-member property")
	errInvalidMember   = errors.New("invalid baggage list-member")
	errMemberNumber    = errors.New("too many list-members in baggage-string")
	errMemberBytes     = errors.New("list-member too large")
	errBaggageBytes    = errors.New("baggage-string too large")
)

// Property is an additional metadata entry for a baggage list-member.
type Property struct {
	key, value string

	// hasValue indicates if a zero-value value means the property does not
	// have a value or if it was the zero-value.
	hasValue bool
}

// NewKeyProperty returns a new Property for key.
//
// If key is invalid, an error will be returned.
func NewKeyProperty(key string) (Property, error) {
	if !validateKey(key) {
		return newInvalidProperty(), fmt.Errorf("%w: %q", errInvalidKey, key)
	}

	p := Property{key: key}
	return p, nil
}

// NewKeyValueProperty returns a new Property for key with value.
//
// The passed key must be compliant with W3C Baggage specification.
// The passed value must be precent-encoded as defined in W3C Baggage specification.
//
// Notice: Consider using [NewKeyValuePropertyRaw] instead
// that does not require precent-encoding of the value.
func NewKeyValueProperty(key, value string) (Property, error) {
	if !validateValue(value) {
		return newInvalidProperty(), fmt.Errorf("%w: %q", errInvalidValue, value)
	}
	decodedValue, err := url.PathUnescape(value)
	if err != nil {
		return newInvalidProperty(), fmt.Errorf("%w: %q", errInvalidValue, value)
	}
	return NewKeyValuePropertyRaw(key, decodedValue)
}

// NewKeyValuePropertyRaw returns a new Property for key with value.
//
// The passed key must be compliant with W3C Baggage specification.
func NewKeyValuePropertyRaw(key, value string) (Property, error) {
	if !validateKey(key) {
		return newInvalidProperty(), fmt.Errorf("%w: %q", errInvalidKey, key)
	}

	p := Property{
		key:      key,
		value:    value,
		hasValue: true,
	}
	return p, nil
}

func newInvalidProperty() Property {
	return Property{}
}

// parseProperty attempts to decode a Property from the passed string. It
// returns an error if the input is invalid according to the W3C Baggage
// specification.
func parseProperty(property string) (Property, error) {
	if property == "" {
		return newInvalidProperty(), nil
	}

	p, ok := parsePropertyInternal(property)
	if !ok {
		return newInvalidProperty(), fmt.Errorf("%w: %q", errInvalidProperty, property)
	}

	return p, nil
}

// validate ensures p conforms to the W3C Baggage specification, returning an
// error otherwise.
func (p Property) validate() error {
	errFunc := func(err error) error {
		return fmt.Errorf("invalid property: %w", err)
	}

	if !validateKey(p.key) {
		return errFunc(fmt.Errorf("%w: %q", errInvalidKey, p.key))
	}
	if !p.hasValue && p.value != "" {
		return errFunc(errors.New("inconsistent value"))
	}
	return nil
}

// Key returns the Property key.
func (p Property) Key() string {
	return p.key
}

// Value returns the Property value. Additionally, a boolean value is returned
// indicating if the returned value is the empty if the Property has a value
// that is empty or if the value is not set.
func (p Property) Value() (string, bool) {
	return p.value, p.hasValue
}

// String encodes Property into a header string compliant with the W3C Baggage
// specification.
func (p Property) String() string {
	if p.hasValue {
		return fmt.Sprintf("%s%s%v", p.key, keyValueDelimiter, valueEscape(p.value))
	}
	return p.key
}

type properties []Property

func fromInternalProperties(iProps []baggage.Property) properties {
	if len(iProps) == 0 {
		return nil
	}

	props := make(properties, len(iProps))
	for i, p := range iProps {
		props[i] = Property{
			key:      p.Key,
			value:    p.Value,
			hasValue: p.HasValue,
		}
	}
	return props
}

func (p properties) asInternal() []baggage.Property {
	if len(p) == 0 {
		return nil
	}

	iProps := make([]baggage.Property, len(p))
	for i, prop := range p {
		iProps[i] = baggage.Property{
			Key:      prop.key,
			Value:    prop.value,
			HasValue: prop.hasValue,
		}
	}
	return iProps
}

func (p properties) Copy() properties {
	if len(p) == 0 {
		return nil
	}

	props := make(properties, len(p))
	copy(props, p)
	return props
}

// validate ensures each Property in p conforms to the W3C Baggage
// specification, returning an error otherwise.
func (p properties) validate() error {
	for _, prop := range p {
		if err := prop.validate(); err != nil {
			return err
		}
	}
	return nil
}

// String encodes properties into a header string compliant with the W3C Baggage
// specification.
func (p properties) String() string {
	props := make([]string, len(p))
	for i, prop := range p {
		props[i] = prop.String()
	}
	return strings.Join(props, propertyDelimiter)
}

// Member is a list-member of a baggage-string as defined by the W3C Baggage
// specification.
type Member struct {
	key, value string
	properties properties

	// hasData indicates whether the created property contains data or not.
	// Properties that do not contain data are invalid with no other check
	// required.
	hasData bool
}

// NewMemberRaw returns a new Member from the passed arguments.
//
// The passed key must be compliant with W3C Baggage specification.
// The passed value must be precent-encoded as defined in W3C Baggage specification.
//
// Notice: Consider using [NewMemberRaw] instead
// that does not require precent-encoding of the value.
func NewMember(key, value string, props ...Property) (Member, error) {
	if !validateValue(value) {
		return newInvalidMember(), fmt.Errorf("%w: %q", errInvalidValue, value)
	}
	decodedValue, err := url.PathUnescape(value)
	if err != nil {
		return newInvalidMember(), fmt.Errorf("%w: %q", errInvalidValue, value)
	}
	return NewMemberRaw(key, decodedValue, props...)
}

// NewMemberRaw returns a new Member from the passed arguments.
//
// The passed key must be compliant with W3C Baggage specification.
func NewMemberRaw(key, value string, props ...Property) (Member, error) {
	m := Member{
		key:        key,
		value:      value,
		properties: properties(props).Copy(),
		hasData:    true,
	}
	if err := m.validate(); err != nil {
		return newInvalidMember(), err
	}
	return m, nil
}

func newInvalidMember() Member {
	return Member{}
}

// parseMember attempts to decode a Member from the passed string. It returns
// an error if the input is invalid according to the W3C Baggage
// specification.
func parseMember(member string) (Member, error) {
	if n := len(member); n > maxBytesPerMembers {
		return newInvalidMember(), fmt.Errorf("%w: %d", errMemberBytes, n)
	}

	var props properties
	keyValue, properties, found := strings.Cut(member, propertyDelimiter)
	if found {
		// Parse the member properties.
		for _, pStr := range strings.Split(properties, propertyDelimiter) {
			p, err := parseProperty(pStr)
			if err != nil {
				return newInvalidMember(), err
			}
			props = append(props, p)
		}
	}
	// Parse the member key/value pair.

	// Take into account a value can contain equal signs (=).
	k, v, found := strings.Cut(keyValue, keyValueDelimiter)
	if !found {
		return newInvalidMember(), fmt.Errorf("%w: %q", errInvalidMember, member)
	}
	// "Leading and trailing whitespaces are allowed but MUST be trimmed
	// when converting the header into a data structure."
	key := strings.TrimSpace(k)
	if !validateKey(key) {
		return newInvalidMember(), fmt.Errorf("%w: %q", errInvalidKey, key)
	}

	val := strings.TrimSpace(v)
	if !validateValue(val) {
		return newInvalidMember(), fmt.Errorf("%w: %q", errInvalidValue, v)
	}

	// Decode a precent-encoded value.
	value, err := url.PathUnescape(val)
	if err != nil {
		return newInvalidMember(), fmt.Errorf("%w: %v", errInvalidValue, err)
	}
	return Member{key: key, value: value, properties: props, hasData: true}, nil
}

// validate ensures m conforms to the W3C Baggage specification.
// A key must be an ASCII string, returning an error otherwise.
func (m Member) validate() error {
	if !m.hasData {
		return fmt.Errorf("%w: %q", errInvalidMember, m)
	}

	if !validateKey(m.key) {
		return fmt.Errorf("%w: %q", errInvalidKey, m.key)
	}
	return m.properties.validate()
}

// Key returns the Member key.
func (m Member) Key() string { return m.key }

// Value returns the Member value.
func (m Member) Value() string { return m.value }

// Properties returns a copy of the Member properties.
func (m Member) Properties() []Property { return m.properties.Copy() }

// String encodes Member into a header string compliant with the W3C Baggage
// specification.
func (m Member) String() string {
	// A key is just an ASCII string. A value is restricted to be
	// US-ASCII characters excluding CTLs, whitespace,
	// DQUOTE, comma, semicolon, and backslash.
	s := fmt.Sprintf("%s%s%s", m.key, keyValueDelimiter, valueEscape(m.value))
	if len(m.properties) > 0 {
		s = fmt.Sprintf("%s%s%s", s, propertyDelimiter, m.properties.String())
	}
	return s
}

// Baggage is a list of baggage members representing the baggage-string as
// defined by the W3C Baggage specification.
type Baggage struct { //nolint:golint
	list baggage.List
}

// New returns a new valid Baggage. It returns an error if it results in a
// Baggage exceeding limits set in that specification.
//
// It expects all the provided members to have already been validated.
func New(members ...Member) (Baggage, error) {
	if len(members) == 0 {
		return Baggage{}, nil
	}

	b := make(baggage.List)
	for _, m := range members {
		if !m.hasData {
			return Baggage{}, errInvalidMember
		}

		// OpenTelemetry resolves duplicates by last-one-wins.
		b[m.key] = baggage.Item{
			Value:      m.value,
			Properties: m.properties.asInternal(),
		}
	}

	// Check member numbers after deduplication.
	if len(b) > maxMembers {
		return Baggage{}, errMemberNumber
	}

	bag := Baggage{b}
	if n := len(bag.String()); n > maxBytesPerBaggageString {
		return Baggage{}, fmt.Errorf("%w: %d", errBaggageBytes, n)
	}

	return bag, nil
}

// Parse attempts to decode a baggage-string from the passed string. It
// returns an error if the input is invalid according to the W3C Baggage
// specification.
//
// If there are duplicate list-members contained in baggage, the last one
// defined (reading left-to-right) will be the only one kept. This diverges
// from the W3C Baggage specification which allows duplicate list-members, but
// conforms to the OpenTelemetry Baggage specification.
func Parse(bStr string) (Baggage, error) {
	if bStr == "" {
		return Baggage{}, nil
	}

	if n := len(bStr); n > maxBytesPerBaggageString {
		return Baggage{}, fmt.Errorf("%w: %d", errBaggageBytes, n)
	}

	b := make(baggage.List)
	for _, memberStr := range strings.Split(bStr, listDelimiter) {
		m, err := parseMember(memberStr)
		if err != nil {
			return Baggage{}, err
		}
		// OpenTelemetry resolves duplicates by last-one-wins.
		b[m.key] = baggage.Item{
			Value:      m.value,
			Properties: m.properties.asInternal(),
		}
	}

	// OpenTelemetry does not allow for duplicate list-members, but the W3C
	// specification does. Now that we have deduplicated, ensure the baggage
	// does not exceed list-member limits.
	if len(b) > maxMembers {
		return Baggage{}, errMemberNumber
	}

	return Baggage{b}, nil
}

// Member returns the baggage list-member identified by key.
//
// If there is no list-member matching the passed key the returned Member will
// be a zero-value Member.
// The returned member is not validated, as we assume the validation happened
// when it was added to the Baggage.
func (b Baggage) Member(key string) Member {
	v, ok := b.list[key]
	if !ok {
		// We do not need to worry about distinguishing between the situation
		// where a zero-valued Member is included in the Baggage because a
		// zero-valued Member is invalid according to the W3C Baggage
		// specification (it has an empty key).
		return newInvalidMember()
	}

	return Member{
		key:        key,
		value:      v.Value,
		properties: fromInternalProperties(v.Properties),
		hasData:    true,
	}
}

// Members returns all the baggage list-members.
// The order of the returned list-members does not have significance.
//
// The returned members are not validated, as we assume the validation happened
// when they were added to the Baggage.
func (b Baggage) Members() []Member {
	if len(b.list) == 0 {
		return nil
	}

	members := make([]Member, 0, len(b.list))
	for k, v := range b.list {
		members = append(members, Member{
			key:        k,
			value:      v.Value,
			properties: fromInternalProperties(v.Properties),
			hasData:    true,
		})
	}
	return members
}

// SetMember returns a copy the Baggage with the member included. If the
// baggage contains a Member with the same key the existing Member is
// replaced.
//
// If member is invalid according to the W3C Baggage specification, an error
// is returned with the original Baggage.
func (b Baggage) SetMember(member Member) (Baggage, error) {
	if !member.hasData {
		return b, errInvalidMember
	}

	n := len(b.list)
	if _, ok := b.list[member.key]; !ok {
		n++
	}
	list := make(baggage.List, n)

	for k, v := range b.list {
		// Do not copy if we are just going to overwrite.
		if k == member.key {
			continue
		}
		list[k] = v
	}

	list[member.key] = baggage.Item{
		Value:      member.value,
		Properties: member.properties.asInternal(),
	}

	return Baggage{list: list}, nil
}

// DeleteMember returns a copy of the Baggage with the list-member identified
// by key removed.
func (b Baggage) DeleteMember(key string) Baggage {
	n := len(b.list)
	if _, ok := b.list[key]; ok {
		n--
	}
	list := make(baggage.List, n)

	for k, v := range b.list {
		if k == key {
			continue
		}
		list[k] = v
	}

	return Baggage{list: list}
}

// Len returns the number of list-members in the Baggage.
func (b Baggage) Len() int {
	return len(b.list)
}

// String encodes Baggage into a header string compliant with the W3C Baggage
// specification.
func (b Baggage) String() string {
	members := make([]string, 0, len(b.list))
	for k, v := range b.list {
		members = append(members, Member{
			key:        k,
			value:      v.Value,
			properties: fromInternalProperties(v.Properties),
		}.String())
	}
	return strings.Join(members, listDelimiter)
}

// parsePropertyInternal attempts to decode a Property from the passed string.
// It follows the spec at https://www.w3.org/TR/baggage/#definition.
func parsePropertyInternal(s string) (p Property, ok bool) {
	// For the entire function we will use "   key    =    value  " as an example.
	// Attempting to parse the key.
	// First skip spaces at the beginning "<   >key    =    value  " (they could be empty).
	index := skipSpace(s, 0)

	// Parse the key: "   <key>    =    value  ".
	keyStart := index
	keyEnd := index
	for _, c := range s[keyStart:] {
		if !validateKeyChar(c) {
			break
		}
		keyEnd++
	}

	// If we couldn't find any valid key character,
	// it means the key is either empty or invalid.
	if keyStart == keyEnd {
		return
	}

	// Skip spaces after the key: "   key<    >=    value  ".
	index = skipSpace(s, keyEnd)

	if index == len(s) {
		// A key can have no value, like: "   key    ".
		ok = true
		p.key = s[keyStart:keyEnd]
		return
	}

	// If we have not reached the end and we can't find the '=' delimiter,
	// it means the property is invalid.
	if s[index] != keyValueDelimiter[0] {
		return
	}

	// Attempting to parse the value.
	// Match: "   key    =<    >value  ".
	index = skipSpace(s, index+1)

	// Match the value string: "   key    =    <value>  ".
	// A valid property can be: "   key    =".
	// Therefore, we don't have to check if the value is empty.
	valueStart := index
	valueEnd := index
	for _, c := range s[valueStart:] {
		if !validateValueChar(c) {
			break
		}
		valueEnd++
	}

	// Skip all trailing whitespaces: "   key    =    value<  >".
	index = skipSpace(s, valueEnd)

	// If after looking for the value and skipping whitespaces
	// we have not reached the end, it means the property is
	// invalid, something like: "   key    =    value  value1".
	if index != len(s) {
		return
	}

	// Decode a precent-encoded value.
	value, err := url.PathUnescape(s[valueStart:valueEnd])
	if err != nil {
		return
	}

	ok = true
	p.key = s[keyStart:keyEnd]
	p.hasValue = true

	p.value = value
	return
}

func skipSpace(s string, offset int) int {
	i := offset
	for ; i < len(s); i++ {
		c := s[i]
		if c != ' ' && c != '\t' {
			break
		}
	}
	return i
}

func validateKey(s string) bool {
	if len(s) == 0 {
		return false
	}

	for _, c := range s {
		if !validateKeyChar(c) {
			return false
		}
	}

	return true
}

func validateKeyChar(c int32) bool {
	return (c >= 0x23 && c <= 0x27) ||
		(c >= 0x30 && c <= 0x39) ||
		(c >= 0x41 && c <= 0x5a) ||
		(c >= 0x5e && c <= 0x7a) ||
		c == 0x21 ||
		c == 0x2a ||
		c == 0x2b ||
		c == 0x2d ||
		c == 0x2e ||
		c == 0x7c ||
		c == 0x7e
}

func validateValue(s string) bool {
	for _, c := range s {
		if !validateValueChar(c) {
			return false
		}
	}

	return true
}

func validateValueChar(c int32) bool {
	return c == 0x21 ||
		(c >= 0x23 && c <= 0x2b) ||
		(c >= 0x2d && c <= 0x3a) ||
		(c >= 0x3c && c <= 0x5b) ||
		(c >= 0x5d && c <= 0x7e)
}

// valueEscape escapes the string so it can be safely placed inside a baggage value,
// replacing special characters with %XX sequences as needed.
//
// The implementation is based on:
// https://github.com/golang/go/blob/f6509cf5cdbb5787061b784973782933c47f1782/src/net/url/url.go#L285.
func valueEscape(s string) string {
	hexCount := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if shouldEscape(c) {
			hexCount++
		}
	}

	if hexCount == 0 {
		return s
	}

	var buf [64]byte
	var t []byte

	required := len(s) + 2*hexCount
	if required <= len(buf) {
		t = buf[:required]
	} else {
		t = make([]byte, required)
	}

	j := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if shouldEscape(s[i]) {
			const upperhex = "0123456789ABCDEF"
			t[j] = '%'
			t[j+1] = upperhex[c>>4]
			t[j+2] = upperhex[c&15]
			j += 3
		} else {
			t[j] = c
			j++
		}
	}

	return string(t)
}

// shouldEscape returns true if the specified byte should be escaped when
// appearing in a baggage value string.
func shouldEscape(c byte) bool {
	if c == '%' {
		// The percent character must be encoded so that percent-encoding can work.
		return true
	}
	return !validateValueChar(int32(c))
}
