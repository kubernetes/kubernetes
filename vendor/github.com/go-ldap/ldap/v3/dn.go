package ldap

import (
	"encoding/hex"
	"errors"
	"fmt"
	ber "github.com/go-asn1-ber/asn1-ber"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// AttributeTypeAndValue represents an attributeTypeAndValue from https://tools.ietf.org/html/rfc4514
type AttributeTypeAndValue struct {
	// Type is the attribute type
	Type string
	// Value is the attribute value
	Value string
}

func (a *AttributeTypeAndValue) setType(str string) error {
	result, err := decodeString(str)
	if err != nil {
		return err
	}
	a.Type = result

	return nil
}

func (a *AttributeTypeAndValue) setValue(s string) error {
	// https://www.ietf.org/rfc/rfc4514.html#section-2.4
	// If the AttributeType is of the dotted-decimal form, the
	// AttributeValue is represented by an number sign ('#' U+0023)
	// character followed by the hexadecimal encoding of each of the octets
	// of the BER encoding of the X.500 AttributeValue.
	if len(s) > 0 && s[0] == '#' {
		decodedString, err := decodeEncodedString(s[1:])
		if err != nil {
			return err
		}

		a.Value = decodedString
		return nil
	} else {
		decodedString, err := decodeString(s)
		if err != nil {
			return err
		}

		a.Value = decodedString
		return nil
	}
}

// String returns a normalized string representation of this attribute type and
// value pair which is the lowercase join of the Type and Value with a "=".
func (a *AttributeTypeAndValue) String() string {
	return encodeString(foldString(a.Type), false) + "=" + encodeString(a.Value, true)
}

// RelativeDN represents a relativeDistinguishedName from https://tools.ietf.org/html/rfc4514
type RelativeDN struct {
	Attributes []*AttributeTypeAndValue
}

// String returns a normalized string representation of this relative DN which
// is the a join of all attributes (sorted in increasing order) with a "+".
func (r *RelativeDN) String() string {
	attrs := make([]string, len(r.Attributes))
	for i := range r.Attributes {
		attrs[i] = r.Attributes[i].String()
	}
	sort.Strings(attrs)
	return strings.Join(attrs, "+")
}

// DN represents a distinguishedName from https://tools.ietf.org/html/rfc4514
type DN struct {
	RDNs []*RelativeDN
}

// String returns a normalized string representation of this DN which is the
// join of all relative DNs with a ",".
func (d *DN) String() string {
	rdns := make([]string, len(d.RDNs))
	for i := range d.RDNs {
		rdns[i] = d.RDNs[i].String()
	}
	return strings.Join(rdns, ",")
}

func stripLeadingAndTrailingSpaces(inVal string) string {
	noSpaces := strings.Trim(inVal, " ")

	// Re-add the trailing space if it was an escaped space
	if len(noSpaces) > 0 && noSpaces[len(noSpaces)-1] == '\\' && inVal[len(inVal)-1] == ' ' {
		noSpaces = noSpaces + " "
	}

	return noSpaces
}

// Remove leading and trailing spaces from the attribute type and value
// and unescape any escaped characters in these fields
//
// decodeString is based on https://github.com/inteon/cert-manager/blob/ed280d28cd02b262c5db46054d88e70ab518299c/pkg/util/pki/internal/dn.go#L170
func decodeString(str string) (string, error) {
	s := []rune(stripLeadingAndTrailingSpaces(str))

	builder := strings.Builder{}
	for i := 0; i < len(s); i++ {
		char := s[i]

		// If the character is not an escape character, just add it to the
		// builder and continue
		if char != '\\' {
			builder.WriteRune(char)
			continue
		}

		// If the escape character is the last character, it's a corrupted
		// escaped character
		if i+1 >= len(s) {
			return "", fmt.Errorf("got corrupted escaped character: '%s'", string(s))
		}

		// If the escaped character is a special character, just add it to
		// the builder and continue
		switch s[i+1] {
		case ' ', '"', '#', '+', ',', ';', '<', '=', '>', '\\':
			builder.WriteRune(s[i+1])
			i++
			continue
		}

		// If the escaped character is not a special character, it should
		// be a hex-encoded character of the form \XX if it's not at least
		// two characters long, it's a corrupted escaped character
		if i+2 >= len(s) {
			return "", errors.New("failed to decode escaped character: encoding/hex: invalid byte: " + string(s[i+1]))
		}

		// Get the runes for the two characters after the escape character
		// and convert them to a byte slice
		xx := []byte(string(s[i+1 : i+3]))

		// If the two runes are not hex characters and result in more than
		// two bytes when converted to a byte slice, it's a corrupted
		// escaped character
		if len(xx) != 2 {
			return "", errors.New("failed to decode escaped character: invalid byte: " + string(xx))
		}

		// Decode the hex-encoded character and add it to the builder
		dst := []byte{0}
		if n, err := hex.Decode(dst, xx); err != nil {
			return "", errors.New("failed to decode escaped character: " + err.Error())
		} else if n != 1 {
			return "", fmt.Errorf("failed to decode escaped character: encoding/hex: expected 1 byte when un-escaping, got %d", n)
		}

		builder.WriteByte(dst[0])
		i += 2
	}

	return builder.String(), nil
}

// Escape a string according to RFC 4514
func encodeString(value string, isValue bool) string {
	builder := strings.Builder{}

	escapeChar := func(c byte) {
		builder.WriteByte('\\')
		builder.WriteByte(c)
	}

	escapeHex := func(c byte) {
		builder.WriteByte('\\')
		builder.WriteString(hex.EncodeToString([]byte{c}))
	}

	// Loop through each byte and escape as necessary.
	// Runes that take up more than one byte are escaped
	// byte by byte (since both bytes are non-ASCII).
	for i := 0; i < len(value); i++ {
		char := value[i]
		if i == 0 && (char == ' ' || char == '#') {
			// Special case leading space or number sign.
			escapeChar(char)
			continue
		}
		if i == len(value)-1 && char == ' ' {
			// Special case trailing space.
			escapeChar(char)
			continue
		}

		switch char {
		case '"', '+', ',', ';', '<', '>', '\\':
			// Each of these special characters must be escaped.
			escapeChar(char)
			continue
		}

		if !isValue && char == '=' {
			// Equal signs have to be escaped only in the type part of
			// the attribute type and value pair.
			escapeChar(char)
			continue
		}

		if char < ' ' || char > '~' {
			// All special character escapes are handled first
			// above. All bytes less than ASCII SPACE and all bytes
			// greater than ASCII TILDE must be hex-escaped.
			escapeHex(char)
			continue
		}

		// Any other character does not require escaping.
		builder.WriteByte(char)
	}

	return builder.String()
}

func decodeEncodedString(str string) (string, error) {
	decoded, err := hex.DecodeString(str)
	if err != nil {
		return "", fmt.Errorf("failed to decode BER encoding: %w", err)
	}

	packet, err := ber.DecodePacketErr(decoded)
	if err != nil {
		return "", fmt.Errorf("failed to decode BER encoding: %w", err)
	}

	return packet.Data.String(), nil
}

// ParseDN returns a distinguishedName or an error.
// The function respects https://tools.ietf.org/html/rfc4514
func ParseDN(str string) (*DN, error) {
	var dn = &DN{RDNs: make([]*RelativeDN, 0)}
	if strings.TrimSpace(str) == "" {
		return dn, nil
	}

	var (
		rdn                   = &RelativeDN{}
		attr                  = &AttributeTypeAndValue{}
		escaping              bool
		startPos              int
		appendAttributesToRDN = func(end bool) {
			rdn.Attributes = append(rdn.Attributes, attr)
			attr = &AttributeTypeAndValue{}
			if end {
				dn.RDNs = append(dn.RDNs, rdn)
				rdn = &RelativeDN{}
			}
		}
	)

	// Loop through each character in the string and
	// build up the attribute type and value pairs.
	// We only check for ascii characters here, which
	// allows us to iterate over the string byte by byte.
	for i := 0; i < len(str); i++ {
		char := str[i]
		switch {
		case escaping:
			escaping = false
		case char == '\\':
			escaping = true
		case char == '=' && len(attr.Type) == 0:
			if err := attr.setType(str[startPos:i]); err != nil {
				return nil, err
			}
			startPos = i + 1
		case char == ',' || char == '+' || char == ';':
			if len(attr.Type) == 0 {
				return dn, errors.New("incomplete type, value pair")
			}
			if err := attr.setValue(str[startPos:i]); err != nil {
				return nil, err
			}

			startPos = i + 1
			last := char == ',' || char == ';'
			appendAttributesToRDN(last)
		}
	}

	if len(attr.Type) == 0 {
		return dn, errors.New("DN ended with incomplete type, value pair")
	}

	if err := attr.setValue(str[startPos:]); err != nil {
		return dn, err
	}
	appendAttributesToRDN(true)

	return dn, nil
}

// Equal returns true if the DNs are equal as defined by rfc4517 4.2.15 (distinguishedNameMatch).
// Returns true if they have the same number of relative distinguished names
// and corresponding relative distinguished names (by position) are the same.
func (d *DN) Equal(other *DN) bool {
	if len(d.RDNs) != len(other.RDNs) {
		return false
	}
	for i := range d.RDNs {
		if !d.RDNs[i].Equal(other.RDNs[i]) {
			return false
		}
	}
	return true
}

// AncestorOf returns true if the other DN consists of at least one RDN followed by all the RDNs of the current DN.
// "ou=widgets,o=acme.com" is an ancestor of "ou=sprockets,ou=widgets,o=acme.com"
// "ou=widgets,o=acme.com" is not an ancestor of "ou=sprockets,ou=widgets,o=foo.com"
// "ou=widgets,o=acme.com" is not an ancestor of "ou=widgets,o=acme.com"
func (d *DN) AncestorOf(other *DN) bool {
	if len(d.RDNs) >= len(other.RDNs) {
		return false
	}
	// Take the last `len(d.RDNs)` RDNs from the other DN to compare against
	otherRDNs := other.RDNs[len(other.RDNs)-len(d.RDNs):]
	for i := range d.RDNs {
		if !d.RDNs[i].Equal(otherRDNs[i]) {
			return false
		}
	}
	return true
}

// Equal returns true if the RelativeDNs are equal as defined by rfc4517 4.2.15 (distinguishedNameMatch).
// Relative distinguished names are the same if and only if they have the same number of AttributeTypeAndValues
// and each attribute of the first RDN is the same as the attribute of the second RDN with the same attribute type.
// The order of attributes is not significant.
// Case of attribute types is not significant.
func (r *RelativeDN) Equal(other *RelativeDN) bool {
	if len(r.Attributes) != len(other.Attributes) {
		return false
	}
	return r.hasAllAttributes(other.Attributes) && other.hasAllAttributes(r.Attributes)
}

func (r *RelativeDN) hasAllAttributes(attrs []*AttributeTypeAndValue) bool {
	for _, attr := range attrs {
		found := false
		for _, myattr := range r.Attributes {
			if myattr.Equal(attr) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// Equal returns true if the AttributeTypeAndValue is equivalent to the specified AttributeTypeAndValue
// Case of the attribute type is not significant
func (a *AttributeTypeAndValue) Equal(other *AttributeTypeAndValue) bool {
	return strings.EqualFold(a.Type, other.Type) && a.Value == other.Value
}

// EqualFold returns true if the DNs are equal as defined by rfc4517 4.2.15 (distinguishedNameMatch).
// Returns true if they have the same number of relative distinguished names
// and corresponding relative distinguished names (by position) are the same.
// Case of the attribute type and value is not significant
func (d *DN) EqualFold(other *DN) bool {
	if len(d.RDNs) != len(other.RDNs) {
		return false
	}
	for i := range d.RDNs {
		if !d.RDNs[i].EqualFold(other.RDNs[i]) {
			return false
		}
	}
	return true
}

// AncestorOfFold returns true if the other DN consists of at least one RDN followed by all the RDNs of the current DN.
// Case of the attribute type and value is not significant
func (d *DN) AncestorOfFold(other *DN) bool {
	if len(d.RDNs) >= len(other.RDNs) {
		return false
	}
	// Take the last `len(d.RDNs)` RDNs from the other DN to compare against
	otherRDNs := other.RDNs[len(other.RDNs)-len(d.RDNs):]
	for i := range d.RDNs {
		if !d.RDNs[i].EqualFold(otherRDNs[i]) {
			return false
		}
	}
	return true
}

// EqualFold returns true if the RelativeDNs are equal as defined by rfc4517 4.2.15 (distinguishedNameMatch).
// Case of the attribute type is not significant
func (r *RelativeDN) EqualFold(other *RelativeDN) bool {
	if len(r.Attributes) != len(other.Attributes) {
		return false
	}
	return r.hasAllAttributesFold(other.Attributes) && other.hasAllAttributesFold(r.Attributes)
}

func (r *RelativeDN) hasAllAttributesFold(attrs []*AttributeTypeAndValue) bool {
	for _, attr := range attrs {
		found := false
		for _, myattr := range r.Attributes {
			if myattr.EqualFold(attr) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// EqualFold returns true if the AttributeTypeAndValue is equivalent to the specified AttributeTypeAndValue
// Case of the attribute type and value is not significant
func (a *AttributeTypeAndValue) EqualFold(other *AttributeTypeAndValue) bool {
	return strings.EqualFold(a.Type, other.Type) && strings.EqualFold(a.Value, other.Value)
}

// foldString returns a folded string such that foldString(x) == foldString(y)
// is identical to bytes.EqualFold(x, y).
// based on https://go.dev/src/encoding/json/fold.go
func foldString(s string) string {
	builder := strings.Builder{}
	for _, char := range s {
		// Handle single-byte ASCII.
		if char < utf8.RuneSelf {
			if 'A' <= char && char <= 'Z' {
				char += 'a' - 'A'
			}
			builder.WriteRune(char)
			continue
		}

		builder.WriteRune(foldRune(char))
	}
	return builder.String()
}

// foldRune is returns the smallest rune for all runes in the same fold set.
func foldRune(r rune) rune {
	for {
		r2 := unicode.SimpleFold(r)
		if r2 <= r {
			return r
		}
		r = r2
	}
}
