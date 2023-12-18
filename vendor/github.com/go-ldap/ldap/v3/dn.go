package ldap

import (
	"bytes"
	enchex "encoding/hex"
	"errors"
	"fmt"
	"strings"

	ber "github.com/go-asn1-ber/asn1-ber"
)

// AttributeTypeAndValue represents an attributeTypeAndValue from https://tools.ietf.org/html/rfc4514
type AttributeTypeAndValue struct {
	// Type is the attribute type
	Type string
	// Value is the attribute value
	Value string
}

// RelativeDN represents a relativeDistinguishedName from https://tools.ietf.org/html/rfc4514
type RelativeDN struct {
	Attributes []*AttributeTypeAndValue
}

// DN represents a distinguishedName from https://tools.ietf.org/html/rfc4514
type DN struct {
	RDNs []*RelativeDN
}

// ParseDN returns a distinguishedName or an error.
// The function respects https://tools.ietf.org/html/rfc4514
func ParseDN(str string) (*DN, error) {
	dn := new(DN)
	dn.RDNs = make([]*RelativeDN, 0)
	rdn := new(RelativeDN)
	rdn.Attributes = make([]*AttributeTypeAndValue, 0)
	buffer := bytes.Buffer{}
	attribute := new(AttributeTypeAndValue)
	escaping := false

	unescapedTrailingSpaces := 0
	stringFromBuffer := func() string {
		s := buffer.String()
		s = s[0 : len(s)-unescapedTrailingSpaces]
		buffer.Reset()
		unescapedTrailingSpaces = 0
		return s
	}

	for i := 0; i < len(str); i++ {
		char := str[i]
		switch {
		case escaping:
			unescapedTrailingSpaces = 0
			escaping = false
			switch char {
			case ' ', '"', '#', '+', ',', ';', '<', '=', '>', '\\':
				buffer.WriteByte(char)
				continue
			}
			// Not a special character, assume hex encoded octet
			if len(str) == i+1 {
				return nil, errors.New("got corrupted escaped character")
			}

			dst := []byte{0}
			n, err := enchex.Decode([]byte(dst), []byte(str[i:i+2]))
			if err != nil {
				return nil, fmt.Errorf("failed to decode escaped character: %s", err)
			} else if n != 1 {
				return nil, fmt.Errorf("expected 1 byte when un-escaping, got %d", n)
			}
			buffer.WriteByte(dst[0])
			i++
		case char == '\\':
			unescapedTrailingSpaces = 0
			escaping = true
		case char == '=':
			attribute.Type = stringFromBuffer()
			// Special case: If the first character in the value is # the
			// following data is BER encoded so we can just fast forward
			// and decode.
			if len(str) > i+1 && str[i+1] == '#' {
				i += 2
				index := strings.IndexAny(str[i:], ",+")
				data := str
				if index > 0 {
					data = str[i : i+index]
				} else {
					data = str[i:]
				}
				rawBER, err := enchex.DecodeString(data)
				if err != nil {
					return nil, fmt.Errorf("failed to decode BER encoding: %s", err)
				}
				packet, err := ber.DecodePacketErr(rawBER)
				if err != nil {
					return nil, fmt.Errorf("failed to decode BER packet: %s", err)
				}
				buffer.WriteString(packet.Data.String())
				i += len(data) - 1
			}
		case char == ',' || char == '+' || char == ';':
			// We're done with this RDN or value, push it
			if len(attribute.Type) == 0 {
				return nil, errors.New("incomplete type, value pair")
			}
			attribute.Value = stringFromBuffer()
			rdn.Attributes = append(rdn.Attributes, attribute)
			attribute = new(AttributeTypeAndValue)
			if char == ',' || char == ';' {
				dn.RDNs = append(dn.RDNs, rdn)
				rdn = new(RelativeDN)
				rdn.Attributes = make([]*AttributeTypeAndValue, 0)
			}
		case char == ' ' && buffer.Len() == 0:
			// ignore unescaped leading spaces
			continue
		default:
			if char == ' ' {
				// Track unescaped spaces in case they are trailing and we need to remove them
				unescapedTrailingSpaces++
			} else {
				// Reset if we see a non-space char
				unescapedTrailingSpaces = 0
			}
			buffer.WriteByte(char)
		}
	}
	if buffer.Len() > 0 {
		if len(attribute.Type) == 0 {
			return nil, errors.New("DN ended with incomplete type, value pair")
		}
		attribute.Value = stringFromBuffer()
		rdn.Attributes = append(rdn.Attributes, attribute)
		dn.RDNs = append(dn.RDNs, rdn)
	}
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

// Equal returns true if the DNs are equal as defined by rfc4517 4.2.15 (distinguishedNameMatch).
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

// Equal returns true if the RelativeDNs are equal as defined by rfc4517 4.2.15 (distinguishedNameMatch).
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
