// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// File contains DN parsing functionality
//
// https://tools.ietf.org/html/rfc4514
//
//   distinguishedName = [ relativeDistinguishedName
//         *( COMMA relativeDistinguishedName ) ]
//     relativeDistinguishedName = attributeTypeAndValue
//         *( PLUS attributeTypeAndValue )
//     attributeTypeAndValue = attributeType EQUALS attributeValue
//     attributeType = descr / numericoid
//     attributeValue = string / hexstring
//
//     ; The following characters are to be escaped when they appear
//     ; in the value to be encoded: ESC, one of <escaped>, leading
//     ; SHARP or SPACE, trailing SPACE, and NULL.
//     string =   [ ( leadchar / pair ) [ *( stringchar / pair )
//        ( trailchar / pair ) ] ]
//
//     leadchar = LUTF1 / UTFMB
//     LUTF1 = %x01-1F / %x21 / %x24-2A / %x2D-3A /
//        %x3D / %x3F-5B / %x5D-7F
//
//     trailchar  = TUTF1 / UTFMB
//     TUTF1 = %x01-1F / %x21 / %x23-2A / %x2D-3A /
//        %x3D / %x3F-5B / %x5D-7F
//
//     stringchar = SUTF1 / UTFMB
//     SUTF1 = %x01-21 / %x23-2A / %x2D-3A /
//        %x3D / %x3F-5B / %x5D-7F
//
//     pair = ESC ( ESC / special / hexpair )
//     special = escaped / SPACE / SHARP / EQUALS
//     escaped = DQUOTE / PLUS / COMMA / SEMI / LANGLE / RANGLE
//     hexstring = SHARP 1*hexpair
//     hexpair = HEX HEX
//
//  where the productions <descr>, <numericoid>, <COMMA>, <DQUOTE>,
//  <EQUALS>, <ESC>, <HEX>, <LANGLE>, <NULL>, <PLUS>, <RANGLE>, <SEMI>,
//  <SPACE>, <SHARP>, and <UTFMB> are defined in [RFC4512].
//

package ldap

import (
	"bytes"
	enchex "encoding/hex"
	"errors"
	"fmt"
	"strings"

	"gopkg.in/asn1-ber.v1"
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

// ParseDN returns a distinguishedName or an error
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
		if escaping {
			unescapedTrailingSpaces = 0
			escaping = false
			switch char {
			case ' ', '"', '#', '+', ',', ';', '<', '=', '>', '\\':
				buffer.WriteByte(char)
				continue
			}
			// Not a special character, assume hex encoded octet
			if len(str) == i+1 {
				return nil, errors.New("Got corrupted escaped character")
			}

			dst := []byte{0}
			n, err := enchex.Decode([]byte(dst), []byte(str[i:i+2]))
			if err != nil {
				return nil, fmt.Errorf("Failed to decode escaped character: %s", err)
			} else if n != 1 {
				return nil, fmt.Errorf("Expected 1 byte when un-escaping, got %d", n)
			}
			buffer.WriteByte(dst[0])
			i++
		} else if char == '\\' {
			unescapedTrailingSpaces = 0
			escaping = true
		} else if char == '=' {
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
					return nil, fmt.Errorf("Failed to decode BER encoding: %s", err)
				}
				packet := ber.DecodePacket(rawBER)
				buffer.WriteString(packet.Data.String())
				i += len(data) - 1
			}
		} else if char == ',' || char == '+' {
			// We're done with this RDN or value, push it
			if len(attribute.Type) == 0 {
				return nil, errors.New("incomplete type, value pair")
			}
			attribute.Value = stringFromBuffer()
			rdn.Attributes = append(rdn.Attributes, attribute)
			attribute = new(AttributeTypeAndValue)
			if char == ',' {
				dn.RDNs = append(dn.RDNs, rdn)
				rdn = new(RelativeDN)
				rdn.Attributes = make([]*AttributeTypeAndValue, 0)
			}
		} else if char == ' ' && buffer.Len() == 0 {
			// ignore unescaped leading spaces
			continue
		} else {
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
