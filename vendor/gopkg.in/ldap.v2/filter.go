// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ldap

import (
	"bytes"
	hexpac "encoding/hex"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	"gopkg.in/asn1-ber.v1"
)

const (
	FilterAnd             = 0
	FilterOr              = 1
	FilterNot             = 2
	FilterEqualityMatch   = 3
	FilterSubstrings      = 4
	FilterGreaterOrEqual  = 5
	FilterLessOrEqual     = 6
	FilterPresent         = 7
	FilterApproxMatch     = 8
	FilterExtensibleMatch = 9
)

var FilterMap = map[uint64]string{
	FilterAnd:             "And",
	FilterOr:              "Or",
	FilterNot:             "Not",
	FilterEqualityMatch:   "Equality Match",
	FilterSubstrings:      "Substrings",
	FilterGreaterOrEqual:  "Greater Or Equal",
	FilterLessOrEqual:     "Less Or Equal",
	FilterPresent:         "Present",
	FilterApproxMatch:     "Approx Match",
	FilterExtensibleMatch: "Extensible Match",
}

const (
	FilterSubstringsInitial = 0
	FilterSubstringsAny     = 1
	FilterSubstringsFinal   = 2
)

var FilterSubstringsMap = map[uint64]string{
	FilterSubstringsInitial: "Substrings Initial",
	FilterSubstringsAny:     "Substrings Any",
	FilterSubstringsFinal:   "Substrings Final",
}

const (
	MatchingRuleAssertionMatchingRule = 1
	MatchingRuleAssertionType         = 2
	MatchingRuleAssertionMatchValue   = 3
	MatchingRuleAssertionDNAttributes = 4
)

var MatchingRuleAssertionMap = map[uint64]string{
	MatchingRuleAssertionMatchingRule: "Matching Rule Assertion Matching Rule",
	MatchingRuleAssertionType:         "Matching Rule Assertion Type",
	MatchingRuleAssertionMatchValue:   "Matching Rule Assertion Match Value",
	MatchingRuleAssertionDNAttributes: "Matching Rule Assertion DN Attributes",
}

func CompileFilter(filter string) (*ber.Packet, error) {
	if len(filter) == 0 || filter[0] != '(' {
		return nil, NewError(ErrorFilterCompile, errors.New("ldap: filter does not start with an '('"))
	}
	packet, pos, err := compileFilter(filter, 1)
	if err != nil {
		return nil, err
	}
	if pos != len(filter) {
		return nil, NewError(ErrorFilterCompile, errors.New("ldap: finished compiling filter with extra at end: "+fmt.Sprint(filter[pos:])))
	}
	return packet, nil
}

func DecompileFilter(packet *ber.Packet) (ret string, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = NewError(ErrorFilterDecompile, errors.New("ldap: error decompiling filter"))
		}
	}()
	ret = "("
	err = nil
	childStr := ""

	switch packet.Tag {
	case FilterAnd:
		ret += "&"
		for _, child := range packet.Children {
			childStr, err = DecompileFilter(child)
			if err != nil {
				return
			}
			ret += childStr
		}
	case FilterOr:
		ret += "|"
		for _, child := range packet.Children {
			childStr, err = DecompileFilter(child)
			if err != nil {
				return
			}
			ret += childStr
		}
	case FilterNot:
		ret += "!"
		childStr, err = DecompileFilter(packet.Children[0])
		if err != nil {
			return
		}
		ret += childStr

	case FilterSubstrings:
		ret += ber.DecodeString(packet.Children[0].Data.Bytes())
		ret += "="
		for i, child := range packet.Children[1].Children {
			if i == 0 && child.Tag != FilterSubstringsInitial {
				ret += "*"
			}
			ret += EscapeFilter(ber.DecodeString(child.Data.Bytes()))
			if child.Tag != FilterSubstringsFinal {
				ret += "*"
			}
		}
	case FilterEqualityMatch:
		ret += ber.DecodeString(packet.Children[0].Data.Bytes())
		ret += "="
		ret += EscapeFilter(ber.DecodeString(packet.Children[1].Data.Bytes()))
	case FilterGreaterOrEqual:
		ret += ber.DecodeString(packet.Children[0].Data.Bytes())
		ret += ">="
		ret += EscapeFilter(ber.DecodeString(packet.Children[1].Data.Bytes()))
	case FilterLessOrEqual:
		ret += ber.DecodeString(packet.Children[0].Data.Bytes())
		ret += "<="
		ret += EscapeFilter(ber.DecodeString(packet.Children[1].Data.Bytes()))
	case FilterPresent:
		ret += ber.DecodeString(packet.Data.Bytes())
		ret += "=*"
	case FilterApproxMatch:
		ret += ber.DecodeString(packet.Children[0].Data.Bytes())
		ret += "~="
		ret += EscapeFilter(ber.DecodeString(packet.Children[1].Data.Bytes()))
	case FilterExtensibleMatch:
		attr := ""
		dnAttributes := false
		matchingRule := ""
		value := ""

		for _, child := range packet.Children {
			switch child.Tag {
			case MatchingRuleAssertionMatchingRule:
				matchingRule = ber.DecodeString(child.Data.Bytes())
			case MatchingRuleAssertionType:
				attr = ber.DecodeString(child.Data.Bytes())
			case MatchingRuleAssertionMatchValue:
				value = ber.DecodeString(child.Data.Bytes())
			case MatchingRuleAssertionDNAttributes:
				dnAttributes = child.Value.(bool)
			}
		}

		if len(attr) > 0 {
			ret += attr
		}
		if dnAttributes {
			ret += ":dn"
		}
		if len(matchingRule) > 0 {
			ret += ":"
			ret += matchingRule
		}
		ret += ":="
		ret += EscapeFilter(value)
	}

	ret += ")"
	return
}

func compileFilterSet(filter string, pos int, parent *ber.Packet) (int, error) {
	for pos < len(filter) && filter[pos] == '(' {
		child, newPos, err := compileFilter(filter, pos+1)
		if err != nil {
			return pos, err
		}
		pos = newPos
		parent.AppendChild(child)
	}
	if pos == len(filter) {
		return pos, NewError(ErrorFilterCompile, errors.New("ldap: unexpected end of filter"))
	}

	return pos + 1, nil
}

func compileFilter(filter string, pos int) (*ber.Packet, int, error) {
	var (
		packet *ber.Packet
		err    error
	)

	defer func() {
		if r := recover(); r != nil {
			err = NewError(ErrorFilterCompile, errors.New("ldap: error compiling filter"))
		}
	}()
	newPos := pos

	currentRune, currentWidth := utf8.DecodeRuneInString(filter[newPos:])

	switch currentRune {
	case utf8.RuneError:
		return nil, 0, NewError(ErrorFilterCompile, fmt.Errorf("ldap: error reading rune at position %d", newPos))
	case '(':
		packet, newPos, err = compileFilter(filter, pos+currentWidth)
		newPos++
		return packet, newPos, err
	case '&':
		packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterAnd, nil, FilterMap[FilterAnd])
		newPos, err = compileFilterSet(filter, pos+currentWidth, packet)
		return packet, newPos, err
	case '|':
		packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterOr, nil, FilterMap[FilterOr])
		newPos, err = compileFilterSet(filter, pos+currentWidth, packet)
		return packet, newPos, err
	case '!':
		packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterNot, nil, FilterMap[FilterNot])
		var child *ber.Packet
		child, newPos, err = compileFilter(filter, pos+currentWidth)
		packet.AppendChild(child)
		return packet, newPos, err
	default:
		READING_ATTR := 0
		READING_EXTENSIBLE_MATCHING_RULE := 1
		READING_CONDITION := 2

		state := READING_ATTR

		attribute := ""
		extensibleDNAttributes := false
		extensibleMatchingRule := ""
		condition := ""

		for newPos < len(filter) {
			remainingFilter := filter[newPos:]
			currentRune, currentWidth = utf8.DecodeRuneInString(remainingFilter)
			if currentRune == ')' {
				break
			}
			if currentRune == utf8.RuneError {
				return packet, newPos, NewError(ErrorFilterCompile, fmt.Errorf("ldap: error reading rune at position %d", newPos))
			}

			switch state {
			case READING_ATTR:
				switch {
				// Extensible rule, with only DN-matching
				case currentRune == ':' && strings.HasPrefix(remainingFilter, ":dn:="):
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterExtensibleMatch, nil, FilterMap[FilterExtensibleMatch])
					extensibleDNAttributes = true
					state = READING_CONDITION
					newPos += 5

				// Extensible rule, with DN-matching and a matching OID
				case currentRune == ':' && strings.HasPrefix(remainingFilter, ":dn:"):
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterExtensibleMatch, nil, FilterMap[FilterExtensibleMatch])
					extensibleDNAttributes = true
					state = READING_EXTENSIBLE_MATCHING_RULE
					newPos += 4

				// Extensible rule, with attr only
				case currentRune == ':' && strings.HasPrefix(remainingFilter, ":="):
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterExtensibleMatch, nil, FilterMap[FilterExtensibleMatch])
					state = READING_CONDITION
					newPos += 2

				// Extensible rule, with no DN attribute matching
				case currentRune == ':':
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterExtensibleMatch, nil, FilterMap[FilterExtensibleMatch])
					state = READING_EXTENSIBLE_MATCHING_RULE
					newPos += 1

				// Equality condition
				case currentRune == '=':
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterEqualityMatch, nil, FilterMap[FilterEqualityMatch])
					state = READING_CONDITION
					newPos += 1

				// Greater-than or equal
				case currentRune == '>' && strings.HasPrefix(remainingFilter, ">="):
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterGreaterOrEqual, nil, FilterMap[FilterGreaterOrEqual])
					state = READING_CONDITION
					newPos += 2

				// Less-than or equal
				case currentRune == '<' && strings.HasPrefix(remainingFilter, "<="):
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterLessOrEqual, nil, FilterMap[FilterLessOrEqual])
					state = READING_CONDITION
					newPos += 2

				// Approx
				case currentRune == '~' && strings.HasPrefix(remainingFilter, "~="):
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterApproxMatch, nil, FilterMap[FilterApproxMatch])
					state = READING_CONDITION
					newPos += 2

				// Still reading the attribute name
				default:
					attribute += fmt.Sprintf("%c", currentRune)
					newPos += currentWidth
				}

			case READING_EXTENSIBLE_MATCHING_RULE:
				switch {

				// Matching rule OID is done
				case currentRune == ':' && strings.HasPrefix(remainingFilter, ":="):
					state = READING_CONDITION
					newPos += 2

				// Still reading the matching rule oid
				default:
					extensibleMatchingRule += fmt.Sprintf("%c", currentRune)
					newPos += currentWidth
				}

			case READING_CONDITION:
				// append to the condition
				condition += fmt.Sprintf("%c", currentRune)
				newPos += currentWidth
			}
		}

		if newPos == len(filter) {
			err = NewError(ErrorFilterCompile, errors.New("ldap: unexpected end of filter"))
			return packet, newPos, err
		}
		if packet == nil {
			err = NewError(ErrorFilterCompile, errors.New("ldap: error parsing filter"))
			return packet, newPos, err
		}

		switch {
		case packet.Tag == FilterExtensibleMatch:
			// MatchingRuleAssertion ::= SEQUENCE {
			//         matchingRule    [1] MatchingRuleID OPTIONAL,
			//         type            [2] AttributeDescription OPTIONAL,
			//         matchValue      [3] AssertionValue,
			//         dnAttributes    [4] BOOLEAN DEFAULT FALSE
			// }

			// Include the matching rule oid, if specified
			if len(extensibleMatchingRule) > 0 {
				packet.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, MatchingRuleAssertionMatchingRule, extensibleMatchingRule, MatchingRuleAssertionMap[MatchingRuleAssertionMatchingRule]))
			}

			// Include the attribute, if specified
			if len(attribute) > 0 {
				packet.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, MatchingRuleAssertionType, attribute, MatchingRuleAssertionMap[MatchingRuleAssertionType]))
			}

			// Add the value (only required child)
			encodedString, err := escapedStringToEncodedBytes(condition)
			if err != nil {
				return packet, newPos, err
			}
			packet.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, MatchingRuleAssertionMatchValue, encodedString, MatchingRuleAssertionMap[MatchingRuleAssertionMatchValue]))

			// Defaults to false, so only include in the sequence if true
			if extensibleDNAttributes {
				packet.AppendChild(ber.NewBoolean(ber.ClassContext, ber.TypePrimitive, MatchingRuleAssertionDNAttributes, extensibleDNAttributes, MatchingRuleAssertionMap[MatchingRuleAssertionDNAttributes]))
			}

		case packet.Tag == FilterEqualityMatch && condition == "*":
			packet = ber.NewString(ber.ClassContext, ber.TypePrimitive, FilterPresent, attribute, FilterMap[FilterPresent])
		case packet.Tag == FilterEqualityMatch && strings.Contains(condition, "*"):
			packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, attribute, "Attribute"))
			packet.Tag = FilterSubstrings
			packet.Description = FilterMap[uint64(packet.Tag)]
			seq := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Substrings")
			parts := strings.Split(condition, "*")
			for i, part := range parts {
				if part == "" {
					continue
				}
				var tag ber.Tag
				switch i {
				case 0:
					tag = FilterSubstringsInitial
				case len(parts) - 1:
					tag = FilterSubstringsFinal
				default:
					tag = FilterSubstringsAny
				}
				encodedString, err := escapedStringToEncodedBytes(part)
				if err != nil {
					return packet, newPos, err
				}
				seq.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, tag, encodedString, FilterSubstringsMap[uint64(tag)]))
			}
			packet.AppendChild(seq)
		default:
			encodedString, err := escapedStringToEncodedBytes(condition)
			if err != nil {
				return packet, newPos, err
			}
			packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, attribute, "Attribute"))
			packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, encodedString, "Condition"))
		}

		newPos += currentWidth
		return packet, newPos, err
	}
}

// Convert from "ABC\xx\xx\xx" form to literal bytes for transport
func escapedStringToEncodedBytes(escapedString string) (string, error) {
	var buffer bytes.Buffer
	i := 0
	for i < len(escapedString) {
		currentRune, currentWidth := utf8.DecodeRuneInString(escapedString[i:])
		if currentRune == utf8.RuneError {
			return "", NewError(ErrorFilterCompile, fmt.Errorf("ldap: error reading rune at position %d", i))
		}

		// Check for escaped hex characters and convert them to their literal value for transport.
		if currentRune == '\\' {
			// http://tools.ietf.org/search/rfc4515
			// \ (%x5C) is not a valid character unless it is followed by two HEX characters due to not
			// being a member of UTF1SUBSET.
			if i+2 > len(escapedString) {
				return "", NewError(ErrorFilterCompile, errors.New("ldap: missing characters for escape in filter"))
			}
			if escByte, decodeErr := hexpac.DecodeString(escapedString[i+1 : i+3]); decodeErr != nil {
				return "", NewError(ErrorFilterCompile, errors.New("ldap: invalid characters for escape in filter"))
			} else {
				buffer.WriteByte(escByte[0])
				i += 2 // +1 from end of loop, so 3 total for \xx.
			}
		} else {
			buffer.WriteRune(currentRune)
		}

		i += currentWidth
	}
	return buffer.String(), nil
}
