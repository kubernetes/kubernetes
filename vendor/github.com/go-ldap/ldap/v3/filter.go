package ldap

import (
	"bytes"
	hexpac "encoding/hex"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode"
	"unicode/utf8"

	ber "github.com/go-asn1-ber/asn1-ber"
)

// Filter choices
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

// FilterMap contains human readable descriptions of Filter choices
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

// SubstringFilter options
const (
	FilterSubstringsInitial = 0
	FilterSubstringsAny     = 1
	FilterSubstringsFinal   = 2
)

// FilterSubstringsMap contains human readable descriptions of SubstringFilter choices
var FilterSubstringsMap = map[uint64]string{
	FilterSubstringsInitial: "Substrings Initial",
	FilterSubstringsAny:     "Substrings Any",
	FilterSubstringsFinal:   "Substrings Final",
}

// MatchingRuleAssertion choices
const (
	MatchingRuleAssertionMatchingRule = 1
	MatchingRuleAssertionType         = 2
	MatchingRuleAssertionMatchValue   = 3
	MatchingRuleAssertionDNAttributes = 4
)

// MatchingRuleAssertionMap contains human readable descriptions of MatchingRuleAssertion choices
var MatchingRuleAssertionMap = map[uint64]string{
	MatchingRuleAssertionMatchingRule: "Matching Rule Assertion Matching Rule",
	MatchingRuleAssertionType:         "Matching Rule Assertion Type",
	MatchingRuleAssertionMatchValue:   "Matching Rule Assertion Match Value",
	MatchingRuleAssertionDNAttributes: "Matching Rule Assertion DN Attributes",
}

var _SymbolAny = []byte{'*'}

// CompileFilter converts a string representation of a filter into a BER-encoded packet
func CompileFilter(filter string) (*ber.Packet, error) {
	if len(filter) == 0 || filter[0] != '(' {
		return nil, NewError(ErrorFilterCompile, errors.New("ldap: filter does not start with an '('"))
	}
	packet, pos, err := compileFilter(filter, 1)
	if err != nil {
		return nil, err
	}
	switch {
	case pos > len(filter):
		return nil, NewError(ErrorFilterCompile, errors.New("ldap: unexpected end of filter"))
	case pos < len(filter):
		return nil, NewError(ErrorFilterCompile, errors.New("ldap: finished compiling filter with extra at end: "+fmt.Sprint(filter[pos:])))
	}
	return packet, nil
}

// DecompileFilter converts a packet representation of a filter into a string representation
func DecompileFilter(packet *ber.Packet) (_ string, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = NewError(ErrorFilterDecompile, errors.New("ldap: error decompiling filter"))
		}
	}()

	buf := bytes.NewBuffer(nil)
	buf.WriteByte('(')
	childStr := ""

	switch packet.Tag {
	case FilterAnd:
		buf.WriteByte('&')
		for _, child := range packet.Children {
			childStr, err = DecompileFilter(child)
			if err != nil {
				return
			}
			buf.WriteString(childStr)
		}
	case FilterOr:
		buf.WriteByte('|')
		for _, child := range packet.Children {
			childStr, err = DecompileFilter(child)
			if err != nil {
				return
			}
			buf.WriteString(childStr)
		}
	case FilterNot:
		buf.WriteByte('!')
		childStr, err = DecompileFilter(packet.Children[0])
		if err != nil {
			return
		}
		buf.WriteString(childStr)

	case FilterSubstrings:
		buf.WriteString(ber.DecodeString(packet.Children[0].Data.Bytes()))
		buf.WriteByte('=')
		for i, child := range packet.Children[1].Children {
			if i == 0 && child.Tag != FilterSubstringsInitial {
				buf.Write(_SymbolAny)
			}
			buf.WriteString(EscapeFilter(ber.DecodeString(child.Data.Bytes())))
			if child.Tag != FilterSubstringsFinal {
				buf.Write(_SymbolAny)
			}
		}
	case FilterEqualityMatch:
		buf.WriteString(ber.DecodeString(packet.Children[0].Data.Bytes()))
		buf.WriteByte('=')
		buf.WriteString(EscapeFilter(ber.DecodeString(packet.Children[1].Data.Bytes())))
	case FilterGreaterOrEqual:
		buf.WriteString(ber.DecodeString(packet.Children[0].Data.Bytes()))
		buf.WriteString(">=")
		buf.WriteString(EscapeFilter(ber.DecodeString(packet.Children[1].Data.Bytes())))
	case FilterLessOrEqual:
		buf.WriteString(ber.DecodeString(packet.Children[0].Data.Bytes()))
		buf.WriteString("<=")
		buf.WriteString(EscapeFilter(ber.DecodeString(packet.Children[1].Data.Bytes())))
	case FilterPresent:
		buf.WriteString(ber.DecodeString(packet.Data.Bytes()))
		buf.WriteString("=*")
	case FilterApproxMatch:
		buf.WriteString(ber.DecodeString(packet.Children[0].Data.Bytes()))
		buf.WriteString("~=")
		buf.WriteString(EscapeFilter(ber.DecodeString(packet.Children[1].Data.Bytes())))
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
			buf.WriteString(attr)
		}
		if dnAttributes {
			buf.WriteString(":dn")
		}
		if len(matchingRule) > 0 {
			buf.WriteString(":")
			buf.WriteString(matchingRule)
		}
		buf.WriteString(":=")
		buf.WriteString(EscapeFilter(value))
	}

	buf.WriteByte(')')

	return buf.String(), nil
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
		const (
			stateReadingAttr                   = 0
			stateReadingExtensibleMatchingRule = 1
			stateReadingCondition              = 2
		)

		state := stateReadingAttr
		attribute := bytes.NewBuffer(nil)
		extensibleDNAttributes := false
		extensibleMatchingRule := bytes.NewBuffer(nil)
		condition := bytes.NewBuffer(nil)

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
			case stateReadingAttr:
				switch {
				// Extensible rule, with only DN-matching
				case currentRune == ':' && strings.HasPrefix(remainingFilter, ":dn:="):
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterExtensibleMatch, nil, FilterMap[FilterExtensibleMatch])
					extensibleDNAttributes = true
					state = stateReadingCondition
					newPos += 5

				// Extensible rule, with DN-matching and a matching OID
				case currentRune == ':' && strings.HasPrefix(remainingFilter, ":dn:"):
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterExtensibleMatch, nil, FilterMap[FilterExtensibleMatch])
					extensibleDNAttributes = true
					state = stateReadingExtensibleMatchingRule
					newPos += 4

				// Extensible rule, with attr only
				case currentRune == ':' && strings.HasPrefix(remainingFilter, ":="):
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterExtensibleMatch, nil, FilterMap[FilterExtensibleMatch])
					state = stateReadingCondition
					newPos += 2

				// Extensible rule, with no DN attribute matching
				case currentRune == ':':
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterExtensibleMatch, nil, FilterMap[FilterExtensibleMatch])
					state = stateReadingExtensibleMatchingRule
					newPos++

				// Equality condition
				case currentRune == '=':
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterEqualityMatch, nil, FilterMap[FilterEqualityMatch])
					state = stateReadingCondition
					newPos++

				// Greater-than or equal
				case currentRune == '>' && strings.HasPrefix(remainingFilter, ">="):
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterGreaterOrEqual, nil, FilterMap[FilterGreaterOrEqual])
					state = stateReadingCondition
					newPos += 2

				// Less-than or equal
				case currentRune == '<' && strings.HasPrefix(remainingFilter, "<="):
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterLessOrEqual, nil, FilterMap[FilterLessOrEqual])
					state = stateReadingCondition
					newPos += 2

				// Approx
				case currentRune == '~' && strings.HasPrefix(remainingFilter, "~="):
					packet = ber.Encode(ber.ClassContext, ber.TypeConstructed, FilterApproxMatch, nil, FilterMap[FilterApproxMatch])
					state = stateReadingCondition
					newPos += 2

				// Still reading the attribute name
				default:
					attribute.WriteRune(currentRune)
					newPos += currentWidth
				}

			case stateReadingExtensibleMatchingRule:
				switch {

				// Matching rule OID is done
				case currentRune == ':' && strings.HasPrefix(remainingFilter, ":="):
					state = stateReadingCondition
					newPos += 2

				// Still reading the matching rule oid
				default:
					extensibleMatchingRule.WriteRune(currentRune)
					newPos += currentWidth
				}

			case stateReadingCondition:
				// append to the condition
				condition.WriteRune(currentRune)
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
			if extensibleMatchingRule.Len() > 0 {
				packet.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, MatchingRuleAssertionMatchingRule, extensibleMatchingRule.String(), MatchingRuleAssertionMap[MatchingRuleAssertionMatchingRule]))
			}

			// Include the attribute, if specified
			if attribute.Len() > 0 {
				packet.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, MatchingRuleAssertionType, attribute.String(), MatchingRuleAssertionMap[MatchingRuleAssertionType]))
			}

			// Add the value (only required child)
			encodedString, encodeErr := decodeEscapedSymbols(condition.Bytes())
			if encodeErr != nil {
				return packet, newPos, encodeErr
			}
			packet.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, MatchingRuleAssertionMatchValue, encodedString, MatchingRuleAssertionMap[MatchingRuleAssertionMatchValue]))

			// Defaults to false, so only include in the sequence if true
			if extensibleDNAttributes {
				packet.AppendChild(ber.NewBoolean(ber.ClassContext, ber.TypePrimitive, MatchingRuleAssertionDNAttributes, extensibleDNAttributes, MatchingRuleAssertionMap[MatchingRuleAssertionDNAttributes]))
			}

		case packet.Tag == FilterEqualityMatch && bytes.Equal(condition.Bytes(), _SymbolAny):
			packet = ber.NewString(ber.ClassContext, ber.TypePrimitive, FilterPresent, attribute.String(), FilterMap[FilterPresent])
		case packet.Tag == FilterEqualityMatch && bytes.Contains(condition.Bytes(), _SymbolAny):
			packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, attribute.String(), "Attribute"))
			packet.Tag = FilterSubstrings
			packet.Description = FilterMap[uint64(packet.Tag)]
			seq := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Substrings")
			parts := bytes.Split(condition.Bytes(), _SymbolAny)
			for i, part := range parts {
				if len(part) == 0 {
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
				encodedString, encodeErr := decodeEscapedSymbols(part)
				if encodeErr != nil {
					return packet, newPos, encodeErr
				}
				seq.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, tag, encodedString, FilterSubstringsMap[uint64(tag)]))
			}
			packet.AppendChild(seq)
		default:
			encodedString, encodeErr := decodeEscapedSymbols(condition.Bytes())
			if encodeErr != nil {
				return packet, newPos, encodeErr
			}
			packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, attribute.String(), "Attribute"))
			packet.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, encodedString, "Condition"))
		}

		newPos += currentWidth
		return packet, newPos, err
	}
}

// Convert from "ABC\xx\xx\xx" form to literal bytes for transport
func decodeEscapedSymbols(src []byte) (string, error) {
	var (
		buffer  bytes.Buffer
		offset  int
		reader  = bytes.NewReader(src)
		byteHex []byte
		byteVal []byte
	)

	for {
		runeVal, runeSize, err := reader.ReadRune()
		if err == io.EOF {
			return buffer.String(), nil
		} else if err != nil {
			return "", NewError(ErrorFilterCompile, fmt.Errorf("ldap: failed to read filter: %v", err))
		} else if runeVal == unicode.ReplacementChar {
			return "", NewError(ErrorFilterCompile, fmt.Errorf("ldap: error reading rune at position %d", offset))
		}

		if runeVal == '\\' {
			// http://tools.ietf.org/search/rfc4515
			// \ (%x5C) is not a valid character unless it is followed by two HEX characters due to not
			// being a member of UTF1SUBSET.
			if byteHex == nil {
				byteHex = make([]byte, 2)
				byteVal = make([]byte, 1)
			}

			if _, err := io.ReadFull(reader, byteHex); err != nil {
				if err == io.ErrUnexpectedEOF {
					return "", NewError(ErrorFilterCompile, errors.New("ldap: missing characters for escape in filter"))
				}
				return "", NewError(ErrorFilterCompile, fmt.Errorf("ldap: invalid characters for escape in filter: %v", err))
			}

			if _, err := hexpac.Decode(byteVal, byteHex); err != nil {
				return "", NewError(ErrorFilterCompile, fmt.Errorf("ldap: invalid characters for escape in filter: %v", err))
			}

			buffer.Write(byteVal)
		} else {
			buffer.WriteRune(runeVal)
		}

		offset += runeSize
	}
}
