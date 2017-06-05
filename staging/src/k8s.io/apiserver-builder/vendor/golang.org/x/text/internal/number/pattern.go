// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package number

import (
	"errors"
	"unicode/utf8"
)

// This file contains a parser for the CLDR number patterns as described in
// http://unicode.org/reports/tr35/tr35-numbers.html#Number_Format_Patterns.
//
// The following BNF is derived from this standard.
//
// pattern    := subpattern (';' subpattern)?
// subpattern := affix? number exponent? affix?
// number     := decimal | sigDigits
// decimal    := '#'* '0'* ('.' fraction)? | '#' | '0'
// fraction   := '0'* '#'*
// sigDigits  := '#'* '@' '@'* '#'*
// exponent   := 'E' '+'? '0'* '0'
// padSpec    := '*' \L
//
// Notes:
// - An affix pattern may contain any runes, but runes with special meaning
//   should be escaped.
// - Sequences of digits, '#', and '@' in decimal and sigDigits may have
//   interstitial commas.

// TODO: replace special characters in affixes (-, +, ¤) with control codes.

// Format holds information for formatting numbers. It is designed to hold
// information from CLDR number patterns.
//
// This pattern is precompiled  for all patterns for all languages. Even though
// the number of patterns is not very large, we want to keep this small.
//
// This type is only intended for internal use.
type Format struct {
	// TODO: this struct can be packed a lot better than it is now. Should be
	// possible to make it 32 bytes.

	Affix     string // includes prefix and suffix. First byte is prefix length.
	Offset    uint16 // Offset into Affix for prefix and suffix
	NegOffset uint16 // Offset into Affix for negative prefix and suffix or 0.

	Multiplier     uint32
	RoundIncrement uint32 // Use Min*Digits to determine scale
	PadRune        rune

	FormatWidth uint16

	GroupingSize [2]uint8
	Flags        FormatFlag

	// Number of digits.
	MinIntegerDigits     uint8
	MaxIntegerDigits     uint8
	MinFractionDigits    uint8
	MaxFractionDigits    uint8
	MinSignificantDigits uint8
	MaxSignificantDigits uint8
	MinExponentDigits    uint8
}

// A FormatFlag is a bit mask for the flag field of a Format.
type FormatFlag uint8

const (
	AlwaysSign FormatFlag = 1 << iota
	AlwaysExpSign
	AlwaysDecimalSeparator
	ParenthesisForNegative // Common pattern. Saves space.

	PadAfterNumber
	PadAfterAffix

	PadBeforePrefix = 0 // Default
	PadAfterPrefix  = PadAfterAffix
	PadBeforeSuffix = PadAfterNumber
	PadAfterSuffix  = PadAfterNumber | PadAfterAffix
	PadMask         = PadAfterNumber | PadAfterAffix
)

type parser struct {
	*Format

	leadingSharps int

	pos            int
	err            error
	doNotTerminate bool
	groupingCount  uint
	hasGroup       bool
	buf            []byte
}

func (p *parser) setError(err error) {
	if p.err == nil {
		p.err = err
	}
}

func (p *parser) updateGrouping() {
	if p.hasGroup && p.groupingCount < 255 {
		p.GroupingSize[1] = p.GroupingSize[0]
		p.GroupingSize[0] = uint8(p.groupingCount)
	}
	p.groupingCount = 0
	p.hasGroup = true
}

var (
	// TODO: more sensible and localizeable error messages.
	errMultiplePadSpecifiers = errors.New("format: pattern has multiple pad specifiers")
	errInvalidPadSpecifier   = errors.New("format: invalid pad specifier")
	errInvalidQuote          = errors.New("format: invalid quote")
	errAffixTooLarge         = errors.New("format: prefix or suffix exceeds maximum UTF-8 length of 256 bytes")
	errDuplicatePercentSign  = errors.New("format: duplicate percent sign")
	errDuplicatePermilleSign = errors.New("format: duplicate permille sign")
	errUnexpectedEnd         = errors.New("format: unexpected end of pattern")
)

// ParsePattern extracts formatting information from a CLDR number pattern.
//
// See http://unicode.org/reports/tr35/tr35-numbers.html#Number_Format_Patterns.
func ParsePattern(s string) (f *Format, err error) {
	p := parser{Format: &Format{}}

	s = p.parseSubPattern(s)

	if s != "" {
		// Parse negative sub pattern.
		if s[0] != ';' {
			p.setError(errors.New("format: error parsing first sub pattern"))
			return nil, p.err
		}
		neg := parser{Format: &Format{}} // just for extracting the affixes.
		s = neg.parseSubPattern(s[len(";"):])
		p.NegOffset = uint16(len(p.buf))
		p.buf = append(p.buf, neg.buf...)
	}
	if s != "" {
		p.setError(errors.New("format: spurious characters at end of pattern"))
	}
	if p.err != nil {
		return nil, p.err
	}
	if affix := string(p.buf); affix == "\x00\x00" || affix == "\x00\x00\x00\x00" {
		// No prefix or suffixes.
		p.NegOffset = 0
	} else {
		p.Affix = affix
	}
	return p.Format, nil
}

func (p *parser) parseSubPattern(s string) string {
	s = p.parsePad(s, PadBeforePrefix)
	s = p.parseAffix(s)
	s = p.parsePad(s, PadAfterPrefix)

	s = p.parse(p.number, s)

	s = p.parsePad(s, PadBeforeSuffix)
	s = p.parseAffix(s)
	s = p.parsePad(s, PadAfterSuffix)
	return s
}

func (p *parser) parsePad(s string, f FormatFlag) (tail string) {
	if len(s) >= 2 && s[0] == '*' {
		r, sz := utf8.DecodeRuneInString(s[1:])
		if p.PadRune != 0 {
			p.err = errMultiplePadSpecifiers
		} else {
			p.Flags |= f
			p.PadRune = r
		}
		return s[1+sz:]
	}
	return s
}

func (p *parser) parseAffix(s string) string {
	x := len(p.buf)
	p.buf = append(p.buf, 0) // placeholder for affix length

	s = p.parse(p.affix, s)

	n := len(p.buf) - x - 1
	if n > 0xFF {
		p.setError(errAffixTooLarge)
	}
	p.buf[x] = uint8(n)
	return s
}

// state implements a state transition. It returns the new state. A state
// function may set an error on the parser or may simply return on an incorrect
// token and let the next phase fail.
type state func(r rune) state

// parse repeatedly applies a state function on the given string until a
// termination condition is reached.
func (p *parser) parse(fn state, s string) (tail string) {
	for i, r := range s {
		p.doNotTerminate = false
		if fn = fn(r); fn == nil || p.err != nil {
			return s[i:]
		}
		p.FormatWidth++
	}
	if p.doNotTerminate {
		p.setError(errUnexpectedEnd)
	}
	return ""
}

func (p *parser) affix(r rune) state {
	switch r {
	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
		'#', '@', '.', '*', ',', ';':
		return nil
	case '\'':
		return p.escape
	case '%':
		if p.Multiplier != 0 {
			p.setError(errDuplicatePercentSign)
		}
		p.Multiplier = 100
	case '\u2030': // ‰ Per mille
		if p.Multiplier != 0 {
			p.setError(errDuplicatePermilleSign)
		}
		p.Multiplier = 1000
		// TODO: handle currency somehow: ¤, ¤¤, ¤¤¤, ¤¤¤¤
	}
	p.buf = append(p.buf, string(r)...)
	return p.affix
}

func (p *parser) escape(r rune) state {
	switch r {
	case '\'':
		return p.affix
	default:
		p.buf = append(p.buf, string(r)...)
	}
	return p.escape
}

// number parses a number. The BNF says the integer part should always have
// a '0', but that does not appear to be the case according to the rest of the
// documentation. We will allow having only '#' numbers.
func (p *parser) number(r rune) state {
	switch r {
	case '#':
		p.groupingCount++
		p.leadingSharps++
	case '@':
		p.groupingCount++
		p.leadingSharps = 0
		return p.sigDigits(r)
	case ',':
		if p.leadingSharps == 0 { // no leading commas
			return nil
		}
		p.updateGrouping()
	case 'E':
		p.MaxIntegerDigits = uint8(p.leadingSharps)
		return p.exponent
	case '.': // allow ".##" etc.
		p.updateGrouping()
		return p.fraction
	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		return p.integer(r)
	default:
		return nil
	}
	return p.number
}

func (p *parser) integer(r rune) state {
	if !('0' <= r && r <= '9') {
		var next state
		switch r {
		case 'E':
			if p.leadingSharps > 0 {
				p.MaxIntegerDigits = uint8(p.leadingSharps) + p.MinIntegerDigits
			}
			next = p.exponent
		case '.':
			next = p.fraction
		}
		p.updateGrouping()
		return next
	}
	p.RoundIncrement = p.RoundIncrement*10 + uint32(r-'0')
	p.groupingCount++
	p.MinIntegerDigits++
	return p.integer
}

func (p *parser) sigDigits(r rune) state {
	switch r {
	case '@':
		p.groupingCount++
		p.MaxSignificantDigits++
		p.MinSignificantDigits++
	case '#':
		return p.sigDigitsFinal(r)
	case 'E':
		p.updateGrouping()
		return p.normalizeSigDigitsWithExponent()
	default:
		p.updateGrouping()
		return nil
	}
	return p.sigDigits
}

func (p *parser) sigDigitsFinal(r rune) state {
	switch r {
	case '#':
		p.groupingCount++
		p.MaxSignificantDigits++
	case 'E':
		p.updateGrouping()
		return p.normalizeSigDigitsWithExponent()
	default:
		p.updateGrouping()
		return nil
	}
	return p.sigDigitsFinal
}

func (p *parser) normalizeSigDigitsWithExponent() state {
	p.MinIntegerDigits, p.MaxIntegerDigits = 1, 1
	p.MinFractionDigits = p.MinSignificantDigits - 1
	p.MaxFractionDigits = p.MaxSignificantDigits - 1
	p.MinSignificantDigits, p.MaxSignificantDigits = 0, 0
	return p.exponent
}

func (p *parser) fraction(r rune) state {
	switch r {
	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		p.RoundIncrement = p.RoundIncrement*10 + uint32(r-'0')
		p.MinFractionDigits++
		p.MaxFractionDigits++
	case '#':
		p.MaxFractionDigits++
	case 'E':
		if p.leadingSharps > 0 {
			p.MaxIntegerDigits = uint8(p.leadingSharps) + p.MinIntegerDigits
		}
		return p.exponent
	default:
		return nil
	}
	return p.fraction
}

func (p *parser) exponent(r rune) state {
	switch r {
	case '+':
		// Set mode and check it wasn't already set.
		if p.Flags&AlwaysExpSign != 0 || p.MinExponentDigits > 0 {
			break
		}
		p.Flags |= AlwaysExpSign
		p.doNotTerminate = true
		return p.exponent
	case '0':
		p.MinExponentDigits++
		return p.exponent
	}
	// termination condition
	if p.MinExponentDigits == 0 {
		p.setError(errors.New("format: need at least one digit"))
	}
	return nil
}
