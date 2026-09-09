// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package format

import (
	"reflect"
	"unicode/utf8"
)

// A Parser parses a format string. The result from the parse are set in the
// struct fields.
type Parser struct {
	Verb rune

	WidthPresent bool
	PrecPresent  bool
	Minus        bool
	Plus         bool
	Sharp        bool
	Space        bool
	Zero         bool

	// For the formats %+v %#v, we set the plusV/sharpV flags
	// and clear the plus/sharp flags since %+v and %#v are in effect
	// different, flagless formats set at the top level.
	PlusV  bool
	SharpV bool

	HasIndex bool

	Width int
	Prec  int // precision

	// retain arguments across calls.
	Args []interface{}
	// retain current argument number across calls
	ArgNum int

	// reordered records whether the format string used argument reordering.
	Reordered bool
	// goodArgNum records whether the most recent reordering directive was valid.
	goodArgNum bool

	// position info
	format   string
	startPos int
	endPos   int
	Status   Status
}

// Reset initializes a parser to scan format strings for the given args.
func (p *Parser) Reset(args []interface{}) {
	p.Args = args
	p.ArgNum = 0
	p.startPos = 0
	p.Reordered = false
}

// Text returns the part of the format string that was parsed by the last call
// to Scan. It returns the original substitution clause if the current scan
// parsed a substitution.
func (p *Parser) Text() string { return p.format[p.startPos:p.endPos] }

// SetFormat sets a new format string to parse. It does not reset the argument
// count.
func (p *Parser) SetFormat(format string) {
	p.format = format
	p.startPos = 0
	p.endPos = 0
}

// Status indicates the result type of a call to Scan.
type Status int

const (
	StatusText Status = iota
	StatusSubstitution
	StatusBadWidthSubstitution
	StatusBadPrecSubstitution
	StatusNoVerb
	StatusBadArgNum
	StatusMissingArg
)

// ClearFlags reset the parser to default behavior.
func (p *Parser) ClearFlags() {
	p.WidthPresent = false
	p.PrecPresent = false
	p.Minus = false
	p.Plus = false
	p.Sharp = false
	p.Space = false
	p.Zero = false

	p.PlusV = false
	p.SharpV = false

	p.HasIndex = false
}

// Scan scans the next part of the format string and sets the status to
// indicate whether it scanned a string literal, substitution or error.
func (p *Parser) Scan() bool {
	p.Status = StatusText
	format := p.format
	end := len(format)
	if p.endPos >= end {
		return false
	}
	afterIndex := false // previous item in format was an index like [3].

	p.startPos = p.endPos
	p.goodArgNum = true
	i := p.startPos
	for i < end && format[i] != '%' {
		i++
	}
	if i > p.startPos {
		p.endPos = i
		return true
	}
	// Process one verb
	i++

	p.Status = StatusSubstitution

	// Do we have flags?
	p.ClearFlags()

simpleFormat:
	for ; i < end; i++ {
		c := p.format[i]
		switch c {
		case '#':
			p.Sharp = true
		case '0':
			p.Zero = !p.Minus // Only allow zero padding to the left.
		case '+':
			p.Plus = true
		case '-':
			p.Minus = true
			p.Zero = false // Do not pad with zeros to the right.
		case ' ':
			p.Space = true
		default:
			// Fast path for common case of ascii lower case simple verbs
			// without precision or width or argument indices.
			if 'a' <= c && c <= 'z' && p.ArgNum < len(p.Args) {
				if c == 'v' {
					// Go syntax
					p.SharpV = p.Sharp
					p.Sharp = false
					// Struct-field syntax
					p.PlusV = p.Plus
					p.Plus = false
				}
				p.Verb = rune(c)
				p.ArgNum++
				p.endPos = i + 1
				return true
			}
			// Format is more complex than simple flags and a verb or is malformed.
			break simpleFormat
		}
	}

	// Do we have an explicit argument index?
	i, afterIndex = p.updateArgNumber(format, i)

	// Do we have width?
	if i < end && format[i] == '*' {
		i++
		p.Width, p.WidthPresent = p.intFromArg()

		if !p.WidthPresent {
			p.Status = StatusBadWidthSubstitution
		}

		// We have a negative width, so take its value and ensure
		// that the minus flag is set
		if p.Width < 0 {
			p.Width = -p.Width
			p.Minus = true
			p.Zero = false // Do not pad with zeros to the right.
		}
		afterIndex = false
	} else {
		p.Width, p.WidthPresent, i = parsenum(format, i, end)
		if afterIndex && p.WidthPresent { // "%[3]2d"
			p.goodArgNum = false
		}
	}

	// Do we have precision?
	if i+1 < end && format[i] == '.' {
		i++
		if afterIndex { // "%[3].2d"
			p.goodArgNum = false
		}
		i, afterIndex = p.updateArgNumber(format, i)
		if i < end && format[i] == '*' {
			i++
			p.Prec, p.PrecPresent = p.intFromArg()
			// Negative precision arguments don't make sense
			if p.Prec < 0 {
				p.Prec = 0
				p.PrecPresent = false
			}
			if !p.PrecPresent {
				p.Status = StatusBadPrecSubstitution
			}
			afterIndex = false
		} else {
			p.Prec, p.PrecPresent, i = parsenum(format, i, end)
			if !p.PrecPresent {
				p.Prec = 0
				p.PrecPresent = true
			}
		}
	}

	if !afterIndex {
		i, afterIndex = p.updateArgNumber(format, i)
	}
	p.HasIndex = afterIndex

	if i >= end {
		p.endPos = i
		p.Status = StatusNoVerb
		return true
	}

	verb, w := utf8.DecodeRuneInString(format[i:])
	p.endPos = i + w
	p.Verb = verb

	switch {
	case verb == '%': // Percent does not absorb operands and ignores f.wid and f.prec.
		p.startPos = p.endPos - 1
		p.Status = StatusText
	case !p.goodArgNum:
		p.Status = StatusBadArgNum
	case p.ArgNum >= len(p.Args): // No argument left over to print for the current verb.
		p.Status = StatusMissingArg
		p.ArgNum++
	case verb == 'v':
		// Go syntax
		p.SharpV = p.Sharp
		p.Sharp = false
		// Struct-field syntax
		p.PlusV = p.Plus
		p.Plus = false
		fallthrough
	default:
		p.ArgNum++
	}
	return true
}

// intFromArg gets the ArgNumth element of Args. On return, isInt reports
// whether the argument has integer type.
func (p *Parser) intFromArg() (num int, isInt bool) {
	if p.ArgNum < len(p.Args) {
		arg := p.Args[p.ArgNum]
		num, isInt = arg.(int) // Almost always OK.
		if !isInt {
			// Work harder.
			switch v := reflect.ValueOf(arg); v.Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				n := v.Int()
				if int64(int(n)) == n {
					num = int(n)
					isInt = true
				}
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
				n := v.Uint()
				if int64(n) >= 0 && uint64(int(n)) == n {
					num = int(n)
					isInt = true
				}
			default:
				// Already 0, false.
			}
		}
		p.ArgNum++
		if tooLarge(num) {
			num = 0
			isInt = false
		}
	}
	return
}

// parseArgNumber returns the value of the bracketed number, minus 1
// (explicit argument numbers are one-indexed but we want zero-indexed).
// The opening bracket is known to be present at format[0].
// The returned values are the index, the number of bytes to consume
// up to the closing paren, if present, and whether the number parsed
// ok. The bytes to consume will be 1 if no closing paren is present.
func parseArgNumber(format string) (index int, wid int, ok bool) {
	// There must be at least 3 bytes: [n].
	if len(format) < 3 {
		return 0, 1, false
	}

	// Find closing bracket.
	for i := 1; i < len(format); i++ {
		if format[i] == ']' {
			width, ok, newi := parsenum(format, 1, i)
			if !ok || newi != i {
				return 0, i + 1, false
			}
			return width - 1, i + 1, true // arg numbers are one-indexed and skip paren.
		}
	}
	return 0, 1, false
}

// updateArgNumber returns the next argument to evaluate, which is either the value of the passed-in
// argNum or the value of the bracketed integer that begins format[i:]. It also returns
// the new value of i, that is, the index of the next byte of the format to process.
func (p *Parser) updateArgNumber(format string, i int) (newi int, found bool) {
	if len(format) <= i || format[i] != '[' {
		return i, false
	}
	p.Reordered = true
	index, wid, ok := parseArgNumber(format[i:])
	if ok && 0 <= index && index < len(p.Args) {
		p.ArgNum = index
		return i + wid, true
	}
	p.goodArgNum = false
	return i + wid, ok
}

// tooLarge reports whether the magnitude of the integer is
// too large to be used as a formatting width or precision.
func tooLarge(x int) bool {
	const max int = 1e6
	return x > max || x < -max
}

// parsenum converts ASCII to integer.  num is 0 (and isnum is false) if no number present.
func parsenum(s string, start, end int) (num int, isnum bool, newi int) {
	if start >= end {
		return 0, false, end
	}
	for newi = start; newi < end && '0' <= s[newi] && s[newi] <= '9'; newi++ {
		if tooLarge(num) {
			return 0, false, end // Overflow; crazy long number most likely.
		}
		num = num*10 + int(s[newi]-'0')
		isnum = true
	}
	return
}
