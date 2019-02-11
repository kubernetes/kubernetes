package ini

import (
	"bytes"
	"fmt"
	"strconv"
)

const (
	none = numberFormat(iota)
	binary
	octal
	decimal
	hex
	exponent
)

type numberFormat int

// numberHelper is used to dictate what format a number is in
// and what to do for negative values. Since -1e-4 is a valid
// number, we cannot just simply check for duplicate negatives.
type numberHelper struct {
	numberFormat numberFormat

	negative         bool
	negativeExponent bool
}

func (b numberHelper) Exists() bool {
	return b.numberFormat != none
}

func (b numberHelper) IsNegative() bool {
	return b.negative || b.negativeExponent
}

func (b *numberHelper) Determine(c rune) error {
	if b.Exists() {
		return NewParseError(fmt.Sprintf("multiple number formats: 0%v", string(c)))
	}

	switch c {
	case 'b':
		b.numberFormat = binary
	case 'o':
		b.numberFormat = octal
	case 'x':
		b.numberFormat = hex
	case 'e', 'E':
		b.numberFormat = exponent
	case '-':
		if b.numberFormat != exponent {
			b.negative = true
		} else {
			b.negativeExponent = true
		}
	case '.':
		b.numberFormat = decimal
	default:
		return NewParseError(fmt.Sprintf("invalid number character: %v", string(c)))
	}

	return nil
}

func (b numberHelper) CorrectByte(c rune) bool {
	switch {
	case b.numberFormat == binary:
		if !isBinaryByte(c) {
			return false
		}
	case b.numberFormat == octal:
		if !isOctalByte(c) {
			return false
		}
	case b.numberFormat == hex:
		if !isHexByte(c) {
			return false
		}
	case b.numberFormat == decimal:
		if !isDigit(c) {
			return false
		}
	case b.numberFormat == exponent:
		if !isDigit(c) {
			return false
		}
	case b.negativeExponent:
		if !isDigit(c) {
			return false
		}
	case b.negative:
		if !isDigit(c) {
			return false
		}
	default:
		if !isDigit(c) {
			return false
		}
	}

	return true
}

func (b numberHelper) Base() int {
	switch b.numberFormat {
	case binary:
		return 2
	case octal:
		return 8
	case hex:
		return 16
	default:
		return 10
	}
}

func (b numberHelper) String() string {
	buf := bytes.Buffer{}
	i := 0

	switch b.numberFormat {
	case binary:
		i++
		buf.WriteString(strconv.Itoa(i) + ": binary format\n")
	case octal:
		i++
		buf.WriteString(strconv.Itoa(i) + ": octal format\n")
	case hex:
		i++
		buf.WriteString(strconv.Itoa(i) + ": hex format\n")
	case exponent:
		i++
		buf.WriteString(strconv.Itoa(i) + ": exponent format\n")
	default:
		i++
		buf.WriteString(strconv.Itoa(i) + ": integer format\n")
	}

	if b.negative {
		i++
		buf.WriteString(strconv.Itoa(i) + ": negative format\n")
	}

	if b.negativeExponent {
		i++
		buf.WriteString(strconv.Itoa(i) + ": negative exponent format\n")
	}

	return buf.String()
}
