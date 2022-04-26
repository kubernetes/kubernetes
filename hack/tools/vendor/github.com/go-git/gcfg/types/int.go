package types

import (
	"fmt"
	"strings"
)

// An IntMode is a mode for parsing integer values, representing a set of
// accepted bases.
type IntMode uint8

// IntMode values for ParseInt; can be combined using binary or.
const (
	Dec IntMode = 1 << iota
	Hex
	Oct
)

// String returns a string representation of IntMode; e.g. `IntMode(Dec|Hex)`.
func (m IntMode) String() string {
	var modes []string
	if m&Dec != 0 {
		modes = append(modes, "Dec")
	}
	if m&Hex != 0 {
		modes = append(modes, "Hex")
	}
	if m&Oct != 0 {
		modes = append(modes, "Oct")
	}
	return "IntMode(" + strings.Join(modes, "|") + ")"
}

var errIntAmbig = fmt.Errorf("ambiguous integer value; must include '0' prefix")

func prefix0(val string) bool {
	return strings.HasPrefix(val, "0") || strings.HasPrefix(val, "-0")
}

func prefix0x(val string) bool {
	return strings.HasPrefix(val, "0x") || strings.HasPrefix(val, "-0x")
}

// ParseInt parses val using mode into intptr, which must be a pointer to an
// integer kind type. Non-decimal value require prefix `0` or `0x` in the cases
// when mode permits ambiguity of base; otherwise the prefix can be omitted.
func ParseInt(intptr interface{}, val string, mode IntMode) error {
	val = strings.TrimSpace(val)
	verb := byte(0)
	switch mode {
	case Dec:
		verb = 'd'
	case Dec + Hex:
		if prefix0x(val) {
			verb = 'v'
		} else {
			verb = 'd'
		}
	case Dec + Oct:
		if prefix0(val) && !prefix0x(val) {
			verb = 'v'
		} else {
			verb = 'd'
		}
	case Dec + Hex + Oct:
		verb = 'v'
	case Hex:
		if prefix0x(val) {
			verb = 'v'
		} else {
			verb = 'x'
		}
	case Oct:
		verb = 'o'
	case Hex + Oct:
		if prefix0(val) {
			verb = 'v'
		} else {
			return errIntAmbig
		}
	}
	if verb == 0 {
		panic("unsupported mode")
	}
	return ScanFully(intptr, val, verb)
}
