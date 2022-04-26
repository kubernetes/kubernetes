// Package printf implements a parser for fmt.Printf-style format
// strings.
//
// It parses verbs according to the following syntax:
//     Numeric -> '0'-'9'
//     Letter -> 'a'-'z' | 'A'-'Z'
//     Index -> '[' Numeric+ ']'
//     Star -> '*'
//     Star -> Index '*'
//
//     Precision -> Numeric+ | Star
//     Width -> Numeric+ | Star
//
//     WidthAndPrecision -> Width '.' Precision
//     WidthAndPrecision -> Width '.'
//     WidthAndPrecision -> Width
//     WidthAndPrecision -> '.' Precision
//     WidthAndPrecision -> '.'
//
//     Flag -> '+' | '-' | '#' | ' ' | '0'
//     Verb -> Letter | '%'
//
//     Input -> '%' [ Flag+ ] [ WidthAndPrecision ] [ Index ] Verb
package printf

import (
	"errors"
	"regexp"
	"strconv"
	"strings"
)

// ErrInvalid is returned for invalid format strings or verbs.
var ErrInvalid = errors.New("invalid format string")

type Verb struct {
	Letter rune
	Flags  string

	Width     Argument
	Precision Argument
	// Which value in the argument list the verb uses.
	// -1 denotes the next argument,
	// values > 0 denote explicit arguments.
	// The value 0 denotes that no argument is consumed. This is the case for %%.
	Value int

	Raw string
}

// Argument is an implicit or explicit width or precision.
type Argument interface {
	isArgument()
}

// The Default value, when no width or precision is provided.
type Default struct{}

// Zero is the implicit zero value.
// This value may only appear for precisions in format strings like %6.f
type Zero struct{}

// Star is a * value, which may either refer to the next argument (Index == -1) or an explicit argument.
type Star struct{ Index int }

// A Literal value, such as 6 in %6d.
type Literal int

func (Default) isArgument() {}
func (Zero) isArgument()    {}
func (Star) isArgument()    {}
func (Literal) isArgument() {}

// Parse parses f and returns a list of actions.
// An action may either be a literal string, or a Verb.
func Parse(f string) ([]interface{}, error) {
	var out []interface{}
	for len(f) > 0 {
		if f[0] == '%' {
			v, n, err := ParseVerb(f)
			if err != nil {
				return nil, err
			}
			f = f[n:]
			out = append(out, v)
		} else {
			n := strings.IndexByte(f, '%')
			if n > -1 {
				out = append(out, f[:n])
				f = f[n:]
			} else {
				out = append(out, f)
				f = ""
			}
		}
	}

	return out, nil
}

func atoi(s string) int {
	n, _ := strconv.Atoi(s)
	return n
}

// ParseVerb parses the verb at the beginning of f.
// It returns the verb, how much of the input was consumed, and an error, if any.
func ParseVerb(f string) (Verb, int, error) {
	if len(f) < 2 {
		return Verb{}, 0, ErrInvalid
	}
	const (
		flags = 1

		width      = 2
		widthStar  = 3
		widthIndex = 5

		dot       = 6
		prec      = 7
		precStar  = 8
		precIndex = 10

		verbIndex = 11
		verb      = 12
	)

	m := re.FindStringSubmatch(f)
	if m == nil {
		return Verb{}, 0, ErrInvalid
	}

	v := Verb{
		Letter: []rune(m[verb])[0],
		Flags:  m[flags],
		Raw:    m[0],
	}

	if m[width] != "" {
		// Literal width
		v.Width = Literal(atoi(m[width]))
	} else if m[widthStar] != "" {
		// Star width
		if m[widthIndex] != "" {
			v.Width = Star{atoi(m[widthIndex])}
		} else {
			v.Width = Star{-1}
		}
	} else {
		// Default width
		v.Width = Default{}
	}

	if m[dot] == "" {
		// default precision
		v.Precision = Default{}
	} else {
		if m[prec] != "" {
			// Literal precision
			v.Precision = Literal(atoi(m[prec]))
		} else if m[precStar] != "" {
			// Star precision
			if m[precIndex] != "" {
				v.Precision = Star{atoi(m[precIndex])}
			} else {
				v.Precision = Star{-1}
			}
		} else {
			// Zero precision
			v.Precision = Zero{}
		}
	}

	if m[verb] == "%" {
		v.Value = 0
	} else if m[verbIndex] != "" {
		v.Value = atoi(m[verbIndex])
	} else {
		v.Value = -1
	}

	return v, len(m[0]), nil
}

const (
	flags             = `([+#0 -]*)`
	verb              = `([a-zA-Z%])`
	index             = `(?:\[([0-9]+)\])`
	star              = `((` + index + `)?\*)`
	width1            = `([0-9]+)`
	width2            = star
	width             = `(?:` + width1 + `|` + width2 + `)`
	precision         = width
	widthAndPrecision = `(?:(?:` + width + `)?(?:(\.)(?:` + precision + `)?)?)`
)

var re = regexp.MustCompile(`^%` + flags + widthAndPrecision + `?` + index + `?` + verb)
