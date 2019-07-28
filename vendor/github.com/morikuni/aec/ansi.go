package aec

import (
	"fmt"
	"strings"
)

const esc = "\x1b["

// Reset resets SGR effect.
const Reset string = "\x1b[0m"

var empty = newAnsi("")

// ANSI represents ANSI escape code.
type ANSI interface {
	fmt.Stringer

	// With adapts given ANSIs.
	With(...ANSI) ANSI

	// Apply wraps given string in ANSI.
	Apply(string) string
}

type ansiImpl string

func newAnsi(s string) *ansiImpl {
	r := ansiImpl(s)
	return &r
}

func (a *ansiImpl) With(ansi ...ANSI) ANSI {
	return concat(append([]ANSI{a}, ansi...))
}

func (a *ansiImpl) Apply(s string) string {
	return a.String() + s + Reset
}

func (a *ansiImpl) String() string {
	return string(*a)
}

// Apply wraps given string in ANSIs.
func Apply(s string, ansi ...ANSI) string {
	if len(ansi) == 0 {
		return s
	}
	return concat(ansi).Apply(s)
}

func concat(ansi []ANSI) ANSI {
	strs := make([]string, 0, len(ansi))
	for _, p := range ansi {
		strs = append(strs, p.String())
	}
	return newAnsi(strings.Join(strs, ""))
}
