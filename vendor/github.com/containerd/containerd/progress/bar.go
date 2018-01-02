package progress

import (
	"bytes"
	"fmt"
)

// TODO(stevvooe): We may want to support more interesting parameterization of
// the bar. For now, it is very simple.

// Bar provides a very simple progress bar implementation.
//
// Use with fmt.Printf and "r" to format the progress bar. A "-" flag makes it
// progress from right to left.
type Bar float64

var _ fmt.Formatter = Bar(1.0)

// Format the progress bar as output
func (h Bar) Format(state fmt.State, r rune) {
	switch r {
	case 'r':
	default:
		panic(fmt.Sprintf("%v: unexpected format character", float64(h)))
	}

	if h > 1.0 {
		h = 1.0
	}

	if h < 0.0 {
		h = 0.0
	}

	if state.Flag('-') {
		h = 1.0 - h
	}

	width, ok := state.Width()
	if !ok {
		// default width of 40
		width = 40
	}

	var pad int

	extra := len([]byte(green)) + len([]byte(reset))

	p := make([]byte, width+extra)
	p[0], p[len(p)-1] = '|', '|'
	pad += 2

	positive := int(Bar(width-pad) * h)
	negative := width - pad - positive

	n := 1
	n += copy(p[n:], []byte(green))
	n += copy(p[n:], bytes.Repeat([]byte("+"), positive))
	n += copy(p[n:], []byte(reset))

	if negative > 0 {
		copy(p[n:len(p)-1], bytes.Repeat([]byte("-"), negative))
	}

	state.Write(p)
}
