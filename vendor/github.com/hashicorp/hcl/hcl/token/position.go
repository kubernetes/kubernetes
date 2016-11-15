package token

import "fmt"

// Pos describes an arbitrary source position
// including the file, line, and column location.
// A Position is valid if the line number is > 0.
type Pos struct {
	Filename string // filename, if any
	Offset   int    // offset, starting at 0
	Line     int    // line number, starting at 1
	Column   int    // column number, starting at 1 (character count)
}

// IsValid returns true if the position is valid.
func (p *Pos) IsValid() bool { return p.Line > 0 }

// String returns a string in one of several forms:
//
//	file:line:column    valid position with file name
//	line:column         valid position without file name
//	file                invalid position with file name
//	-                   invalid position without file name
func (p Pos) String() string {
	s := p.Filename
	if p.IsValid() {
		if s != "" {
			s += ":"
		}
		s += fmt.Sprintf("%d:%d", p.Line, p.Column)
	}
	if s == "" {
		s = "-"
	}
	return s
}

// Before reports whether the position p is before u.
func (p Pos) Before(u Pos) bool {
	return u.Offset > p.Offset || u.Line > p.Line
}

// After reports whether the position p is after u.
func (p Pos) After(u Pos) bool {
	return u.Offset < p.Offset || u.Line < p.Line
}
