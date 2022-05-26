package ssh_config

import "fmt"

// Position of a document element within a SSH document.
//
// Line and Col are both 1-indexed positions for the element's line number and
// column number, respectively.  Values of zero or less will cause Invalid(),
// to return true.
type Position struct {
	Line int // line within the document
	Col  int // column within the line
}

// String representation of the position.
// Displays 1-indexed line and column numbers.
func (p Position) String() string {
	return fmt.Sprintf("(%d, %d)", p.Line, p.Col)
}

// Invalid returns whether or not the position is valid (i.e. with negative or
// null values)
func (p Position) Invalid() bool {
	return p.Line <= 0 || p.Col <= 0
}
