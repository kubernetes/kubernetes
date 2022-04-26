package syntax

import (
	"fmt"
)

type ParseError struct {
	Pos     Position
	Message string
}

func (e ParseError) Error() string { return e.Message }

func throwfPos(pos Position, format string, args ...interface{}) {
	panic(ParseError{
		Pos:     pos,
		Message: fmt.Sprintf(format, args...),
	})
}

func throwErrorf(posBegin, posEnd int, format string, args ...interface{}) {
	pos := Position{
		Begin: uint16(posBegin),
		End:   uint16(posEnd),
	}
	throwfPos(pos, format, args...)
}
