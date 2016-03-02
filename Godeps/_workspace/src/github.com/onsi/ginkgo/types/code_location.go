package types

import (
	"fmt"
)

type CodeLocation struct {
	FileName       string
	LineNumber     int
	FullStackTrace string
}

func (codeLocation CodeLocation) String() string {
	return fmt.Sprintf("%s:%d", codeLocation.FileName, codeLocation.LineNumber)
}
