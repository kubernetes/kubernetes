package gosec

import (
	"sort"
)

// Error is used when there are golang errors while parsing the AST
type Error struct {
	Line   int    `json:"line"`
	Column int    `json:"column"`
	Err    string `json:"error"`
}

// NewError creates Error object
func NewError(line, column int, err string) *Error {
	return &Error{
		Line:   line,
		Column: column,
		Err:    err,
	}
}

// sortErrors sorts the golang errors by line
func sortErrors(allErrors map[string][]Error) {
	for _, errors := range allErrors {
		sort.Slice(errors, func(i, j int) bool {
			if errors[i].Line == errors[j].Line {
				return errors[i].Column <= errors[j].Column
			}
			return errors[i].Line < errors[j].Line
		})
	}
}
