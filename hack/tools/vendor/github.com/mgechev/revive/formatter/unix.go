package formatter

import (
	"fmt"

	"github.com/mgechev/revive/lint"
)

// Unix is an implementation of the Formatter interface
// which formats the errors to a simple line based error format
//  main.go:24:9: [errorf] should replace errors.New(fmt.Sprintf(...)) with fmt.Errorf(...)
type Unix struct {
	Metadata lint.FormatterMetadata
}

// Name returns the name of the formatter
func (f *Unix) Name() string {
	return "unix"
}

// Format formats the failures gotten from the lint.
func (f *Unix) Format(failures <-chan lint.Failure, _ lint.Config) (string, error) {
	for failure := range failures {
		fmt.Printf("%v: [%s] %s\n", failure.Position.Start, failure.RuleName, failure.Failure)
	}
	return "", nil
}
