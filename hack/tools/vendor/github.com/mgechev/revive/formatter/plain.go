package formatter

import (
	"fmt"

	"github.com/mgechev/revive/lint"
)

// Plain is an implementation of the Formatter interface
// which formats the errors to JSON.
type Plain struct {
	Metadata lint.FormatterMetadata
}

// Name returns the name of the formatter
func (f *Plain) Name() string {
	return "plain"
}

// Format formats the failures gotten from the lint.
func (f *Plain) Format(failures <-chan lint.Failure, _ lint.Config) (string, error) {
	for failure := range failures {
		fmt.Printf("%v: %s %s\n", failure.Position.Start, failure.Failure, "https://revive.run/r#"+failure.RuleName)
	}
	return "", nil
}
