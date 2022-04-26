package lint

import (
	"go/ast"
	"go/token"
)

const (
	// SeverityWarning declares failures of type warning
	SeverityWarning = "warning"
	// SeverityError declares failures of type error.
	SeverityError = "error"
)

// Severity is the type for the failure types.
type Severity string

// FailurePosition returns the failure position
type FailurePosition struct {
	Start token.Position
	End   token.Position
}

// Failure defines a struct for a linting failure.
type Failure struct {
	Failure    string
	RuleName   string
	Category   string
	Position   FailurePosition
	Node       ast.Node `json:"-"`
	Confidence float64
	// For future use
	ReplacementLine string
}

// GetFilename returns the filename.
func (f *Failure) GetFilename() string {
	return f.Position.Start.Filename
}
