package lint

import (
	"go/token"
)

// DisabledInterval contains a single disabled interval and the associated rule name.
type DisabledInterval struct {
	From     token.Position
	To       token.Position
	RuleName string
}

// Rule defines an abstract rule interface
type Rule interface {
	Name() string
	Apply(*File, Arguments) []Failure
}

// AbstractRule defines an abstract rule.
type AbstractRule struct {
	Failures []Failure
}

// ToFailurePosition returns the failure position.
func ToFailurePosition(start token.Pos, end token.Pos, file *File) FailurePosition {
	return FailurePosition{
		Start: file.ToPosition(start),
		End:   file.ToPosition(end),
	}
}
