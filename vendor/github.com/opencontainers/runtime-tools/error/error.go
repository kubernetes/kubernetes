// Package error implements generic tooling for tracking RFC 2119
// violations and linking back to the appropriate specification section.
package error

import (
	"fmt"
	"strings"
)

// Level represents the RFC 2119 compliance levels
type Level int

const (
	// MAY-level

	// May represents 'MAY' in RFC 2119.
	May Level = iota
	// Optional represents 'OPTIONAL' in RFC 2119.
	Optional

	// SHOULD-level

	// Should represents 'SHOULD' in RFC 2119.
	Should
	// ShouldNot represents 'SHOULD NOT' in RFC 2119.
	ShouldNot
	// Recommended represents 'RECOMMENDED' in RFC 2119.
	Recommended
	// NotRecommended represents 'NOT RECOMMENDED' in RFC 2119.
	NotRecommended

	// MUST-level

	// Must represents 'MUST' in RFC 2119
	Must
	// MustNot represents 'MUST NOT' in RFC 2119.
	MustNot
	// Shall represents 'SHALL' in RFC 2119.
	Shall
	// ShallNot represents 'SHALL NOT' in RFC 2119.
	ShallNot
	// Required represents 'REQUIRED' in RFC 2119.
	Required
)

// Error represents an error with compliance level and specification reference.
type Error struct {
	// Level represents the RFC 2119 compliance level.
	Level Level

	// Reference is a URL for the violated specification requirement.
	Reference string

	// Err holds additional details about the violation.
	Err error
}

// ParseLevel takes a string level and returns the RFC 2119 compliance level constant.
func ParseLevel(level string) (Level, error) {
	switch strings.ToUpper(level) {
	case "MAY":
		fallthrough
	case "OPTIONAL":
		return May, nil
	case "SHOULD":
		fallthrough
	case "SHOULDNOT":
		fallthrough
	case "RECOMMENDED":
		fallthrough
	case "NOTRECOMMENDED":
		return Should, nil
	case "MUST":
		fallthrough
	case "MUSTNOT":
		fallthrough
	case "SHALL":
		fallthrough
	case "SHALLNOT":
		fallthrough
	case "REQUIRED":
		return Must, nil
	}

	var l Level
	return l, fmt.Errorf("%q is not a valid compliance level", level)
}

// String takes a RFC 2119 compliance level constant and returns a string representation.
func (level Level) String() string {
	switch level {
	case May:
		return "MAY"
	case Optional:
		return "OPTIONAL"
	case Should:
		return "SHOULD"
	case ShouldNot:
		return "SHOULD NOT"
	case Recommended:
		return "RECOMMENDED"
	case NotRecommended:
		return "NOT RECOMMENDED"
	case Must:
		return "MUST"
	case MustNot:
		return "MUST NOT"
	case Shall:
		return "SHALL"
	case ShallNot:
		return "SHALL NOT"
	case Required:
		return "REQUIRED"
	}

	panic(fmt.Sprintf("%d is not a valid compliance level", level))
}

// Error returns the error message with specification reference.
func (err *Error) Error() string {
	return fmt.Sprintf("%s\nRefer to: %s", err.Err.Error(), err.Reference)
}
