package goabnf

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

var (
	// ErrNoSolutionFound is an error returned when parsing an ABNF
	// grammar and no solution has been found.
	//
	// On the ABNF parsing path it is wrapped by a [*ParseError] carrying the
	// position at which parsing got stuck; errors.Is(err, ErrNoSolutionFound)
	// keeps matching.
	ErrNoSolutionFound = errors.New("no solution found, input ABNF grammar may be invalid")

	// ErrHandlingProseVal is an error returned when an operation tried
	// to produce something on a prose-val, but it can't be handled properly.
	ErrHandlingProseVal = errors.New("can't handle prose-val descriptions")

	// ErrProseValInTransitionGraph is returned when building a transition graph
	// from a grammar that contains a prose-val, which has no defined expansion.
	ErrProseValInTransitionGraph = errors.New("prose value is not supported in transition graphs")

	// ErrRepetitionThreshold is returned when expanding a repetition whose bound
	// exceeds the threshold set with WithRepetitionThreshold.
	ErrRepetitionThreshold = errors.New("repetition threshold reached")

	// ErrRepetitionMinNotOne is returned when a mandatory (max == 1) repetition
	// has a minimum other than 1.
	ErrRepetitionMinNotOne = errors.New("repetition minimum must be 1 when maximum is 1")

	// ErrRepetitionMinGtMax is returned when a repetition has a minimum greater
	// than its maximum.
	ErrRepetitionMinGtMax = errors.New("repetition minimum is greater than maximum")
)

// ParseError reports where parsing input against a grammar got stuck. It is
// returned (wrapped around [ErrNoSolutionFound]) when an input does not parse,
// to help locate the offending construct -- in particular when writing or
// debugging an ABNF grammar.
type ParseError struct {
	// Offset is the byte offset into the input of the furthest position the
	// parser was able to reach before failing.
	Offset int
	// Line and Col are the 1-based line and column of Offset.
	Line, Col int
	// Expected lists the terminals the parser could have consumed at Offset, if
	// that information was available. It may be empty.
	Expected []string
	// Found describes the input byte at Offset (e.g. `%x0A (LF)`), or "EOF" when
	// Offset is at the end of input. It may be empty.
	Found string
	// Hint, when set, gives a plain-language diagnosis of a common mistake (e.g.
	// bare LF line endings where ABNF requires CR LF).
	Hint string
}

var _ error = (*ParseError)(nil)

func (e *ParseError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "parse error at line %d, column %d (offset %d)", e.Line, e.Col, e.Offset)
	switch {
	case len(e.Expected) > 0:
		fmt.Fprintf(&b, ": expected %s", strings.Join(e.Expected, ", "))
		if e.Found != "" {
			fmt.Fprintf(&b, ", found %s", e.Found)
		}
	case e.Found != "":
		fmt.Fprintf(&b, ": unexpected %s", e.Found)
	default:
		fmt.Fprintf(&b, ": %s", ErrNoSolutionFound)
	}
	if e.Hint != "" {
		fmt.Fprintf(&b, " (hint: %s)", e.Hint)
	}
	return b.String()
}

// Unwrap reports ErrNoSolutionFound so that callers matching the historical
// sentinel with errors.Is keep working.
func (e *ParseError) Unwrap() error { return ErrNoSolutionFound }

// describeByte renders an input byte the way ABNF terminals are written, naming
// the control characters that matter most when debugging a grammar.
func describeByte(b byte) string {
	switch b {
	case '\r':
		return "%x0D (CR)"
	case '\n':
		return "%x0A (LF)"
	case '\t':
		return "%x09 (HTAB)"
	case ' ':
		return "%x20 (SP)"
	}
	if b >= 0x20 && b < 0x7f {
		return fmt.Sprintf("%q", string(b))
	}
	return fmt.Sprintf("%%x%02X", b)
}

// newParseError builds a ParseError from a byte offset into input, computing the
// 1-based line and column, the byte found there, and a hint for the bare-LF
// line-ending mistake (the most common ABNF authoring error).
func newParseError(input []byte, offset int, expected []string) *ParseError {
	if offset < 0 {
		offset = 0
	}
	if offset > len(input) {
		offset = len(input)
	}
	line, col := 1, 1
	for i := 0; i < offset; i++ {
		if input[i] == '\n' {
			line++
			col = 1
		} else {
			col++
		}
	}
	exp := append([]string(nil), expected...)
	sort.Strings(exp)
	pe := &ParseError{Offset: offset, Line: line, Col: col, Expected: exp}
	if offset < len(input) {
		pe.Found = describeByte(input[offset])
		// Bare LF where the previous byte is not CR: ABNF lines end with CR LF,
		// so a lone LF is almost always the cause.
		if input[offset] == '\n' && (offset == 0 || input[offset-1] != '\r') {
			pe.Hint = `input uses bare LF line endings, but ABNF requires CR LF ("\r\n")`
		}
	} else {
		pe.Found = "EOF"
	}
	return pe
}

// ErrMultipleSolutionsFound is an error returned when a parser found
// multiple solutions when none or one were expected.
type ErrMultipleSolutionsFound struct{}

var _ error = (*ErrMultipleSolutionsFound)(nil)

func (err ErrMultipleSolutionsFound) Error() string {
	return "multiple solutions found, this should not happen. Please open an issue. This could eventually need an Erratum from IETF tracking"
}

// ErrRuleNotFound is an error returned when the rule was not found
// as part of the grammar.
type ErrRuleNotFound struct {
	Rulename string
}

var _ error = (*ErrRuleNotFound)(nil)

func (err ErrRuleNotFound) Error() string {
	return fmt.Sprintf("rule %s was not found in grammar", err.Rulename)
}

// ErrCoreRuleModify is an error returned when an incremental alternative
// for a core rule.
type ErrCoreRuleModify struct {
	CoreRulename string
}

var _ error = (*ErrCoreRuleModify)(nil)

func (err ErrCoreRuleModify) Error() string {
	return fmt.Sprintf("core rule %s can't be modified", err.CoreRulename)
}

// ErrDependencyNotFound is an error returned during ABNF grammar
// semantic vaildation, if a rule depends on an unexisting rule.
type ErrDependencyNotFound struct {
	Rulename string
}

var _ error = (*ErrDependencyNotFound)(nil)

func (err ErrDependencyNotFound) Error() string {
	return fmt.Sprintf("unsatisfied dependency (rule) %s", err.Rulename)
}

// ErrSemanticRepetition is an error returned during ABNF grammar
// semantic validation, if a repetition has min < max.
type ErrSemanticRepetition struct {
	Repetition Repetition
}

var _ error = (*ErrSemanticRepetition)(nil)

func (err ErrSemanticRepetition) Error() string {
	return fmt.Sprintf("invalid semantic of input ABNF grammar for repetition %s", err.Repetition)
}

// ErrTooLargeNumeral is an error returned when the numeral value
// provided to parse cannot be handled as a 7-bit US-ASCII valid value.
type ErrTooLargeNumeral struct {
	Base, Value string
}

var _ error = (*ErrTooLargeNumeral)(nil)

func (err ErrTooLargeNumeral) Error() string {
	return fmt.Sprintf("too large numeral value %s for base %s", err.Value, err.Base)
}

// ErrDuplicatedRule is an error returned when the rule already
// exist as part of the grammar.
type ErrDuplicatedRule struct {
	Rulename string
}

var _ error = (*ErrDuplicatedRule)(nil)

func (err ErrDuplicatedRule) Error() string {
	return fmt.Sprintf("rule %s was already defined in grammar", err.Rulename)
}

// ErrCyclicRule is an error returned when can't work due to an
// unavoidable cyclic rule.
type ErrCyclicRule struct {
	Rulename string
}

var _ error = (*ErrCyclicRule)(nil)

func (err ErrCyclicRule) Error() string {
	return fmt.Sprintf("can't generate a content as the rule %s involves an unavoidable cycle", err.Rulename)
}

// ErrMaxNodesExceeded is returned by TransitionGraph when construction would
// allocate more nodes than the budget set with WithMaxNodes.
type ErrMaxNodesExceeded struct {
	Max int
}

var _ error = (*ErrMaxNodesExceeded)(nil)

func (err ErrMaxNodesExceeded) Error() string {
	return fmt.Sprintf("transition graph node budget of %d exceeded", err.Max)
}
