package goabnf

import (
	"fmt"
	"strconv"
	"strings"
)

// ElemItf defines the interface of all the element alternations:
// - ElemRulename
// - ElemGroup
// - ElemOption
// - ElemProseVal
// - ElemNumVal
// - ElemCharVal
//
// This is exposed for custom evaluation purposes, please don't use it else.
type ElemItf interface {
	fmt.Stringer
}

// Rule represents an ABNF rule, with its name and underlying alternation.
//
// This is exposed for custom evaluation purposes, please don't use it else.
type Rule struct {
	// Name is the unique rule name.
	// Notice it is case-insensitive according to RFC 5234 Section 2.1.
	Name string

	Alternation Alternation
}

func (rl Rule) String() string {
	return fmt.Sprintf("%s = %s", rl.Name, rl.Alternation)
}

// Alternation represents an ABNF alternation object.
//
// This is exposed for custom evaluation purposes, please don't use it else.
type Alternation struct {
	// Concatenations contains the variants that validate the upper
	// object (i.e. rule, group or option).
	Concatenations []Concatenation
}

func (alt Alternation) String() string {
	str := ""
	for _, concat := range alt.Concatenations {
		str += fmt.Sprintf("%s / ", concat)
	}
	return strings.TrimSuffix(str, " / ")
}

// Concatenation represents an ABNF concatenation object.
//
// This is exposed for custom evaluation purposes, please don't use it else.
type Concatenation struct {
	// Repetitions contains the following repetitions, order matter.
	Repetitions []Repetition
}

func (cnt Concatenation) String() string {
	str := ""
	for _, rep := range cnt.Repetitions {
		str += fmt.Sprintf("%s ", rep)
	}
	return strings.TrimSuffix(str, " ")
}

// Repetition represents an ABNF repetition object.
//
// This is exposed for custom evaluation purposes, please don't use it else.
type Repetition struct {
	Min, Max int
	Element  ElemItf
}

func (rep Repetition) String() string {
	if rep.Min == rep.Max {
		if rep.Min == 1 {
			return rep.Element.String()
		}
		return strconv.Itoa(rep.Min) + rep.Element.String()
	}
	str := ""
	if rep.Min != 0 {
		str += strconv.Itoa(rep.Min)
	}
	str += "*"
	if rep.Max != inf {
		str += strconv.Itoa(rep.Max)
	}
	return str + rep.Element.String()
}

// ElemRulename represents an ABNF rulename object.
//
// This is exposed for custom evaluation purposes, please don't use it else.
type ElemRulename struct {
	// Name is the unique rule name.
	// Notice it is case-insensitive according to RFC 5234 Section 2.1.
	Name string
}

func (erln ElemRulename) String() string {
	return erln.Name
}

var _ ElemItf = (*ElemRulename)(nil)

// ElemGroup represents an ABNF group object.
// It will be straightly passed through with its underlying alternation.
//
// This is exposed for custom evaluation purposes, please don't use it else.
type ElemGroup struct {
	Alternation Alternation
}

func (egrp ElemGroup) String() string {
	return "(" + egrp.Alternation.String() + ")"
}

var _ ElemItf = (*ElemGroup)(nil)

// ElemOption represents an ABNF option object.
// It will be straightly converted to a 0*1 repetition.
//
// This is exposed for custom evaluation purposes, please don't use it else.
type ElemOption struct {
	Alternation Alternation
}

func (eopt ElemOption) String() string {
	return "[" + eopt.Alternation.String() + "]"
}

var _ ElemItf = (*ElemOption)(nil)

// ElemCharVal represents an ABNF char-val object.
//
// This is exposed for custom evaluation purposes, please don't use it else.
type ElemCharVal struct {
	// Sensitive is by default false, support clarrified by RFC 7405 hence
	// only works on a-z and A-Z.
	Sensitive bool
	Values    []rune
}

func (ecvl ElemCharVal) String() string {
	str := ""
	for _, val := range ecvl.Values {
		str += string(val)
	}
	return `"` + str + `"`
}

var _ ElemItf = (*ElemCharVal)(nil)

// ElemNumVal represents an ABNF num-val object.
//
// This is exposed for custom evaluation purposes, please don't use it else.
type ElemNumVal struct {
	Base string
	// Status could be:
	// - `statSeries`: `elems` contains all the expected
	//   values in the order of the grammar defined them ;
	// - `statRange`: `elems` contains the start and end
	//   bounds (so no more than two).
	Status Status
	Elems  []string
}

func (envl ElemNumVal) String() string {
	str := "%" + envl.Base
	spl := "."
	if envl.Status == StatRange {
		spl = "-"
	}
	for _, val := range envl.Elems {
		str += val + spl
	}
	return strings.TrimSuffix(str, spl)
}

var _ ElemItf = (*ElemNumVal)(nil)

// ElemProseVal represents an ABNF prose-val object.
//
// This is exposed for custom evaluation purposes, please don't use it else.
type ElemProseVal struct {
	values []string
}

func (epvl ElemProseVal) String() string {
	str := ""
	for _, val := range epvl.values {
		str += val
	}
	return "<" + str + ">"
}

var _ ElemItf = (*ElemProseVal)(nil)

// Status defines the type of a num-val, rather StatSeries or StatRange.
//
// This is exposed for custom evaluation purposes, please don't use it else.
type Status int

const (
	// StatSeries represents a serie of unique byte value.
	StatSeries Status = iota
	// StatRange represents an interval of possible bytes.
	StatRange
)

// ABNF is the manually parsed+evaluated+validated ABNF grammar.
var ABNF = &Grammar{
	Rulemap: map[string]*Rule{
		abnfRulelist.Name:              abnfRulelist,
		abnfRule.Name:                  abnfRule,
		abnfRulename.Name:              abnfRulename,
		abnfDefinedAs.Name:             abnfDefinedAs,
		abnfElements.Name:              abnfElements,
		abnfCWsp.Name:                  abnfCWsp,
		abnfCNl.Name:                   abnfCNl,
		abnfComment.Name:               abnfComment,
		abnfAlternation.Name:           abnfAlternation,
		abnfConcatenation.Name:         abnfConcatenation,
		abnfRepetition.Name:            abnfRepetition,
		abnfRepeat.Name:                abnfRepeat,
		abnfElement.Name:               abnfElement,
		abnfGroup.Name:                 abnfGroup,
		abnfOption.Name:                abnfOption,
		abnfCharVal.Name:               abnfCharVal,
		abnfCaseInsensitiveString.Name: abnfCaseInsensitiveString,
		abnfCaseSensitiveString.Name:   abnfCaseSensitiveString,
		abnfQuotedString.Name:          abnfQuotedString,
		abnfNumVal.Name:                abnfNumVal,
		abnfBinVal.Name:                abnfBinVal,
		abnfDecVal.Name:                abnfDecVal,
		abnfHexVal.Name:                abnfHexVal,
		abnfProseVal.Name:              abnfProseVal,
	},
}
