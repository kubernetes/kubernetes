// Package filters defines a syntax and parser that can be used for the
// filtration of items across the containerd API. The core is built on the
// concept of protobuf field paths, with quoting.  Several operators allow the
// user to flexibly select items based on field presence, equality, inequality
// and regular expressions. Flexible adaptors support working with any type.
//
// The syntax is fairly familiar, if you've used container ecosystem
// projects.  At the core, we base it on the concept of protobuf field
// paths, augmenting with the ability to quote portions of the field path
// to match arbitrary labels. These "selectors" come in the following
// syntax:
//
// ```
// <fieldpath>[<operator><value>]
// ```
//
// A basic example is as follows:
//
// ```
// name==foo
// ```
//
// This would match all objects that have a field `name` with the value
// `foo`. If we only want to test if the field is present, we can omit the
// operator. This is most useful for matching labels in containerd. The
// following will match objects that have the field "labels" and have the
// label "foo" defined:
//
// ```
// labels.foo
// ```
//
// We also allow for quoting of parts of the field path to allow matching
// of arbitrary items:
//
// ```
// labels."very complex label"==something
// ```
//
// We also define `!=` and `~=` as operators. The `!=` will match all
// objects that don't match the value for a field and `~=` will compile the
// target value as a regular expression and match the field value against that.
//
// Selectors can be combined using a comma, such that the resulting
// selector will require all selectors are matched for the object to match.
// The following example will match objects that are named `foo` and have
// the label `bar`:
//
// ```
// name==foo,labels.bar
// ```
//
package filters

import (
	"regexp"

	"github.com/containerd/containerd/log"
)

// Filter matches specific resources based the provided filter
type Filter interface {
	Match(adaptor Adaptor) bool
}

// FilterFunc is a function that handles matching with an adaptor
type FilterFunc func(Adaptor) bool

// Match matches the FilterFunc returning true if the object matches the filter
func (fn FilterFunc) Match(adaptor Adaptor) bool {
	return fn(adaptor)
}

// Always is a filter that always returns true for any type of object
var Always FilterFunc = func(adaptor Adaptor) bool {
	return true
}

// Any allows multiple filters to be matched aginst the object
type Any []Filter

// Match returns true if any of the provided filters are true
func (m Any) Match(adaptor Adaptor) bool {
	for _, m := range m {
		if m.Match(adaptor) {
			return true
		}
	}

	return false
}

// All allows multiple filters to be matched aginst the object
type All []Filter

// Match only returns true if all filters match the object
func (m All) Match(adaptor Adaptor) bool {
	for _, m := range m {
		if !m.Match(adaptor) {
			return false
		}
	}

	return true
}

type operator int

const (
	operatorPresent = iota
	operatorEqual
	operatorNotEqual
	operatorMatches
)

func (op operator) String() string {
	switch op {
	case operatorPresent:
		return "?"
	case operatorEqual:
		return "=="
	case operatorNotEqual:
		return "!="
	case operatorMatches:
		return "~="
	}

	return "unknown"
}

type selector struct {
	fieldpath []string
	operator  operator
	value     string
	re        *regexp.Regexp
}

func (m selector) Match(adaptor Adaptor) bool {
	value, present := adaptor.Field(m.fieldpath)

	switch m.operator {
	case operatorPresent:
		return present
	case operatorEqual:
		return present && value == m.value
	case operatorNotEqual:
		return value != m.value
	case operatorMatches:
		if m.re == nil {
			r, err := regexp.Compile(m.value)
			if err != nil {
				log.L.Errorf("error compiling regexp %q", m.value)
				return false
			}

			m.re = r
		}

		return m.re.MatchString(value)
	default:
		return false
	}
}
