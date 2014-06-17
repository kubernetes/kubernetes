/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package labels

import (
	"fmt"
	"strings"
)

// Represents a label query.
type Query interface {
	// Returns true if this query matches the given set of labels.
	Matches(Labels) bool

	// Prints a human readable version of this label query.
	String() string
}

// Everything returns a query that matches all labels.
func Everything() Query {
	return &queryTerm{}
}

// A single term of a label query.
type queryTerm struct {
	// Not inverts the meaning of the items in this term.
	not bool

	// Exactly one of the below three items should be used.

	// If non-nil, we match Set l iff l[*label] == *value.
	label, value *string

	// A list of terms which must all match for this query term to return true.
	and []queryTerm

	// A list of terms, any one of which will cause this query term to return true.
	// Parsing/printing not implemented.
	or []queryTerm
}

func (l *queryTerm) Matches(ls Labels) bool {
	matches := !l.not
	switch {
	case l.label != nil && l.value != nil:
		if ls.Get(*l.label) == *l.value {
			return matches
		}
		return !matches
	case len(l.and) > 0:
		for i := range l.and {
			if !l.and[i].Matches(ls) {
				return !matches
			}
		}
		return matches
	case len(l.or) > 0:
		for i := range l.or {
			if l.or[i].Matches(ls) {
				return matches
			}
		}
		return !matches
	}

	// Empty queries match everything
	return matches
}

func try(queryPiece, op string) (lhs, rhs string, ok bool) {
	pieces := strings.Split(queryPiece, op)
	if len(pieces) == 2 {
		return pieces[0], pieces[1], true
	}
	return "", "", false
}

// Given a Set, return a Query which will match exactly that Set.
func QueryFromSet(ls Set) Query {
	var query queryTerm
	for l, v := range ls {
		// Make a copy, because we're taking the address below
		label, value := l, v
		query.and = append(query.and, queryTerm{label: &label, value: &value})
	}
	return &query
}

// Takes a string repsenting a label query and returns an object suitable for matching, or an error.
func ParseQuery(query string) (Query, error) {
	parts := strings.Split(query, ",")
	var items []queryTerm
	for _, part := range parts {
		if part == "" {
			continue
		}
		if lhs, rhs, ok := try(part, "!="); ok {
			items = append(items, queryTerm{not: true, label: &lhs, value: &rhs})
		} else if lhs, rhs, ok := try(part, "=="); ok {
			items = append(items, queryTerm{label: &lhs, value: &rhs})
		} else if lhs, rhs, ok := try(part, "="); ok {
			items = append(items, queryTerm{label: &lhs, value: &rhs})
		} else {
			return nil, fmt.Errorf("invalid label query: '%s'; can't understand '%s'", query, part)
		}
	}
	if len(items) == 1 {
		return &items[0], nil
	}
	return &queryTerm{and: items}, nil
}

// Returns this query as a string in a form that ParseQuery can parse.
func (l *queryTerm) String() (out string) {
	if len(l.and) > 0 {
		for _, part := range l.and {
			if out != "" {
				out += ","
			}
			out += part.String()
		}
		return
	} else if l.label != nil && l.value != nil {
		op := "="
		if l.not {
			op = "!="
		}
		return fmt.Sprintf("%v%v%v", *l.label, op, *l.value)
	}
	return ""
}
