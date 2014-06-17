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

package apiserver

import (
	"fmt"
	"strings"
)

// Labels allows you to present labels.
type Labels interface {
	Get(label string) (value string)
}

// A map of label:value.
type LabelSet map[string]string

func (ls LabelSet) String() string {
	query := make([]string, 0, len(ls))
	for key, value := range ls {
		query = append(query, key+"="+value)
	}
	return strings.Join(query, ",")
}

func (ls LabelSet) Get(label string) string {
	return ls[label]
}

// Represents a query.
type LabelQuery interface {
	// Returns true if this query matches the given set of labels.
	Matches(Labels) bool

	// Prints a human readable version of this label query.
	String() string
}

// A single term of a label query.
type labelQueryTerm struct {
	// Not inverts the meaning of the items in this term.
	not bool

	// Exactly one of the below three items should be used.

	// If non-nil, we match LabelSet l iff l[*label] == *value.
	label, value *string

	// A list of terms which must all match for this query term to return true.
	and []labelQueryTerm

	// A list of terms, any one of which will cause this query term to return true.
	// Parsing/printing not implemented.
	or []labelQueryTerm
}

func (l *labelQueryTerm) Matches(ls Labels) bool {
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

// Takes a string repsenting a label query and returns an object suitable for matching, or an error.
func ParseLabelQuery(query string) (LabelQuery, error) {
	parts := strings.Split(query, ",")
	var items []labelQueryTerm
	for _, part := range parts {
		if part == "" {
			continue
		}
		if lhs, rhs, ok := try(part, "!="); ok {
			items = append(items, labelQueryTerm{not: true, label: &lhs, value: &rhs})
		} else if lhs, rhs, ok := try(part, "=="); ok {
			items = append(items, labelQueryTerm{label: &lhs, value: &rhs})
		} else if lhs, rhs, ok := try(part, "="); ok {
			items = append(items, labelQueryTerm{label: &lhs, value: &rhs})
		} else {
			return nil, fmt.Errorf("invalid label query: '%s'; can't understand '%s'", query, part)
		}
	}
	if len(items) == 1 {
		return &items[0], nil
	}
	return &labelQueryTerm{and: items}, nil
}

// Returns this query as a string in a form that ParseLabelQuery can parse.
func (l *labelQueryTerm) String() (out string) {
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

/*
type parseErr struct {
	Reason string
	Pos token.Pos
}

func (p parseErr) Error() string {
	return fmt.Sprintf("%v: %v", p.Reason, p.Pos)
}

func fromLiteral(expr *ast.BinaryExpr) (*labelQueryTerm, error) {
	lhs, ok := expr.X.(*ast.Ident)
	if !ok {
		return nil, parseErr{"expected literal", expr.X.Pos()}
	}

}

func fromBinaryExpr(expr *ast.BinaryExpr) (*labelQueryTerm, error) {
	switch expr.Op {
		case token.EQL, token.NEQ:
			return fromLiteral(expr)
	}
	lhs, err := fromExpr(expr.X)
	if err != nil {
		return nil, err
	}
	rhs, err := fromExpr(expr.Y)
	if err != nil {
		return nil, err
	}
	switch expr.Op {
		case token.AND, token.LAND:
			return &labelQueryTerm{And: []LabelQuery{lhs, rhs}}
		case token.OR, token.LOR:
			return &labelQueryTerm{Or: []LabelQuery{lhs, rhs}}
	}
}

func fromUnaryExpr(expr *ast.UnaryExpr) (*labelQueryTerm, error) {
	if expr.Op == token.NOT {
		lqt, err := fromExpr(expr.X)
		if err != nil {
			return nil, err
		}
		lqt.not = !lqt.not
		return lqt, nil
	}
	return nil, parseErr{"unrecognized unary expression", expr.OpPos}
}

func fromExpr(expr ast.Expr) (*labelQueryTerm, error) {
	switch v := expr.(type) {
		case *ast.UnaryExpr:
			return fromUnaryExpr(v)
		case *ast.BinaryExpr:
			return fromBinaryExpr(v)
	}
	return nil, parseErr{"unrecognized expression type", expr.Pos()}
}

// Takes a string repsenting a label query and returns an object suitable for matching, or an error.
func ParseLabelQuery(query string) (LabelQuery, error) {
	expr, err := parser.ParseExpr(query)
	if err != nil {
		return nil, err
	}
	log.Printf("%v: %v (%#v)", query, expr, expr)
	return fromExpr(expr)
}
*/
