// +build go1.7

package expression

import (
	"reflect"
	"strings"
	"testing"
)

// projErrorMode will help with error cases and checking error types
type projErrorMode string

const (
	noProjError projErrorMode = ""
	// invalidProjectionOperand error will occur when an invalid OperandBuilder is
	// used as an argument
	invalidProjectionOperand = "BuildOperand error"
	// unsetProjection error will occur if the argument ProjectionBuilder is unset
	unsetProjection = "unset parameter: ProjectionBuilder"
)

func TestProjectionBuilder(t *testing.T) {
	cases := []struct {
		name         string
		input        ProjectionBuilder
		expectedNode exprNode
		err          projErrorMode
	}{
		{
			name:  "names list function call",
			input: NamesList(Name("foo"), Name("bar")),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "$n",
					},
				},
				fmtExpr: "$c, $c",
			},
		},
		{
			name:  "names list method call",
			input: Name("foo").NamesList(Name("bar")),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "$n",
					},
				},
				fmtExpr: "$c, $c",
			},
		},
		{
			name:  "add name",
			input: Name("foo").NamesList(Name("bar")).AddNames(Name("baz"), Name("qux")),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "$n",
					},
					{
						names:   []string{"baz"},
						fmtExpr: "$n",
					}, {
						names:   []string{"qux"},
						fmtExpr: "$n",
					},
				},
				fmtExpr: "$c, $c, $c, $c",
			},
		},
		{
			name:  "invalid operand",
			input: NamesList(Name("")),
			err:   invalidProjectionOperand,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noProjError {
				if err == nil {
					t.Errorf("expect error %q, got no error", c.err)
				} else {
					if e, a := string(c.err), err.Error(); !strings.Contains(a, e) {
						t.Errorf("expect %q error message to be in %q", e, a)
					}
				}
			} else {
				if err != nil {
					t.Errorf("expect no error, got unexpected Error %q", err)
				}
				if e, a := c.expectedNode, actual; !reflect.DeepEqual(a, e) {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}

func TestBuildProjection(t *testing.T) {
	cases := []struct {
		name     string
		input    ProjectionBuilder
		expected string
		err      projErrorMode
	}{
		{
			name:     "build projection 3",
			input:    NamesList(Name("foo"), Name("bar"), Name("baz")),
			expected: "$c, $c, $c",
		},
		{
			name:     "build projection 5",
			input:    NamesList(Name("foo"), Name("bar"), Name("baz")).AddNames(Name("qux"), Name("quux")),
			expected: "$c, $c, $c, $c, $c",
		},
		{
			name:  "empty ProjectionBuilder",
			input: ProjectionBuilder{},
			err:   unsetProjection,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noProjError {
				if err == nil {
					t.Errorf("expect error %q, got no error", c.err)
				} else {
					if e, a := string(c.err), err.Error(); !strings.Contains(a, e) {
						t.Errorf("expect %q error message to be in %q", e, a)
					}
				}
			} else {
				if err != nil {
					t.Errorf("expect no error, got unexpected Error %q", err)
				}
				if e, a := c.expected, actual.fmtExpr; !reflect.DeepEqual(a, e) {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}

func TestBuildProjectionChildNodes(t *testing.T) {
	cases := []struct {
		name     string
		input    ProjectionBuilder
		expected []exprNode
		err      projErrorMode
	}{
		{
			name:  "build child nodes",
			input: NamesList(Name("foo"), Name("bar"), Name("baz")),
			expected: []exprNode{
				{
					names:   []string{"foo"},
					fmtExpr: "$n",
				},
				{
					names:   []string{"bar"},
					fmtExpr: "$n",
				},
				{
					names:   []string{"baz"},
					fmtExpr: "$n",
				},
			},
		},
		{
			name:  "operand error",
			input: NamesList(Name("")),
			err:   invalidProjectionOperand,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noProjError {
				if err == nil {
					t.Errorf("expect error %q, got no error", c.err)
				} else {
					if e, a := string(c.err), err.Error(); !strings.Contains(a, e) {
						t.Errorf("expect %q error message to be in %q", e, a)
					}
				}
			} else {
				if err != nil {
					t.Errorf("expect no error, got unexpected Error %q", err)
				}
				if e, a := c.expected, actual.children; !reflect.DeepEqual(a, e) {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}
