package ini

import (
	"reflect"
	"testing"
)

func TestTrimSpaces(t *testing.T) {
	cases := []struct {
		name         string
		node         AST
		expectedNode AST
	}{
		{
			name: "simple case",
			node: AST{
				Root: Token{
					raw: []rune("foo"),
				},
			},
			expectedNode: AST{
				Root: Token{
					raw: []rune("foo"),
				},
			},
		},
		{
			name: "LHS case",
			node: AST{
				Root: Token{
					raw: []rune("         foo"),
				},
			},
			expectedNode: AST{
				Root: Token{
					raw: []rune("foo"),
				},
			},
		},
		{
			name: "RHS case",
			node: AST{
				Root: Token{
					raw: []rune("foo     "),
				},
			},
			expectedNode: AST{
				Root: Token{
					raw: []rune("foo"),
				},
			},
		},
		{
			name: "both sides case",
			node: AST{
				Root: Token{
					raw: []rune(" foo "),
				},
			},
			expectedNode: AST{
				Root: Token{
					raw: []rune("foo"),
				},
			},
		},
	}

	for _, c := range cases {
		node := trimSpaces(c.node)

		if e, a := c.expectedNode, node; !reflect.DeepEqual(e, a) {
			t.Errorf("%s: expected %v, but received %v", c.name, e, a)
		}
	}
}
