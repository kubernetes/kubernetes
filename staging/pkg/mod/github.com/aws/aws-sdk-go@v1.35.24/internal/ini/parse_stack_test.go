// +build go1.7

package ini

import (
	"reflect"
	"testing"
)

func newMockAST(v []rune) AST {
	return newASTWithRootToken(ASTKindNone, Token{raw: v})
}

func TestStack(t *testing.T) {
	cases := []struct {
		asts     []AST
		expected []AST
	}{
		{
			asts: []AST{
				newMockAST([]rune("0")),
				newMockAST([]rune("1")),
				newMockAST([]rune("2")),
				newMockAST([]rune("3")),
				newMockAST([]rune("4")),
			},
			expected: []AST{
				newMockAST([]rune("0")),
				newMockAST([]rune("1")),
				newMockAST([]rune("2")),
				newMockAST([]rune("3")),
				newMockAST([]rune("4")),
			},
		},
	}

	for _, c := range cases {
		p := newParseStack(10, 10)
		for _, ast := range c.asts {
			p.Push(ast)
			p.MarkComplete(ast)
		}

		if e, a := len(c.expected), p.Len(); e != a {
			t.Errorf("expected the same legnth with %d, but received %d", e, a)
		}
		for i := len(c.expected) - 1; i >= 0; i-- {
			if e, a := c.expected[i], p.Pop(); !reflect.DeepEqual(e, a) {
				t.Errorf("stack element %d invalid: expected %v, but received %v", i, e, a)
			}
		}

		if e, a := len(c.expected), p.index; e != a {
			t.Errorf("expected %d, but received %d", e, a)
		}

		if e, a := c.asts, p.list[:p.index]; !reflect.DeepEqual(e, a) {
			t.Errorf("expected %v, but received %v", e, a)
		}
	}
}
