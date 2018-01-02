package ast

import (
	"reflect"
	"strings"
	"testing"

	"github.com/hashicorp/hcl/hcl/token"
)

func TestObjectListFilter(t *testing.T) {
	var cases = []struct {
		Filter []string
		Input  []*ObjectItem
		Output []*ObjectItem
	}{
		{
			[]string{"foo"},
			[]*ObjectItem{
				&ObjectItem{
					Keys: []*ObjectKey{
						&ObjectKey{
							Token: token.Token{Type: token.STRING, Text: `"foo"`},
						},
					},
				},
			},
			[]*ObjectItem{
				&ObjectItem{
					Keys: []*ObjectKey{},
				},
			},
		},

		{
			[]string{"foo"},
			[]*ObjectItem{
				&ObjectItem{
					Keys: []*ObjectKey{
						&ObjectKey{Token: token.Token{Type: token.STRING, Text: `"foo"`}},
						&ObjectKey{Token: token.Token{Type: token.STRING, Text: `"bar"`}},
					},
				},
				&ObjectItem{
					Keys: []*ObjectKey{
						&ObjectKey{Token: token.Token{Type: token.STRING, Text: `"baz"`}},
					},
				},
			},
			[]*ObjectItem{
				&ObjectItem{
					Keys: []*ObjectKey{
						&ObjectKey{Token: token.Token{Type: token.STRING, Text: `"bar"`}},
					},
				},
			},
		},
	}

	for _, tc := range cases {
		input := &ObjectList{Items: tc.Input}
		expected := &ObjectList{Items: tc.Output}
		if actual := input.Filter(tc.Filter...); !reflect.DeepEqual(actual, expected) {
			t.Fatalf("in order: input, expected, actual\n\n%#v\n\n%#v\n\n%#v", input, expected, actual)
		}
	}
}

func TestWalk(t *testing.T) {
	items := []*ObjectItem{
		&ObjectItem{
			Keys: []*ObjectKey{
				&ObjectKey{Token: token.Token{Type: token.STRING, Text: `"foo"`}},
				&ObjectKey{Token: token.Token{Type: token.STRING, Text: `"bar"`}},
			},
			Val: &LiteralType{Token: token.Token{Type: token.STRING, Text: `"example"`}},
		},
		&ObjectItem{
			Keys: []*ObjectKey{
				&ObjectKey{Token: token.Token{Type: token.STRING, Text: `"baz"`}},
			},
		},
	}

	node := &ObjectList{Items: items}

	order := []string{
		"*ast.ObjectList",
		"*ast.ObjectItem",
		"*ast.ObjectKey",
		"*ast.ObjectKey",
		"*ast.LiteralType",
		"*ast.ObjectItem",
		"*ast.ObjectKey",
	}
	count := 0

	Walk(node, func(n Node) (Node, bool) {
		if n == nil {
			return n, false
		}

		typeName := reflect.TypeOf(n).String()
		if order[count] != typeName {
			t.Errorf("expected '%s' got: '%s'", order[count], typeName)
		}
		count++
		return n, true
	})
}

func TestWalkEquality(t *testing.T) {
	items := []*ObjectItem{
		&ObjectItem{
			Keys: []*ObjectKey{
				&ObjectKey{Token: token.Token{Type: token.STRING, Text: `"foo"`}},
			},
		},
		&ObjectItem{
			Keys: []*ObjectKey{
				&ObjectKey{Token: token.Token{Type: token.STRING, Text: `"bar"`}},
			},
		},
	}

	node := &ObjectList{Items: items}

	rewritten := Walk(node, func(n Node) (Node, bool) { return n, true })

	newNode, ok := rewritten.(*ObjectList)
	if !ok {
		t.Fatalf("expected Objectlist, got %T", rewritten)
	}

	if !reflect.DeepEqual(node, newNode) {
		t.Fatal("rewritten node is not equal to the given node")
	}

	if len(newNode.Items) != 2 {
		t.Error("expected newNode length 2, got: %d", len(newNode.Items))
	}

	expected := []string{
		`"foo"`,
		`"bar"`,
	}

	for i, item := range newNode.Items {
		if len(item.Keys) != 1 {
			t.Error("expected keys newNode length 1, got: %d", len(item.Keys))
		}

		if item.Keys[0].Token.Text != expected[i] {
			t.Errorf("expected key %s, got %s", expected[i], item.Keys[0].Token.Text)
		}

		if item.Val != nil {
			t.Errorf("expected item value should be nil")
		}
	}
}

func TestWalkRewrite(t *testing.T) {
	items := []*ObjectItem{
		&ObjectItem{
			Keys: []*ObjectKey{
				&ObjectKey{Token: token.Token{Type: token.STRING, Text: `"foo"`}},
				&ObjectKey{Token: token.Token{Type: token.STRING, Text: `"bar"`}},
			},
		},
		&ObjectItem{
			Keys: []*ObjectKey{
				&ObjectKey{Token: token.Token{Type: token.STRING, Text: `"baz"`}},
			},
		},
	}

	node := &ObjectList{Items: items}

	suffix := "_example"
	node = Walk(node, func(n Node) (Node, bool) {
		switch i := n.(type) {
		case *ObjectKey:
			i.Token.Text = i.Token.Text + suffix
			n = i
		}
		return n, true
	}).(*ObjectList)

	Walk(node, func(n Node) (Node, bool) {
		switch i := n.(type) {
		case *ObjectKey:
			if !strings.HasSuffix(i.Token.Text, suffix) {
				t.Errorf("Token '%s' should have suffix: %s", i.Token.Text, suffix)
			}
		}
		return n, true
	})

}
