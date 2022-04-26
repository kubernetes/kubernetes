package lintutil

import (
	"go/ast"

	"github.com/go-toolsmith/astequal"
)

// AstSet is a simple ast.Node set.
// Zero value is ready to use set.
// Can be reused after Clear call.
type AstSet struct {
	items []ast.Node
}

// Contains reports whether s contains x.
func (s *AstSet) Contains(x ast.Node) bool {
	for i := range s.items {
		if astequal.Node(s.items[i], x) {
			return true
		}
	}
	return false
}

// Insert pushes x in s if it's not already there.
// Returns true if element was inserted.
func (s *AstSet) Insert(x ast.Node) bool {
	if s.Contains(x) {
		return false
	}
	s.items = append(s.items, x)
	return true
}

// Clear removes all element from set.
func (s *AstSet) Clear() {
	s.items = s.items[:0]
}

// Len returns the number of elements contained inside s.
func (s *AstSet) Len() int {
	return len(s.items)
}
