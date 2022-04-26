package lintutil

import (
	"go/ast"

	"golang.org/x/tools/go/ast/astutil"
)

// FindNode applies pred for root and all it's childs until it returns true.
// If followFunc is defined, it's called before following any node to check whether it needs to be followed.
// followFunc has to return true in order to continuing traversing the node and return false otherwise.
// Matched node is returned.
// If none of the nodes matched predicate, nil is returned.
func FindNode(root ast.Node, followFunc, pred func(ast.Node) bool) ast.Node {
	var (
		found   ast.Node
		preFunc func(*astutil.Cursor) bool
	)

	if followFunc != nil {
		preFunc = func(cur *astutil.Cursor) bool {
			return followFunc(cur.Node())
		}
	}

	astutil.Apply(root,
		preFunc,
		func(cur *astutil.Cursor) bool {
			if pred(cur.Node()) {
				found = cur.Node()
				return false
			}
			return true
		})
	return found
}

// ContainsNode reports whether `FindNode(root, pred)!=nil`.
func ContainsNode(root ast.Node, pred func(ast.Node) bool) bool {
	return FindNode(root, nil, pred) != nil
}
