package rule

import (
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// IdenticalBranchesRule warns on constant logical expressions.
type IdenticalBranchesRule struct{}

// Apply applies the rule to given file.
func (r *IdenticalBranchesRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	astFile := file.AST
	w := &lintIdenticalBranches{astFile, onFailure}
	ast.Walk(w, astFile)
	return failures
}

// Name returns the rule name.
func (r *IdenticalBranchesRule) Name() string {
	return "identical-branches"
}

type lintIdenticalBranches struct {
	file      *ast.File
	onFailure func(lint.Failure)
}

func (w *lintIdenticalBranches) Visit(node ast.Node) ast.Visitor {
	n, ok := node.(*ast.IfStmt)
	if !ok {
		return w
	}

	if n.Else == nil {
		return w
	}
	branches := []*ast.BlockStmt{n.Body}

	elseBranch, ok := n.Else.(*ast.BlockStmt)
	if !ok { // if-else-if construction
		return w
	}
	branches = append(branches, elseBranch)

	if w.identicalBranches(branches) {
		w.newFailure(n, "both branches of the if are identical")
	}

	return w
}

func (w *lintIdenticalBranches) identicalBranches(branches []*ast.BlockStmt) bool {
	if len(branches) < 2 {
		return false
	}

	ref := gofmt(branches[0])
	for i := 1; i < len(branches); i++ {
		if gofmt(branches[i]) != ref {
			return false
		}
	}

	return true
}

func (w lintIdenticalBranches) newFailure(node ast.Node, msg string) {
	w.onFailure(lint.Failure{
		Confidence: 1,
		Node:       node,
		Category:   "logic",
		Failure:    msg,
	})
}
