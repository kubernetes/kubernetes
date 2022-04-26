package rule

import (
	"go/ast"
	"go/token"

	"github.com/mgechev/revive/lint"
)

// UnnecessaryStmtRule warns on unnecessary statements.
type UnnecessaryStmtRule struct{}

// Apply applies the rule to given file.
func (r *UnnecessaryStmtRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure
	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintUnnecessaryStmtRule{onFailure}
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (r *UnnecessaryStmtRule) Name() string {
	return "unnecessary-stmt"
}

type lintUnnecessaryStmtRule struct {
	onFailure func(lint.Failure)
}

func (w lintUnnecessaryStmtRule) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.FuncDecl:
		if n.Body == nil || n.Type.Results != nil {
			return w
		}
		stmts := n.Body.List
		if len(stmts) == 0 {
			return w
		}

		lastStmt := stmts[len(stmts)-1]
		rs, ok := lastStmt.(*ast.ReturnStmt)
		if !ok {
			return w
		}

		if len(rs.Results) == 0 {
			w.newFailure(lastStmt, "omit unnecessary return statement")
		}

	case *ast.SwitchStmt:
		w.checkSwitchBody(n.Body)
	case *ast.TypeSwitchStmt:
		w.checkSwitchBody(n.Body)
	case *ast.CaseClause:
		if n.Body == nil {
			return w
		}
		stmts := n.Body
		if len(stmts) == 0 {
			return w
		}

		lastStmt := stmts[len(stmts)-1]
		rs, ok := lastStmt.(*ast.BranchStmt)
		if !ok {
			return w
		}

		if rs.Tok == token.BREAK && rs.Label == nil {
			w.newFailure(lastStmt, "omit unnecessary break at the end of case clause")
		}
	}

	return w
}

func (w lintUnnecessaryStmtRule) checkSwitchBody(b *ast.BlockStmt) {
	cases := b.List
	if len(cases) != 1 {
		return
	}

	cc, ok := cases[0].(*ast.CaseClause)
	if !ok {
		return
	}

	if len(cc.List) > 1 { // skip cases with multiple expressions
		return
	}

	w.newFailure(b, "switch with only one case can be replaced by an if-then")
}

func (w lintUnnecessaryStmtRule) newFailure(node ast.Node, msg string) {
	w.onFailure(lint.Failure{
		Confidence: 1,
		Node:       node,
		Category:   "style",
		Failure:    msg,
	})
}
