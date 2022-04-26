package rule

import (
	"fmt"
	"github.com/mgechev/revive/lint"
	"go/ast"
)

// FlagParamRule lints given else constructs.
type FlagParamRule struct{}

// Apply applies the rule to given file.
func (r *FlagParamRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintFlagParamRule{onFailure: onFailure}
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (r *FlagParamRule) Name() string {
	return "flag-parameter"
}

type lintFlagParamRule struct {
	onFailure func(lint.Failure)
}

func (w lintFlagParamRule) Visit(node ast.Node) ast.Visitor {
	fd, ok := node.(*ast.FuncDecl)
	if !ok {
		return w
	}

	if fd.Body == nil {
		return nil // skip whole function declaration
	}

	for _, p := range fd.Type.Params.List {
		t := p.Type

		id, ok := t.(*ast.Ident)
		if !ok {
			continue
		}

		if id.Name != "bool" {
			continue
		}

		cv := conditionVisitor{p.Names, fd, w}
		ast.Walk(cv, fd.Body)
	}

	return w
}

type conditionVisitor struct {
	ids    []*ast.Ident
	fd     *ast.FuncDecl
	linter lintFlagParamRule
}

func (w conditionVisitor) Visit(node ast.Node) ast.Visitor {
	ifStmt, ok := node.(*ast.IfStmt)
	if !ok {
		return w
	}

	fselect := func(n ast.Node) bool {
		ident, ok := n.(*ast.Ident)
		if !ok {
			return false
		}

		for _, id := range w.ids {
			if ident.Name == id.Name {
				return true
			}
		}

		return false
	}

	uses := pick(ifStmt.Cond, fselect, nil)

	if len(uses) < 1 {
		return w
	}

	w.linter.onFailure(lint.Failure{
		Confidence: 1,
		Node:       w.fd.Type.Params,
		Category:   "bad practice",
		Failure:    fmt.Sprintf("parameter '%s' seems to be a control flag, avoid control coupling", uses[0]),
	})

	return nil
}
