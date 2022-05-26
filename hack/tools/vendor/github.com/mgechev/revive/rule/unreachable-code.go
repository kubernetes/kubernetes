package rule

import (
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// UnreachableCodeRule lints unreachable code.
type UnreachableCodeRule struct{}

// Apply applies the rule to given file.
func (r *UnreachableCodeRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure
	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	var branchingFunctions = map[string]map[string]bool{
		"os": {"Exit": true},
		"log": {
			"Fatal":   true,
			"Fatalf":  true,
			"Fatalln": true,
			"Panic":   true,
			"Panicf":  true,
			"Panicln": true,
		},
	}

	w := lintUnreachableCode{onFailure, branchingFunctions}
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (r *UnreachableCodeRule) Name() string {
	return "unreachable-code"
}

type lintUnreachableCode struct {
	onFailure          func(lint.Failure)
	branchingFunctions map[string]map[string]bool
}

func (w lintUnreachableCode) Visit(node ast.Node) ast.Visitor {
	blk, ok := node.(*ast.BlockStmt)
	if !ok {
		return w
	}

	if len(blk.List) < 2 {
		return w
	}
loop:
	for i, stmt := range blk.List[:len(blk.List)-1] {
		// println("iterating ", len(blk.List))
		next := blk.List[i+1]
		if _, ok := next.(*ast.LabeledStmt); ok {
			continue // skip if next statement is labeled
		}

		switch s := stmt.(type) {
		case *ast.ReturnStmt:
			w.onFailure(newUnreachableCodeFailure(s))
			break loop
		case *ast.BranchStmt:
			token := s.Tok.String()
			if token != "fallthrough" {
				w.onFailure(newUnreachableCodeFailure(s))
				break loop
			}
		case *ast.ExprStmt:
			ce, ok := s.X.(*ast.CallExpr)
			if !ok {
				continue
			}
			// it's a function call
			fc, ok := ce.Fun.(*ast.SelectorExpr)
			if !ok {
				continue
			}

			id, ok := fc.X.(*ast.Ident)

			if !ok {
				continue
			}
			fn := fc.Sel.Name
			pkg := id.Name
			if !w.branchingFunctions[pkg][fn] { // it isn't a call to a branching function
				continue
			}

			if _, ok := next.(*ast.ReturnStmt); ok { // return statement needed to satisfy function signature
				continue
			}

			w.onFailure(newUnreachableCodeFailure(s))
			break loop
		}
	}

	return w
}

func newUnreachableCodeFailure(node ast.Node) lint.Failure {
	return lint.Failure{
		Confidence: 1,
		Node:       node,
		Category:   "logic",
		Failure:    "unreachable code after this statement",
	}
}
