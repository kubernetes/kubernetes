package golinters

import (
	"fmt"
	"go/ast"
	"go/token"
	"sync"

	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const scopelintName = "scopelint"

func NewScopelint() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: scopelintName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		scopelintName,
		"Scopelint checks for unpinned variables in go programs",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var res []result.Issue
			for _, file := range pass.Files {
				n := Node{
					fset:          pass.Fset,
					DangerObjects: map[*ast.Object]int{},
					UnsafeObjects: map[*ast.Object]int{},
					SkipFuncs:     map[*ast.FuncLit]int{},
					issues:        &res,
				}
				ast.Walk(&n, file)
			}

			if len(res) == 0 {
				return nil, nil
			}

			mu.Lock()
			for i := range res {
				resIssues = append(resIssues, goanalysis.NewIssue(&res[i], pass))
			}
			mu.Unlock()

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return resIssues
	}).WithLoadMode(goanalysis.LoadModeSyntax)
}

// The code below is copy-pasted from https://github.com/kyoh86/scopelint 92cbe2cc9276abda0e309f52cc9e309d407f174e

// Node represents a Node being linted.
type Node struct {
	fset          *token.FileSet
	DangerObjects map[*ast.Object]int
	UnsafeObjects map[*ast.Object]int
	SkipFuncs     map[*ast.FuncLit]int
	issues        *[]result.Issue
}

// Visit method is invoked for each node encountered by Walk.
// If the result visitor w is not nil, Walk visits each of the children
// of node with the visitor w, followed by a call of w.Visit(nil).
//nolint:gocyclo,gocritic
func (f *Node) Visit(node ast.Node) ast.Visitor {
	switch typedNode := node.(type) {
	case *ast.ForStmt:
		switch init := typedNode.Init.(type) {
		case *ast.AssignStmt:
			for _, lh := range init.Lhs {
				switch tlh := lh.(type) {
				case *ast.Ident:
					f.UnsafeObjects[tlh.Obj] = 0
				}
			}
		}

	case *ast.RangeStmt:
		// Memory variables declared in range statement
		switch k := typedNode.Key.(type) {
		case *ast.Ident:
			f.UnsafeObjects[k.Obj] = 0
		}
		switch v := typedNode.Value.(type) {
		case *ast.Ident:
			f.UnsafeObjects[v.Obj] = 0
		}

	case *ast.UnaryExpr:
		if typedNode.Op == token.AND {
			switch ident := typedNode.X.(type) {
			case *ast.Ident:
				if _, unsafe := f.UnsafeObjects[ident.Obj]; unsafe {
					f.errorf(ident, "Using a reference for the variable on range scope %s", formatCode(ident.Name, nil))
				}
			}
		}

	case *ast.Ident:
		if _, obj := f.DangerObjects[typedNode.Obj]; obj {
			// It is the naked variable in scope of range statement.
			f.errorf(node, "Using the variable on range scope %s in function literal", formatCode(typedNode.Name, nil))
			break
		}

	case *ast.CallExpr:
		// Ignore func literals that'll be called immediately.
		switch funcLit := typedNode.Fun.(type) {
		case *ast.FuncLit:
			f.SkipFuncs[funcLit] = 0
		}

	case *ast.FuncLit:
		if _, skip := f.SkipFuncs[typedNode]; !skip {
			dangers := map[*ast.Object]int{}
			for d := range f.DangerObjects {
				dangers[d] = 0
			}
			for u := range f.UnsafeObjects {
				dangers[u] = 0
				f.UnsafeObjects[u]++
			}
			return &Node{
				fset:          f.fset,
				DangerObjects: dangers,
				UnsafeObjects: f.UnsafeObjects,
				SkipFuncs:     f.SkipFuncs,
				issues:        f.issues,
			}
		}

	case *ast.ReturnStmt:
		unsafe := map[*ast.Object]int{}
		for u := range f.UnsafeObjects {
			if f.UnsafeObjects[u] == 0 {
				continue
			}
			unsafe[u] = f.UnsafeObjects[u]
		}
		return &Node{
			fset:          f.fset,
			DangerObjects: f.DangerObjects,
			UnsafeObjects: unsafe,
			SkipFuncs:     f.SkipFuncs,
			issues:        f.issues,
		}
	}
	return f
}

// The variadic arguments may start with link and category types,
// and must end with a format string and any arguments.
//nolint:interfacer
func (f *Node) errorf(n ast.Node, format string, args ...interface{}) {
	pos := f.fset.Position(n.Pos())
	f.errorAtf(pos, format, args...)
}

func (f *Node) errorAtf(pos token.Position, format string, args ...interface{}) {
	*f.issues = append(*f.issues, result.Issue{
		Pos:        pos,
		Text:       fmt.Sprintf(format, args...),
		FromLinter: scopelintName,
	})
}
