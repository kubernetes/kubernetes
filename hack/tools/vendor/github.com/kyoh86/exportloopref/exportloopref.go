package exportloopref

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

var Analyzer = &analysis.Analyzer{
	Name:             "exportloopref",
	Doc:              "checks for pointers to enclosing loop variables",
	Run:              run,
	RunDespiteErrors: true,
	Requires:         []*analysis.Analyzer{inspect.Analyzer},
	// ResultType reflect.Type
	// FactTypes []Fact
}

func init() {
	//	Analyzer.Flags.StringVar(&v, "name", "default", "description")
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	search := &Searcher{
		Stats: map[token.Pos]struct{}{},
		Vars:  map[token.Pos]map[token.Pos]struct{}{},
		Types: pass.TypesInfo.Types,
	}

	nodeFilter := []ast.Node{
		(*ast.RangeStmt)(nil),
		(*ast.ForStmt)(nil),
		(*ast.DeclStmt)(nil),
		(*ast.AssignStmt)(nil),
		(*ast.UnaryExpr)(nil),
	}

	inspect.WithStack(nodeFilter, func(n ast.Node, push bool, stack []ast.Node) bool {
		id, insert, digg := search.Check(n, stack)
		if id != nil {
			dMsg := fmt.Sprintf("exporting a pointer for the loop variable %s", id.Name)
			fMsg := fmt.Sprintf("loop variable %s should be pinned", id.Name)
			var suggest []analysis.SuggestedFix
			if insert != token.NoPos {
				suggest = []analysis.SuggestedFix{{
					Message: fMsg,
					TextEdits: []analysis.TextEdit{{
						Pos:     insert,
						End:     insert,
						NewText: []byte(fmt.Sprintf("%[1]s := %[1]s\n", id.Name)),
					}},
				}}
			}
			d := analysis.Diagnostic{Pos: id.Pos(),
				End:            id.End(),
				Message:        dMsg,
				Category:       "exportloopref",
				SuggestedFixes: suggest,
			}
			pass.Report(d)
		}
		return digg
	})

	return nil, nil
}

type Searcher struct {
	// Statement variables : map to collect positions that
	// variables are declared like below.
	//  - for <KEY>, <VALUE> := range ...
	//  - var <X> int
	//  - D := ...
	Stats map[token.Pos]struct{}
	// Local variables maps loop-position, decl-location to ignore
	// safe pointers for variable which declared in the loop.
	Vars  map[token.Pos]map[token.Pos]struct{}
	Types map[ast.Expr]types.TypeAndValue
}

func (s *Searcher) Check(n ast.Node, stack []ast.Node) (*ast.Ident, token.Pos, bool) {
	switch typed := n.(type) {
	case *ast.RangeStmt:
		s.parseRangeStmt(typed)
	case *ast.ForStmt:
		s.parseForStmt(typed)
	case *ast.DeclStmt:
		s.parseDeclStmt(typed, stack)
	case *ast.AssignStmt:
		s.parseAssignStmt(typed, stack)

	case *ast.UnaryExpr:
		return s.checkUnaryExpr(typed, stack)
	}
	return nil, token.NoPos, true
}

func (s *Searcher) parseRangeStmt(n *ast.RangeStmt) {
	s.addStat(n.Key)
	s.addStat(n.Value)
}

func (s *Searcher) parseForStmt(n *ast.ForStmt) {
	switch post := n.Post.(type) {
	case *ast.AssignStmt:
		// e.g. for p = head; p != nil; p = p.next
		for _, lhs := range post.Lhs {
			s.addStat(lhs)
		}
	case *ast.IncDecStmt:
		// e.g. for i := 0; i < n; i++
		s.addStat(post.X)
	}
}

func (s *Searcher) addStat(expr ast.Expr) {
	if id, ok := expr.(*ast.Ident); ok {
		s.Stats[id.Pos()] = struct{}{}
	}
}

func (s *Searcher) parseDeclStmt(n *ast.DeclStmt, stack []ast.Node) {
	loop, _ := s.innermostLoop(stack)
	if loop == nil {
		return
	}

	// Register declaring variables
	if genDecl, ok := n.Decl.(*ast.GenDecl); ok && genDecl.Tok == token.VAR {
		for _, spec := range genDecl.Specs {
			for _, name := range spec.(*ast.ValueSpec).Names {
				s.addVar(loop, name)
			}
		}
	}
}

func (s *Searcher) parseAssignStmt(n *ast.AssignStmt, stack []ast.Node) {
	loop, _ := s.innermostLoop(stack)
	if loop == nil {
		return
	}

	// Find statements declaring local variable
	if n.Tok == token.DEFINE {
		for _, h := range n.Lhs {
			s.addVar(loop, h)
		}
	}
}

func (s *Searcher) addVar(loop ast.Node, expr ast.Expr) {
	loopPos := loop.Pos()
	id, ok := expr.(*ast.Ident)
	if !ok {
		return
	}
	vars, ok := s.Vars[loopPos]
	if !ok {
		vars = map[token.Pos]struct{}{}
	}
	vars[id.Obj.Pos()] = struct{}{}
	s.Vars[loopPos] = vars
}

func insertionPosition(block *ast.BlockStmt) token.Pos {
	if len(block.List) > 0 {
		return block.List[0].Pos()
	}
	return token.NoPos
}

func (s *Searcher) innermostLoop(stack []ast.Node) (ast.Node, token.Pos) {
	for i := len(stack) - 1; i >= 0; i-- {
		switch typed := stack[i].(type) {
		case *ast.RangeStmt:
			return typed, insertionPosition(typed.Body)
		case *ast.ForStmt:
			return typed, insertionPosition(typed.Body)
		}
	}
	return nil, token.NoPos
}

func (s *Searcher) checkUnaryExpr(n *ast.UnaryExpr, stack []ast.Node) (*ast.Ident, token.Pos, bool) {
	loop, insert := s.innermostLoop(stack)
	if loop == nil {
		return nil, token.NoPos, true
	}

	if n.Op != token.AND {
		return nil, token.NoPos, true
	}

	// Get identity of the referred item
	id := s.getIdentity(n.X)
	if id == nil {
		return nil, token.NoPos, true
	}

	// If the identity is not the loop statement variable,
	// it will not be reported.
	if _, isStat := s.Stats[id.Obj.Pos()]; !isStat {
		return nil, token.NoPos, true
	}

	// check stack append(), []X{}, map[Type]X{}, Struct{}, &Struct{}, X.(Type), (X)
	// in the <outer> =
	var mayRHPos token.Pos
	for i := len(stack) - 2; i >= 0; i-- {
		switch typed := stack[i].(type) {
		case (*ast.UnaryExpr):
			// noop
		case (*ast.CompositeLit):
			// noop
		case (*ast.KeyValueExpr):
			// noop
		case (*ast.CallExpr):
			fun, ok := typed.Fun.(*ast.Ident)
			if !ok {
				return nil, token.NoPos, false // it's calling a function other of `append`. It cannot be checked
			}

			if fun.Name != "append" {
				return nil, token.NoPos, false // it's calling a function other of `append`. It cannot be checked
			}

		case (*ast.AssignStmt):
			if len(typed.Rhs) != len(typed.Lhs) {
				return nil, token.NoPos, false // dead logic
			}

			// search x where Rhs[x].Pos() == mayRHPos
			var index int
			for ri, rh := range typed.Rhs {
				if rh.Pos() == mayRHPos {
					index = ri
					break
				}
			}

			// check Lhs[x] is not local variable
			lh := typed.Lhs[index]
			isVar := s.isVar(loop, lh)
			if !isVar {
				return id, insert, false
			}

			return nil, token.NoPos, true
		default:
			// Other statement is not able to be checked.
			return nil, token.NoPos, false
		}

		// memory an expr that may be right-hand in the AssignStmt
		mayRHPos = stack[i].Pos()
	}
	return nil, token.NoPos, true
}

func (s *Searcher) isVar(loop ast.Node, expr ast.Expr) bool {
	vars := s.Vars[loop.Pos()] // map[token.Pos]struct{}
	if vars == nil {
		return false
	}
	switch typed := expr.(type) {
	case (*ast.Ident):
		_, isVar := vars[typed.Obj.Pos()]
		return isVar
	case (*ast.IndexExpr): // like X[Y], check X
		return s.isVar(loop, typed.X)
	case (*ast.SelectorExpr): // like X.Y, check X
		return s.isVar(loop, typed.X)
	}
	return false
}

// Get variable identity
func (s *Searcher) getIdentity(expr ast.Expr) *ast.Ident {
	switch typed := expr.(type) {
	case *ast.SelectorExpr:
		// Ignore if the parent is pointer ref (fix for #2)
		if _, ok := s.Types[typed.X].Type.(*types.Pointer); ok {
			return nil
		}

		// Get parent identity; i.e. `a.b` of the `a.b.c`.
		return s.getIdentity(typed.X)

	case *ast.Ident:
		// Get simple identity; i.e. `a` of the `a`.
		if typed.Obj == nil {
			return nil
		}
		return typed
	}
	return nil
}
