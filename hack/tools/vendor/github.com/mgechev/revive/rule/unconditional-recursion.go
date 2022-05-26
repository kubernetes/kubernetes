package rule

import (
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// UnconditionalRecursionRule lints given else constructs.
type UnconditionalRecursionRule struct{}

// Apply applies the rule to given file.
func (r *UnconditionalRecursionRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintUnconditionalRecursionRule{onFailure: onFailure}
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (r *UnconditionalRecursionRule) Name() string {
	return "unconditional-recursion"
}

type funcDesc struct {
	reciverID *ast.Ident
	id        *ast.Ident
}

func (fd *funcDesc) equal(other *funcDesc) bool {
	receiversAreEqual := (fd.reciverID == nil && other.reciverID == nil) || fd.reciverID != nil && other.reciverID != nil && fd.reciverID.Name == other.reciverID.Name
	idsAreEqual := (fd.id == nil && other.id == nil) || fd.id.Name == other.id.Name

	return receiversAreEqual && idsAreEqual
}

type funcStatus struct {
	funcDesc            *funcDesc
	seenConditionalExit bool
}

type lintUnconditionalRecursionRule struct {
	onFailure   func(lint.Failure)
	currentFunc *funcStatus
}

// Visit will traverse the file AST.
// The rule is based in the following algorithm: inside each function body we search for calls to the function itself.
// We do not search inside conditional control structures (if, for, switch, ...) because any recursive call inside them is conditioned
// We do search inside conditional control structures are statements that will take the control out of the function (return, exit, panic)
// If we find conditional control exits, it means the function is NOT unconditionally-recursive
// If we find a recursive call before finding any conditional exit, a failure is generated
// In resume: if we found a recursive call control-dependant from the entry point of the function then we raise a failure.
func (w lintUnconditionalRecursionRule) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.FuncDecl:
		var rec *ast.Ident
		switch {
		case n.Recv == nil:
			rec = nil
		case n.Recv.NumFields() < 1 || len(n.Recv.List[0].Names) < 1:
			rec = &ast.Ident{Name: "_"}
		default:
			rec = n.Recv.List[0].Names[0]
		}

		w.currentFunc = &funcStatus{&funcDesc{rec, n.Name}, false}
	case *ast.CallExpr:
		var funcID *ast.Ident
		var selector *ast.Ident
		switch c := n.Fun.(type) {
		case *ast.Ident:
			selector = nil
			funcID = c
		case *ast.SelectorExpr:
			var ok bool
			selector, ok = c.X.(*ast.Ident)
			if !ok { // a.b....Foo()
				return nil
			}
			funcID = c.Sel
		default:
			return w
		}

		if w.currentFunc != nil && // not in a func body
			!w.currentFunc.seenConditionalExit && // there is a conditional exit in the function
			w.currentFunc.funcDesc.equal(&funcDesc{selector, funcID}) {
			w.onFailure(lint.Failure{
				Category:   "logic",
				Confidence: 1,
				Node:       n,
				Failure:    "unconditional recursive call",
			})
		}
	case *ast.IfStmt:
		w.updateFuncStatus(n.Body)
		w.updateFuncStatus(n.Else)
		return nil
	case *ast.SelectStmt:
		w.updateFuncStatus(n.Body)
		return nil
	case *ast.RangeStmt:
		w.updateFuncStatus(n.Body)
		return nil
	case *ast.TypeSwitchStmt:
		w.updateFuncStatus(n.Body)
		return nil
	case *ast.SwitchStmt:
		w.updateFuncStatus(n.Body)
		return nil
	case *ast.GoStmt:
		for _, a := range n.Call.Args {
			ast.Walk(w, a) // check if arguments have a recursive call
		}
		return nil // recursive async call is not an issue
	case *ast.ForStmt:
		if n.Cond != nil {
			return nil
		}
		// unconditional loop
		return w
	}

	return w
}

func (w *lintUnconditionalRecursionRule) updateFuncStatus(node ast.Node) {
	if node == nil || w.currentFunc == nil || w.currentFunc.seenConditionalExit {
		return
	}

	w.currentFunc.seenConditionalExit = w.hasControlExit(node)
}

var exitFunctions = map[string]map[string]bool{
	"os":      {"Exit": true},
	"syscall": {"Exit": true},
	"log": {
		"Fatal":   true,
		"Fatalf":  true,
		"Fatalln": true,
		"Panic":   true,
		"Panicf":  true,
		"Panicln": true,
	},
}

func (w *lintUnconditionalRecursionRule) hasControlExit(node ast.Node) bool {
	// isExit returns true if the given node makes control exit the function
	isExit := func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.ReturnStmt:
			return true
		case *ast.CallExpr:
			if isIdent(n.Fun, "panic") {
				return true
			}
			se, ok := n.Fun.(*ast.SelectorExpr)
			if !ok {
				return false
			}

			id, ok := se.X.(*ast.Ident)
			if !ok {
				return false
			}

			fn := se.Sel.Name
			pkg := id.Name
			if exitFunctions[pkg] != nil && exitFunctions[pkg][fn] { // it's a call to an exit function
				return true
			}
		}

		return false
	}

	return len(pick(node, isExit, nil)) != 0
}
