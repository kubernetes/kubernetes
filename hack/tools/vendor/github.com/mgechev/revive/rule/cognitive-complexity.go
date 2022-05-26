package rule

import (
	"fmt"
	"go/ast"
	"go/token"

	"github.com/mgechev/revive/lint"
	"golang.org/x/tools/go/ast/astutil"
)

// CognitiveComplexityRule lints given else constructs.
type CognitiveComplexityRule struct {
	maxComplexity int
}

// Apply applies the rule to given file.
func (r *CognitiveComplexityRule) Apply(file *lint.File, arguments lint.Arguments) []lint.Failure {
	if r.maxComplexity == 0 {
		checkNumberOfArguments(1, arguments, r.Name())

		complexity, ok := arguments[0].(int64)
		if !ok {
			panic(fmt.Sprintf("invalid argument type for cognitive-complexity, expected int64, got %T", arguments[0]))
		}
		r.maxComplexity = int(complexity)
	}

	var failures []lint.Failure
	linter := cognitiveComplexityLinter{
		file:          file,
		maxComplexity: r.maxComplexity,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	linter.lint()

	return failures
}

// Name returns the rule name.
func (r *CognitiveComplexityRule) Name() string {
	return "cognitive-complexity"
}

type cognitiveComplexityLinter struct {
	file          *lint.File
	maxComplexity int
	onFailure     func(lint.Failure)
}

func (w cognitiveComplexityLinter) lint() {
	f := w.file
	for _, decl := range f.AST.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Body != nil {
			v := cognitiveComplexityVisitor{}
			c := v.subTreeComplexity(fn.Body)
			if c > w.maxComplexity {
				w.onFailure(lint.Failure{
					Confidence: 1,
					Category:   "maintenance",
					Failure:    fmt.Sprintf("function %s has cognitive complexity %d (> max enabled %d)", funcName(fn), c, w.maxComplexity),
					Node:       fn,
				})
			}
		}
	}
}

type cognitiveComplexityVisitor struct {
	complexity   int
	nestingLevel int
}

// subTreeComplexity calculates the cognitive complexity of an AST-subtree.
func (v cognitiveComplexityVisitor) subTreeComplexity(n ast.Node) int {
	ast.Walk(&v, n)
	return v.complexity
}

// Visit implements the ast.Visitor interface.
func (v *cognitiveComplexityVisitor) Visit(n ast.Node) ast.Visitor {
	switch n := n.(type) {
	case *ast.IfStmt:
		targets := []ast.Node{n.Cond, n.Body, n.Else}
		v.walk(1, targets...)
		return nil
	case *ast.ForStmt:
		targets := []ast.Node{n.Cond, n.Body}
		v.walk(1, targets...)
		return nil
	case *ast.RangeStmt:
		v.walk(1, n.Body)
		return nil
	case *ast.SelectStmt:
		v.walk(1, n.Body)
		return nil
	case *ast.SwitchStmt:
		v.walk(1, n.Body)
		return nil
	case *ast.TypeSwitchStmt:
		v.walk(1, n.Body)
		return nil
	case *ast.FuncLit:
		v.walk(0, n.Body) // do not increment the complexity, just do the nesting
		return nil
	case *ast.BinaryExpr:
		v.complexity += v.binExpComplexity(n)
		return nil // skip visiting binexp sub-tree (already visited by binExpComplexity)
	case *ast.BranchStmt:
		if n.Label != nil {
			v.complexity++
		}
	}
	// TODO handle (at least) direct recursion

	return v
}

func (v *cognitiveComplexityVisitor) walk(complexityIncrement int, targets ...ast.Node) {
	v.complexity += complexityIncrement + v.nestingLevel
	nesting := v.nestingLevel
	v.nestingLevel++

	for _, t := range targets {
		if t == nil {
			continue
		}

		ast.Walk(v, t)
	}

	v.nestingLevel = nesting
}

func (cognitiveComplexityVisitor) binExpComplexity(n *ast.BinaryExpr) int {
	calculator := binExprComplexityCalculator{opsStack: []token.Token{}}

	astutil.Apply(n, calculator.pre, calculator.post)

	return calculator.complexity
}

type binExprComplexityCalculator struct {
	complexity    int
	opsStack      []token.Token // stack of bool operators
	subexpStarted bool
}

func (becc *binExprComplexityCalculator) pre(c *astutil.Cursor) bool {
	switch n := c.Node().(type) {
	case *ast.BinaryExpr:
		isBoolOp := n.Op == token.LAND || n.Op == token.LOR
		if !isBoolOp {
			break
		}

		ops := len(becc.opsStack)
		// if
		// 		is the first boolop in the expression OR
		// 		is the first boolop inside a subexpression (...) OR
		//		is not the same to the previous one
		// then
		//      increment complexity
		if ops == 0 || becc.subexpStarted || n.Op != becc.opsStack[ops-1] {
			becc.complexity++
			becc.subexpStarted = false
		}

		becc.opsStack = append(becc.opsStack, n.Op)
	case *ast.ParenExpr:
		becc.subexpStarted = true
	}

	return true
}

func (becc *binExprComplexityCalculator) post(c *astutil.Cursor) bool {
	switch n := c.Node().(type) {
	case *ast.BinaryExpr:
		isBoolOp := n.Op == token.LAND || n.Op == token.LOR
		if !isBoolOp {
			break
		}

		ops := len(becc.opsStack)
		if ops > 0 {
			becc.opsStack = becc.opsStack[:ops-1]
		}
	case *ast.ParenExpr:
		becc.subexpStarted = false
	}

	return true
}
