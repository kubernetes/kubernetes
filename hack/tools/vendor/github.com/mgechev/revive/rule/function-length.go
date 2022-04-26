package rule

import (
	"fmt"
	"go/ast"
	"reflect"

	"github.com/mgechev/revive/lint"
)

// FunctionLength lint.
type FunctionLength struct {
	maxStmt  int
	maxLines int
}

// Apply applies the rule to given file.
func (r *FunctionLength) Apply(file *lint.File, arguments lint.Arguments) []lint.Failure {
	if r.maxLines == 0 {
		maxStmt, maxLines := r.parseArguments(arguments)
		r.maxStmt = int(maxStmt)
		r.maxLines = int(maxLines)
	}

	var failures []lint.Failure

	walker := lintFuncLength{
		file:     file,
		maxStmt:  r.maxStmt,
		maxLines: r.maxLines,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	ast.Walk(walker, file.AST)

	return failures
}

// Name returns the rule name.
func (r *FunctionLength) Name() string {
	return "function-length"
}

func (r *FunctionLength) parseArguments(arguments lint.Arguments) (maxStmt int64, maxLines int64) {
	if len(arguments) != 2 {
		panic(fmt.Sprintf(`invalid configuration for "function-length" rule, expected 2 arguments but got %d`, len(arguments)))
	}

	maxStmt, maxStmtOk := arguments[0].(int64)
	if !maxStmtOk {
		panic(fmt.Sprintf(`invalid configuration value for max statements in "function-length" rule; need int64 but got %T`, arguments[0]))
	}
	if maxStmt < 0 {
		panic(fmt.Sprintf(`the configuration value for max statements in "function-length" rule cannot be negative, got %d`, maxStmt))
	}

	maxLines, maxLinesOk := arguments[1].(int64)
	if !maxLinesOk {
		panic(fmt.Sprintf(`invalid configuration value for max lines in "function-length" rule; need int64 but got %T`, arguments[1]))
	}
	if maxLines < 0 {
		panic(fmt.Sprintf(`the configuration value for max statements in "function-length" rule cannot be negative, got %d`, maxLines))
	}

	return
}

type lintFuncLength struct {
	file      *lint.File
	maxStmt   int
	maxLines  int
	onFailure func(lint.Failure)
}

func (w lintFuncLength) Visit(n ast.Node) ast.Visitor {
	node, ok := n.(*ast.FuncDecl)
	if !ok {
		return w
	}

	body := node.Body
	if body == nil || len(node.Body.List) == 0 {
		return nil
	}

	if w.maxStmt > 0 {
		stmtCount := w.countStmts(node.Body.List)
		if stmtCount > w.maxStmt {
			w.onFailure(lint.Failure{
				Confidence: 1,
				Failure:    fmt.Sprintf("maximum number of statements per function exceeded; max %d but got %d", w.maxStmt, stmtCount),
				Node:       node,
			})
		}
	}

	if w.maxLines > 0 {
		lineCount := w.countLines(node.Body)
		if lineCount > w.maxLines {
			w.onFailure(lint.Failure{
				Confidence: 1,
				Failure:    fmt.Sprintf("maximum number of lines per function exceeded; max %d but got %d", w.maxLines, lineCount),
				Node:       node,
			})
		}
	}

	return nil
}

func (w lintFuncLength) countLines(b *ast.BlockStmt) int {
	return w.file.ToPosition(b.End()).Line - w.file.ToPosition(b.Pos()).Line - 1
}

func (w lintFuncLength) countStmts(b []ast.Stmt) int {
	count := 0
	for _, s := range b {
		switch stmt := s.(type) {
		case *ast.BlockStmt:
			count += w.countStmts(stmt.List)
		case *ast.IfStmt:
			count += 1 + w.countBodyListStmts(stmt)
			if stmt.Else != nil {
				elseBody, ok := stmt.Else.(*ast.BlockStmt)
				if ok {
					count += w.countStmts(elseBody.List)
				}
			}
		case *ast.ForStmt, *ast.RangeStmt,
			*ast.SwitchStmt, *ast.TypeSwitchStmt, *ast.SelectStmt:
			count += 1 + w.countBodyListStmts(stmt)
		case *ast.CaseClause:
			count += w.countStmts(stmt.Body)
		case *ast.AssignStmt:
			count += 1 + w.countFuncLitStmts(stmt.Rhs[0])
		case *ast.GoStmt:
			count += 1 + w.countFuncLitStmts(stmt.Call.Fun)
		case *ast.DeferStmt:
			count += 1 + w.countFuncLitStmts(stmt.Call.Fun)
		default:
			count++
		}
	}

	return count
}

func (w lintFuncLength) countFuncLitStmts(stmt ast.Expr) int {
	if block, ok := stmt.(*ast.FuncLit); ok {
		return w.countStmts(block.Body.List)
	}
	return 0
}

func (w lintFuncLength) countBodyListStmts(t interface{}) int {
	i := reflect.ValueOf(t).Elem().FieldByName(`Body`).Elem().FieldByName(`List`).Interface()
	return w.countStmts(i.([]ast.Stmt))
}
