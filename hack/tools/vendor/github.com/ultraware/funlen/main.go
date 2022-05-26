package funlen

import (
	"fmt"
	"go/ast"
	"go/token"
	"reflect"
)

const (
	defaultLineLimit = 60
	defaultStmtLimit = 40
)

// Run runs this linter on the provided code
func Run(file *ast.File, fset *token.FileSet, lineLimit, stmtLimit int) []Message {
	if lineLimit == 0 {
		lineLimit = defaultLineLimit
	}
	if stmtLimit == 0 {
		stmtLimit = defaultStmtLimit
	}

	var msgs []Message
	for _, f := range file.Decls {
		decl, ok := f.(*ast.FuncDecl)
		if !ok || decl.Body == nil { // decl.Body can be nil for e.g. cgo
			continue
		}

		if stmtLimit > 0 {
			if stmts := parseStmts(decl.Body.List); stmts > stmtLimit {
				msgs = append(msgs, makeStmtMessage(fset, decl.Name, stmts, stmtLimit))
				continue
			}
		}

		if lineLimit > 0 {
			if lines := getLines(fset, decl); lines > lineLimit {
				msgs = append(msgs, makeLineMessage(fset, decl.Name, lines, lineLimit))
			}
		}
	}

	return msgs
}

// Message contains a message
type Message struct {
	Pos     token.Position
	Message string
}

func makeLineMessage(fset *token.FileSet, funcInfo *ast.Ident, lines, lineLimit int) Message {
	return Message{
		fset.Position(funcInfo.Pos()),
		fmt.Sprintf("Function '%s' is too long (%d > %d)\n", funcInfo.Name, lines, lineLimit),
	}
}

func makeStmtMessage(fset *token.FileSet, funcInfo *ast.Ident, stmts, stmtLimit int) Message {
	return Message{
		fset.Position(funcInfo.Pos()),
		fmt.Sprintf("Function '%s' has too many statements (%d > %d)\n", funcInfo.Name, stmts, stmtLimit),
	}
}

func getLines(fset *token.FileSet, f *ast.FuncDecl) int { // nolint: interfacer
	return fset.Position(f.End()).Line - fset.Position(f.Pos()).Line - 1
}

func parseStmts(s []ast.Stmt) (total int) {
	for _, v := range s {
		total++
		switch stmt := v.(type) {
		case *ast.BlockStmt:
			total += parseStmts(stmt.List) - 1
		case *ast.ForStmt, *ast.RangeStmt, *ast.IfStmt,
			*ast.SwitchStmt, *ast.TypeSwitchStmt, *ast.SelectStmt:
			total += parseBodyListStmts(stmt)
		case *ast.CaseClause:
			total += parseStmts(stmt.Body)
		case *ast.AssignStmt:
			total += checkInlineFunc(stmt.Rhs[0])
		case *ast.GoStmt:
			total += checkInlineFunc(stmt.Call.Fun)
		case *ast.DeferStmt:
			total += checkInlineFunc(stmt.Call.Fun)
		}
	}
	return
}

func checkInlineFunc(stmt ast.Expr) int {
	if block, ok := stmt.(*ast.FuncLit); ok {
		return parseStmts(block.Body.List)
	}
	return 0
}

func parseBodyListStmts(t interface{}) int {
	i := reflect.ValueOf(t).Elem().FieldByName(`Body`).Elem().FieldByName(`List`).Interface()
	return parseStmts(i.([]ast.Stmt))
}
