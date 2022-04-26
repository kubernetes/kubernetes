package whitespace

import (
	"go/ast"
	"go/token"
)

// Message contains a message
type Message struct {
	Pos     token.Position
	Type    MessageType
	Message string
}

// MessageType describes what should happen to fix the warning
type MessageType uint8

// List of MessageTypes
const (
	MessageTypeLeading MessageType = iota + 1
	MessageTypeTrailing
	MessageTypeAddAfter
)

// Settings contains settings for edge-cases
type Settings struct {
	MultiIf   bool
	MultiFunc bool
}

// Run runs this linter on the provided code
func Run(file *ast.File, fset *token.FileSet, settings Settings) []Message {
	var messages []Message

	for _, f := range file.Decls {
		decl, ok := f.(*ast.FuncDecl)
		if !ok || decl.Body == nil { // decl.Body can be nil for e.g. cgo
			continue
		}

		vis := visitor{file.Comments, fset, nil, make(map[*ast.BlockStmt]bool), settings}
		ast.Walk(&vis, decl)

		messages = append(messages, vis.messages...)
	}

	return messages
}

type visitor struct {
	comments    []*ast.CommentGroup
	fset        *token.FileSet
	messages    []Message
	wantNewline map[*ast.BlockStmt]bool
	settings    Settings
}

func (v *visitor) Visit(node ast.Node) ast.Visitor {
	if node == nil {
		return v
	}

	if stmt, ok := node.(*ast.IfStmt); ok && v.settings.MultiIf {
		checkMultiLine(v, stmt.Body, stmt.Cond)
	}

	if stmt, ok := node.(*ast.FuncLit); ok && v.settings.MultiFunc {
		checkMultiLine(v, stmt.Body, stmt.Type)
	}

	if stmt, ok := node.(*ast.FuncDecl); ok && v.settings.MultiFunc {
		checkMultiLine(v, stmt.Body, stmt.Type)
	}

	if stmt, ok := node.(*ast.BlockStmt); ok {
		wantNewline := v.wantNewline[stmt]

		comments := v.comments
		if wantNewline {
			comments = nil // Comments also count as a newline if we want a newline
		}
		first, last := firstAndLast(comments, v.fset, stmt.Pos(), stmt.End(), stmt.List)

		startMsg := checkStart(v.fset, stmt.Lbrace, first)

		if wantNewline && startMsg == nil {
			v.messages = append(v.messages, Message{v.fset.Position(stmt.Pos()), MessageTypeAddAfter, `multi-line statement should be followed by a newline`})
		} else if !wantNewline && startMsg != nil {
			v.messages = append(v.messages, *startMsg)
		}

		if msg := checkEnd(v.fset, stmt.Rbrace, last); msg != nil {
			v.messages = append(v.messages, *msg)
		}
	}

	return v
}

func checkMultiLine(v *visitor, body *ast.BlockStmt, stmtStart ast.Node) {
	start, end := posLine(v.fset, stmtStart.Pos()), posLine(v.fset, stmtStart.End())

	if end > start { // Check only multi line conditions
		v.wantNewline[body] = true
	}
}

func posLine(fset *token.FileSet, pos token.Pos) int {
	return fset.Position(pos).Line
}

func firstAndLast(comments []*ast.CommentGroup, fset *token.FileSet, start, end token.Pos, stmts []ast.Stmt) (ast.Node, ast.Node) {
	if len(stmts) == 0 {
		return nil, nil
	}

	first, last := ast.Node(stmts[0]), ast.Node(stmts[len(stmts)-1])

	for _, c := range comments {
		if posLine(fset, c.Pos()) == posLine(fset, start) || posLine(fset, c.End()) == posLine(fset, end) {
			continue
		}

		if c.Pos() < start || c.End() > end {
			continue
		}
		if c.Pos() < first.Pos() {
			first = c
		}
		if c.End() > last.End() {
			last = c
		}
	}

	return first, last
}

func checkStart(fset *token.FileSet, start token.Pos, first ast.Node) *Message {
	if first == nil {
		return nil
	}

	if posLine(fset, start)+1 < posLine(fset, first.Pos()) {
		pos := fset.Position(start)
		return &Message{pos, MessageTypeLeading, `unnecessary leading newline`}
	}

	return nil
}

func checkEnd(fset *token.FileSet, end token.Pos, last ast.Node) *Message {
	if last == nil {
		return nil
	}

	if posLine(fset, end)-1 > posLine(fset, last.End()) {
		pos := fset.Position(end)
		return &Message{pos, MessageTypeTrailing, `unnecessary trailing newline`}
	}

	return nil
}
