package rule

import (
	"go/ast"
	"go/token"

	"github.com/mgechev/revive/lint"
)

// EmptyLinesRule lints empty lines in blocks.
type EmptyLinesRule struct{}

// Apply applies the rule to given file.
func (r *EmptyLinesRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintEmptyLines{file, file.CommentMap(), onFailure}
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (r *EmptyLinesRule) Name() string {
	return "empty-lines"
}

type lintEmptyLines struct {
	file      *lint.File
	cmap      ast.CommentMap
	onFailure func(lint.Failure)
}

func (w lintEmptyLines) Visit(node ast.Node) ast.Visitor {
	block, ok := node.(*ast.BlockStmt)
	if !ok {
		return w
	}

	w.checkStart(block)
	w.checkEnd(block)

	return w
}

func (w lintEmptyLines) checkStart(block *ast.BlockStmt) {
	if len(block.List) == 0 {
		return
	}

	start := w.position(block.Lbrace)
	firstNode := block.List[0]

	if w.commentBetween(start, firstNode) {
		return
	}

	first := w.position(firstNode.Pos())
	if first.Line-start.Line > 1 {
		w.onFailure(lint.Failure{
			Confidence: 1,
			Node:       block,
			Category:   "style",
			Failure:    "extra empty line at the start of a block",
		})
	}
}

func (w lintEmptyLines) checkEnd(block *ast.BlockStmt) {
	if len(block.List) < 1 {
		return
	}

	end := w.position(block.Rbrace)
	lastNode := block.List[len(block.List)-1]

	if w.commentBetween(end, lastNode) {
		return
	}

	last := w.position(lastNode.End())
	if end.Line-last.Line > 1 {
		w.onFailure(lint.Failure{
			Confidence: 1,
			Node:       lastNode,
			Category:   "style",
			Failure:    "extra empty line at the end of a block",
		})
	}
}

func (w lintEmptyLines) commentBetween(position token.Position, node ast.Node) bool {
	comments := w.cmap.Filter(node).Comments()
	if len(comments) == 0 {
		return false
	}

	for _, comment := range comments {
		start, end := w.position(comment.Pos()), w.position(comment.End())
		if start.Line-position.Line == 1 || position.Line-end.Line == 1 {
			return true
		}
	}

	return false
}

func (w lintEmptyLines) position(pos token.Pos) token.Position {
	return w.file.ToPosition(pos)
}
