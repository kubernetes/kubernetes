package rule

import (
	"fmt"
	"go/ast"
	"strings"

	"github.com/mgechev/revive/lint"
)

// RangeRule lints given else constructs.
type RangeRule struct{}

// Apply applies the rule to given file.
func (r *RangeRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := &lintRanges{file, onFailure}
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (r *RangeRule) Name() string {
	return "range"
}

type lintRanges struct {
	file      *lint.File
	onFailure func(lint.Failure)
}

func (w *lintRanges) Visit(node ast.Node) ast.Visitor {
	rs, ok := node.(*ast.RangeStmt)
	if !ok {
		return w
	}
	if rs.Value == nil {
		// for x = range m { ... }
		return w // single var form
	}
	if !isIdent(rs.Value, "_") {
		// for ?, y = range m { ... }
		return w
	}

	newRS := *rs // shallow copy
	newRS.Value = nil

	w.onFailure(lint.Failure{
		Failure:         fmt.Sprintf("should omit 2nd value from range; this loop is equivalent to `for %s %s range ...`", w.file.Render(rs.Key), rs.Tok),
		Confidence:      1,
		Node:            rs.Value,
		ReplacementLine: firstLineOf(w.file, &newRS, rs),
	})

	return w
}

func firstLineOf(f *lint.File, node, match ast.Node) string {
	line := f.Render(node)
	if i := strings.Index(line, "\n"); i >= 0 {
		line = line[:i]
	}
	return indentOf(f, match) + line
}

func indentOf(f *lint.File, node ast.Node) string {
	line := srcLine(f.Content(), f.ToPosition(node.Pos()))
	for i, r := range line {
		switch r {
		case ' ', '\t':
		default:
			return line[:i]
		}
	}
	return line // unusual or empty line
}
