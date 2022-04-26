package ruleguard

import (
	"go/ast"

	"github.com/quasilyte/gogrep"
)

// matchData is used to handle both regexp and AST match sets in the same way.
type matchData interface {
	// TODO: don't use gogrep.CapturedNode type here.

	Node() ast.Node
	CaptureList() []gogrep.CapturedNode
	CapturedByName(name string) (ast.Node, bool)
}

type commentMatchData struct {
	node    ast.Node
	capture []gogrep.CapturedNode
}

func (m commentMatchData) Node() ast.Node { return m.node }

func (m commentMatchData) CaptureList() []gogrep.CapturedNode { return m.capture }

func (m commentMatchData) CapturedByName(name string) (ast.Node, bool) {
	for _, c := range m.capture {
		if c.Name == name {
			return c.Node, true
		}
	}
	return nil, false
}

type astMatchData struct {
	match gogrep.MatchData
}

func (m astMatchData) Node() ast.Node { return m.match.Node }

func (m astMatchData) CaptureList() []gogrep.CapturedNode { return m.match.Capture }

func (m astMatchData) CapturedByName(name string) (ast.Node, bool) {
	return m.match.CapturedByName(name)
}
