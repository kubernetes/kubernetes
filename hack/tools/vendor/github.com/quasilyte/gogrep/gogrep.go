package gogrep

import (
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"github.com/quasilyte/gogrep/nodetag"
)

func IsEmptyNodeSlice(n ast.Node) bool {
	if list, ok := n.(NodeSlice); ok {
		return list.Len() == 0
	}
	return false
}

// MatchData describes a successful pattern match.
type MatchData struct {
	Node    ast.Node
	Capture []CapturedNode
}

type CapturedNode struct {
	Name string
	Node ast.Node
}

func (data MatchData) CapturedByName(name string) (ast.Node, bool) {
	if name == "$$" {
		return data.Node, true
	}
	return findNamed(data.Capture, name)
}

type MatcherState struct {
	Types *types.Info

	// node values recorded by name, excluding "_" (used only by the
	// actual matching phase)
	capture []CapturedNode

	pc int
}

func NewMatcherState() MatcherState {
	return MatcherState{
		capture: make([]CapturedNode, 0, 8),
	}
}

type Pattern struct {
	m *matcher
}

type PatternInfo struct {
	Vars map[string]struct{}
}

func (p *Pattern) NodeTag() nodetag.Value {
	return operationInfoTable[p.m.prog.insts[0].op].Tag
}

// MatchNode calls cb if n matches a pattern.
func (p *Pattern) MatchNode(state *MatcherState, n ast.Node, cb func(MatchData)) {
	p.m.MatchNode(state, n, cb)
}

// Clone creates a pattern copy.
func (p *Pattern) Clone() *Pattern {
	clone := *p
	clone.m = &matcher{}
	*clone.m = *p.m
	return &clone
}

type CompileConfig struct {
	Fset *token.FileSet

	// Src is a gogrep pattern expression string.
	Src string

	// When strict is false, gogrep may consider 0xA and 10 to be identical.
	// If true, a compiled pattern will require a full syntax match.
	Strict bool

	// WithTypes controls whether gogrep would have types.Info during the pattern execution.
	// If set to true, it will compile a pattern to a potentially more precise form, where
	// fmt.Printf maps to the stdlib function call but not Printf method call on some
	// random fmt variable.
	WithTypes bool

	// Imports specifies packages that should be recognized for the type-aware matching.
	// It maps a package name to a package path.
	// Only used if WithTypes is true.
	Imports map[string]string
}

func Compile(config CompileConfig) (*Pattern, PatternInfo, error) {
	if strings.HasPrefix(config.Src, "import $") {
		return compileImportPattern(config)
	}
	info := newPatternInfo()
	n, err := parseExpr(config.Fset, config.Src)
	if err != nil {
		return nil, info, err
	}
	var c compiler
	c.config = config
	prog, err := c.Compile(n, &info)
	if err != nil {
		return nil, info, err
	}
	m := newMatcher(prog)
	return &Pattern{m: m}, info, nil
}

func newPatternInfo() PatternInfo {
	return PatternInfo{
		Vars: make(map[string]struct{}),
	}
}
