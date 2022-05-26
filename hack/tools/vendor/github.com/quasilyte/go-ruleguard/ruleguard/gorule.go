package ruleguard

import (
	"fmt"
	"go/ast"
	"go/types"
	"regexp"

	"github.com/quasilyte/go-ruleguard/ruleguard/quasigo"
	"github.com/quasilyte/gogrep"
	"github.com/quasilyte/gogrep/nodetag"
)

type goRuleSet struct {
	universal *scopedGoRuleSet

	groups map[string]*GoRuleGroup // To handle redefinitions
}

type scopedGoRuleSet struct {
	categorizedNum int
	rulesByTag     [nodetag.NumBuckets][]goRule
	commentRules   []goCommentRule
}

type goCommentRule struct {
	base          goRule
	pat           *regexp.Regexp
	captureGroups bool
}

type goRule struct {
	group      *GoRuleGroup
	line       int
	pat        *gogrep.Pattern
	msg        string
	location   string
	suggestion string
	filter     matchFilter
}

type matchFilterResult string

func (s matchFilterResult) Matched() bool { return s == "" }

func (s matchFilterResult) RejectReason() string { return string(s) }

type filterFunc func(*filterParams) matchFilterResult

type matchFilter struct {
	src string
	fn  func(*filterParams) matchFilterResult
}

type filterParams struct {
	ctx      *RunContext
	filename string
	imports  map[string]struct{}
	env      *quasigo.EvalEnv

	importer *goImporter

	match    matchData
	nodePath *nodePath

	nodeText func(n ast.Node) []byte

	deadcode bool

	currentFunc *ast.FuncDecl

	// varname is set only for custom filters before bytecode function is called.
	varname string
}

func (params *filterParams) subNode(name string) ast.Node {
	n, _ := params.match.CapturedByName(name)
	return n
}

func (params *filterParams) subExpr(name string) ast.Expr {
	n, _ := params.match.CapturedByName(name)
	switch n := n.(type) {
	case ast.Expr:
		return n
	case *ast.ExprStmt:
		return n.X
	default:
		return nil
	}
}

func (params *filterParams) typeofNode(n ast.Node) types.Type {
	var e ast.Expr
	switch n := n.(type) {
	case ast.Expr:
		e = n
	case *ast.Field:
		e = n.Type
	}
	if typ := params.ctx.Types.TypeOf(e); typ != nil {
		return typ
	}
	return types.Typ[types.Invalid]
}

func mergeRuleSets(toMerge []*goRuleSet) (*goRuleSet, error) {
	out := &goRuleSet{
		universal: &scopedGoRuleSet{},
		groups:    make(map[string]*GoRuleGroup),
	}

	for _, x := range toMerge {
		out.universal = appendScopedRuleSet(out.universal, x.universal)
		for groupName, group := range x.groups {
			if prevGroup, ok := out.groups[groupName]; ok {
				newRef := fmt.Sprintf("%s:%d", group.Filename, group.Line)
				oldRef := fmt.Sprintf("%s:%d", prevGroup.Filename, prevGroup.Line)
				return nil, fmt.Errorf("%s: redefinition of %s(), previously defined at %s", newRef, groupName, oldRef)
			}
			out.groups[groupName] = group
		}
	}

	return out, nil
}

func appendScopedRuleSet(dst, src *scopedGoRuleSet) *scopedGoRuleSet {
	for tag, rules := range src.rulesByTag {
		dst.rulesByTag[tag] = append(dst.rulesByTag[tag], cloneRuleSlice(rules)...)
		dst.categorizedNum += len(rules)
	}
	dst.commentRules = append(dst.commentRules, src.commentRules...)
	return dst
}

func cloneRuleSlice(slice []goRule) []goRule {
	out := make([]goRule, len(slice))
	for i, rule := range slice {
		clone := rule
		clone.pat = rule.pat.Clone()
		out[i] = clone
	}
	return out
}
