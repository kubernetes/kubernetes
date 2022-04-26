package ir

import (
	"fmt"
	"strings"
)

type File struct {
	PkgPath string

	RuleGroups []RuleGroup

	CustomDecls []string

	BundleImports []BundleImport
}

type BundleImport struct {
	Line int

	PkgPath string
	Prefix  string
}

type RuleGroup struct {
	Line        int
	Name        string
	MatcherName string

	DocTags    []string
	DocSummary string
	DocBefore  string
	DocAfter   string
	DocNote    string

	Imports []PackageImport

	Rules []Rule
}

type PackageImport struct {
	Path string
	Name string
}

type Rule struct {
	Line int

	SyntaxPatterns  []PatternString
	CommentPatterns []PatternString

	ReportTemplate  string
	SuggestTemplate string

	WhereExpr FilterExpr

	LocationVar string
}

type PatternString struct {
	Line  int
	Value string
}

// stringer -type=FilterOp -trimprefix=Filter

//go:generate go run ./gen_filter_op.go
type FilterOp int

func (op FilterOp) String() string { return filterOpNames[op] }

type FilterExpr struct {
	Line int

	Op    FilterOp
	Src   string
	Value interface{}
	Args  []FilterExpr
}

func (e FilterExpr) IsValid() bool { return e.Op != FilterInvalidOp }

func (e FilterExpr) IsBinaryExpr() bool { return filterOpFlags[e.Op]&flagIsBinaryExpr != 0 }
func (e FilterExpr) IsBasicLit() bool   { return filterOpFlags[e.Op]&flagIsBasicLit != 0 }
func (e FilterExpr) HasVar() bool       { return filterOpFlags[e.Op]&flagHasVar != 0 }

func (e FilterExpr) String() string {
	switch e.Op {
	case FilterStringOp:
		return `"` + e.Value.(string) + `"`
	case FilterIntOp:
		return fmt.Sprint(e.Value.(int64))
	}
	parts := make([]string, 0, len(e.Args)+2)
	parts = append(parts, e.Op.String())
	if e.Value != nil {
		parts = append(parts, fmt.Sprintf("[%#v]", e.Value))
	}
	for _, arg := range e.Args {
		parts = append(parts, arg.String())
	}
	if len(parts) == 1 {
		return parts[0]
	}
	return "(" + strings.Join(parts, " ") + ")"
}

const (
	flagIsBinaryExpr uint64 = 1 << iota
	flagIsBasicLit
	flagHasVar
)
