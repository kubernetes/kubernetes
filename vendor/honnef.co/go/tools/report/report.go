package report

import (
	"bytes"
	"go/ast"
	"go/printer"
	"go/token"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/astutil"
	"honnef.co/go/tools/facts"
	"honnef.co/go/tools/lint"
)

type Options struct {
	ShortRange      bool
	FilterGenerated bool
	Fixes           []analysis.SuggestedFix
	Related         []analysis.RelatedInformation
}

type Option func(*Options)

func ShortRange() Option {
	return func(opts *Options) {
		opts.ShortRange = true
	}
}

func FilterGenerated() Option {
	return func(opts *Options) {
		opts.FilterGenerated = true
	}
}

func Fixes(fixes ...analysis.SuggestedFix) Option {
	return func(opts *Options) {
		opts.Fixes = append(opts.Fixes, fixes...)
	}
}

func Related(node Positioner, message string) Option {
	return func(opts *Options) {
		pos, end := getRange(node, opts.ShortRange)
		r := analysis.RelatedInformation{
			Pos:     pos,
			End:     end,
			Message: message,
		}
		opts.Related = append(opts.Related, r)
	}
}

type Positioner interface {
	Pos() token.Pos
}

type fullPositioner interface {
	Pos() token.Pos
	End() token.Pos
}

type sourcer interface {
	Source() ast.Node
}

// shortRange returns the position and end of the main component of an
// AST node. For nodes that have no body, the short range is identical
// to the node's Pos and End. For nodes that do have a body, the short
// range excludes the body.
func shortRange(node ast.Node) (pos, end token.Pos) {
	switch node := node.(type) {
	case *ast.File:
		return node.Pos(), node.Name.End()
	case *ast.CaseClause:
		return node.Pos(), node.Colon + 1
	case *ast.CommClause:
		return node.Pos(), node.Colon + 1
	case *ast.DeferStmt:
		return node.Pos(), node.Defer + token.Pos(len("defer"))
	case *ast.ExprStmt:
		return shortRange(node.X)
	case *ast.ForStmt:
		if node.Post != nil {
			return node.For, node.Post.End()
		} else if node.Cond != nil {
			return node.For, node.Cond.End()
		} else if node.Init != nil {
			// +1 to catch the semicolon, for gofmt'ed code
			return node.Pos(), node.Init.End() + 1
		} else {
			return node.Pos(), node.For + token.Pos(len("for"))
		}
	case *ast.FuncDecl:
		return node.Pos(), node.Type.End()
	case *ast.FuncLit:
		return node.Pos(), node.Type.End()
	case *ast.GoStmt:
		if _, ok := astutil.Unparen(node.Call.Fun).(*ast.FuncLit); ok {
			return node.Pos(), node.Go + token.Pos(len("go"))
		} else {
			return node.Pos(), node.End()
		}
	case *ast.IfStmt:
		return node.Pos(), node.Cond.End()
	case *ast.RangeStmt:
		return node.Pos(), node.X.End()
	case *ast.SelectStmt:
		return node.Pos(), node.Pos() + token.Pos(len("select"))
	case *ast.SwitchStmt:
		if node.Tag != nil {
			return node.Pos(), node.Tag.End()
		} else if node.Init != nil {
			// +1 to catch the semicolon, for gofmt'ed code
			return node.Pos(), node.Init.End() + 1
		} else {
			return node.Pos(), node.Pos() + token.Pos(len("switch"))
		}
	case *ast.TypeSwitchStmt:
		return node.Pos(), node.Assign.End()
	default:
		return node.Pos(), node.End()
	}
}

func getRange(node Positioner, short bool) (pos, end token.Pos) {
	switch node := node.(type) {
	case sourcer:
		s := node.Source()
		if short {
			return shortRange(s)
		}
		return s.Pos(), s.End()
	case fullPositioner:
		if short {
			return shortRange(node)
		}
		return node.Pos(), node.End()
	default:
		return node.Pos(), token.NoPos
	}
}

func Report(pass *analysis.Pass, node Positioner, message string, opts ...Option) {
	cfg := &Options{}
	for _, opt := range opts {
		opt(cfg)
	}

	file := lint.DisplayPosition(pass.Fset, node.Pos()).Filename
	if cfg.FilterGenerated {
		m := pass.ResultOf[facts.Generated].(map[string]facts.Generator)
		if _, ok := m[file]; ok {
			return
		}
	}

	pos, end := getRange(node, cfg.ShortRange)
	d := analysis.Diagnostic{
		Pos:            pos,
		End:            end,
		Message:        message,
		SuggestedFixes: cfg.Fixes,
		Related:        cfg.Related,
	}
	pass.Report(d)
}

func Render(pass *analysis.Pass, x interface{}) string {
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, pass.Fset, x); err != nil {
		panic(err)
	}
	return buf.String()
}

func RenderArgs(pass *analysis.Pass, args []ast.Expr) string {
	var ss []string
	for _, arg := range args {
		ss = append(ss, Render(pass, arg))
	}
	return strings.Join(ss, ", ")
}
