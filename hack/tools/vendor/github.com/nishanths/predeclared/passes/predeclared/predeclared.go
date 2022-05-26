// Package predeclared provides a static analysis (used by the predeclared command)
// that can detect declarations in Go code that shadow one of Go's predeclared identifiers.
package predeclared

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// Flag names used by the analyzer. They are exported for use by analyzer
// driver programs.
const (
	IgnoreFlag    = "ignore"
	QualifiedFlag = "q"
)

var (
	fIgnore    string
	fQualified bool
)

func init() {
	Analyzer.Flags.StringVar(&fIgnore, IgnoreFlag, "", "comma-separated list of predeclared identifiers to not report on")
	Analyzer.Flags.BoolVar(&fQualified, QualifiedFlag, false, "include method names and field names (i.e., qualified names) in checks")
}

var Analyzer = &analysis.Analyzer{
	Name: "predeclared",
	Doc:  "find code that shadows one of Go's predeclared identifiers",
	Run:  run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	cfg := newConfig(fIgnore, fQualified)
	for _, file := range pass.Files {
		processFile(pass.Report, cfg, pass.Fset, file)
	}
	return nil, nil
}

type config struct {
	qualified     bool
	ignoredIdents map[string]struct{}
}

func newConfig(ignore string, qualified bool) *config {
	cfg := &config{
		qualified:     qualified,
		ignoredIdents: map[string]struct{}{},
	}
	for _, s := range strings.Split(ignore, ",") {
		ident := strings.TrimSpace(s)
		if ident == "" {
			continue
		}
		cfg.ignoredIdents[ident] = struct{}{}
	}
	return cfg
}

type issue struct {
	ident *ast.Ident
	kind  string
	fset  *token.FileSet
}

func (i issue) String() string {
	pos := i.fset.Position(i.ident.Pos())
	return fmt.Sprintf("%s: %s %s has same name as predeclared identifier", pos, i.kind, i.ident.Name)
}

func processFile(report func(analysis.Diagnostic), cfg *config, fset *token.FileSet, file *ast.File) []issue { // nolint: gocyclo
	var issues []issue

	maybeReport := func(x *ast.Ident, kind string) {
		if _, isIgnored := cfg.ignoredIdents[x.Name]; !isIgnored && isPredeclaredIdent(x.Name) {
			report(analysis.Diagnostic{
				Pos:     x.Pos(),
				End:     x.End(),
				Message: fmt.Sprintf("%s %s has same name as predeclared identifier", kind, x.Name),
			})
			issues = append(issues, issue{x, kind, fset})
		}
	}

	seenValueSpecs := make(map[*ast.ValueSpec]bool)

	// TODO: consider deduping package name issues for files in the
	// same directory.
	maybeReport(file.Name, "package name")

	for _, spec := range file.Imports {
		if spec.Name == nil {
			continue
		}
		maybeReport(spec.Name, "import name")
	}

	// Handle declarations and fields.
	// https://golang.org/ref/spec#Declarations_and_scope
	ast.Inspect(file, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.GenDecl:
			var kind string
			switch x.Tok {
			case token.CONST:
				kind = "const"
			case token.VAR:
				kind = "variable"
			default:
				return true
			}
			for _, spec := range x.Specs {
				if vspec, ok := spec.(*ast.ValueSpec); ok && !seenValueSpecs[vspec] {
					seenValueSpecs[vspec] = true
					for _, name := range vspec.Names {
						maybeReport(name, kind)
					}
				}
			}
			return true
		case *ast.TypeSpec:
			maybeReport(x.Name, "type")
			return true
		case *ast.StructType:
			if cfg.qualified && x.Fields != nil {
				for _, field := range x.Fields.List {
					for _, name := range field.Names {
						maybeReport(name, "field")
					}
				}
			}
			return true
		case *ast.InterfaceType:
			if cfg.qualified && x.Methods != nil {
				for _, meth := range x.Methods.List {
					for _, name := range meth.Names {
						maybeReport(name, "method")
					}
				}
			}
			return true
		case *ast.FuncDecl:
			if x.Recv == nil {
				// it's a function
				maybeReport(x.Name, "function")
			} else {
				// it's a method
				if cfg.qualified {
					maybeReport(x.Name, "method")
				}
			}
			// add receivers idents
			if x.Recv != nil {
				for _, field := range x.Recv.List {
					for _, name := range field.Names {
						maybeReport(name, "receiver")
					}
				}
			}
			// Params and Results will be checked in the *ast.FuncType case.
			return true
		case *ast.FuncType:
			// add params idents
			for _, field := range x.Params.List {
				for _, name := range field.Names {
					maybeReport(name, "param")
				}
			}
			// add returns idents
			if x.Results != nil {
				for _, field := range x.Results.List {
					for _, name := range field.Names {
						maybeReport(name, "named return")
					}
				}
			}
			return true
		case *ast.LabeledStmt:
			maybeReport(x.Label, "label")
			return true
		case *ast.AssignStmt:
			// We only care about short variable declarations, which use token.DEFINE.
			if x.Tok == token.DEFINE {
				for _, expr := range x.Lhs {
					if ident, ok := expr.(*ast.Ident); ok {
						maybeReport(ident, "variable")
					}
				}
			}
			return true
		default:
			return true
		}
	})

	return issues
}
