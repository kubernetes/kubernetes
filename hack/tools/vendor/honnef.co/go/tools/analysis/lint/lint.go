// Package lint provides abstractions on top of go/analysis.
package lint

import (
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/token"
	"strconv"
	"strings"

	"golang.org/x/tools/go/analysis"
)

type Analyzer struct {
	// The analyzer's documentation. Unlike go/analysis.Analyzer.Doc,
	// this field is structured, providing access to severity, options
	// etc.
	Doc      *Documentation
	Analyzer *analysis.Analyzer
}

func (a *Analyzer) initialize() {
	a.Analyzer.Doc = a.Doc.String()
	if a.Analyzer.Flags.Usage == nil {
		fs := flag.NewFlagSet("", flag.PanicOnError)
		fs.Var(newVersionFlag(), "go", "Target Go version")
		a.Analyzer.Flags = *fs
	}
}

func InitializeAnalyzers(docs map[string]*Documentation, analyzers map[string]*analysis.Analyzer) []*Analyzer {
	out := make([]*Analyzer, 0, len(analyzers))
	for k, v := range analyzers {
		v.Name = k
		a := &Analyzer{
			Doc:      docs[k],
			Analyzer: v,
		}
		a.initialize()
		out = append(out, a)
	}
	return out
}

type Severity int

const (
	SeverityNone Severity = iota
	SeverityError
	SeverityDeprecated
	SeverityWarning
	SeverityInfo
	SeverityHint
)

type Documentation struct {
	Title      string
	Text       string
	Since      string
	NonDefault bool
	Options    []string
	Severity   Severity
}

func Markdownify(m map[string]*Documentation) map[string]*Documentation {
	for _, v := range m {
		v.Title = toMarkdown(v.Title)
		v.Text = toMarkdown(v.Text)
	}
	return m
}

func toMarkdown(s string) string {
	return strings.ReplaceAll(s, `\'`, "`")
}

func (doc *Documentation) String() string {
	if doc == nil {
		return "Error: No documentation."
	}

	b := &strings.Builder{}
	fmt.Fprintf(b, "%s\n\n", doc.Title)
	if doc.Text != "" {
		fmt.Fprintf(b, "%s\n\n", doc.Text)
	}
	fmt.Fprint(b, "Available since\n    ")
	if doc.Since == "" {
		fmt.Fprint(b, "unreleased")
	} else {
		fmt.Fprintf(b, "%s", doc.Since)
	}
	if doc.NonDefault {
		fmt.Fprint(b, ", non-default")
	}
	fmt.Fprint(b, "\n")
	if len(doc.Options) > 0 {
		fmt.Fprintf(b, "\nOptions\n")
		for _, opt := range doc.Options {
			fmt.Fprintf(b, "    %s", opt)
		}
		fmt.Fprint(b, "\n")
	}
	return b.String()
}

func newVersionFlag() flag.Getter {
	tags := build.Default.ReleaseTags
	v := tags[len(tags)-1][2:]
	version := new(VersionFlag)
	if err := version.Set(v); err != nil {
		panic(fmt.Sprintf("internal error: %s", err))
	}
	return version
}

type VersionFlag int

func (v *VersionFlag) String() string {
	return fmt.Sprintf("1.%d", *v)
}

func (v *VersionFlag) Set(s string) error {
	if len(s) < 3 {
		return fmt.Errorf("invalid Go version: %q", s)
	}
	if s[0] != '1' {
		return fmt.Errorf("invalid Go version: %q", s)
	}
	if s[1] != '.' {
		return fmt.Errorf("invalid Go version: %q", s)
	}
	i, err := strconv.Atoi(s[2:])
	if err != nil {
		return fmt.Errorf("invalid Go version: %q", s)
	}
	*v = VersionFlag(i)
	return nil
}

func (v *VersionFlag) Get() interface{} {
	return int(*v)
}

// ExhaustiveTypeSwitch panics when called. It can be used to ensure
// that type switches are exhaustive.
func ExhaustiveTypeSwitch(v interface{}) {
	panic(fmt.Sprintf("internal error: unhandled case %T", v))
}

// A directive is a comment of the form '//lint:<command>
// [arguments...]'. It represents instructions to the static analysis
// tool.
type Directive struct {
	Command   string
	Arguments []string
	Directive *ast.Comment
	Node      ast.Node
}

func parseDirective(s string) (cmd string, args []string) {
	if !strings.HasPrefix(s, "//lint:") {
		return "", nil
	}
	s = strings.TrimPrefix(s, "//lint:")
	fields := strings.Split(s, " ")
	return fields[0], fields[1:]
}

func ParseDirectives(files []*ast.File, fset *token.FileSet) []Directive {
	var dirs []Directive
	for _, f := range files {
		// OPT(dh): in our old code, we skip all the comment map work if we
		// couldn't find any directives, benchmark if that's actually
		// worth doing
		cm := ast.NewCommentMap(fset, f, f.Comments)
		for node, cgs := range cm {
			for _, cg := range cgs {
				for _, c := range cg.List {
					if !strings.HasPrefix(c.Text, "//lint:") {
						continue
					}
					cmd, args := parseDirective(c.Text)
					d := Directive{
						Command:   cmd,
						Arguments: args,
						Directive: c,
						Node:      node,
					}
					dirs = append(dirs, d)
				}
			}
		}
	}
	return dirs
}
