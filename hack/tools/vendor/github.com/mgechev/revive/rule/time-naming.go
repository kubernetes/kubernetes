package rule

import (
	"fmt"
	"go/ast"
	"go/types"
	"strings"

	"github.com/mgechev/revive/lint"
)

// TimeNamingRule lints given else constructs.
type TimeNamingRule struct{}

// Apply applies the rule to given file.
func (r *TimeNamingRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := &lintTimeNames{file, onFailure}

	file.Pkg.TypeCheck()
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (r *TimeNamingRule) Name() string {
	return "time-naming"
}

type lintTimeNames struct {
	file      *lint.File
	onFailure func(lint.Failure)
}

func (w *lintTimeNames) Visit(node ast.Node) ast.Visitor {
	v, ok := node.(*ast.ValueSpec)
	if !ok {
		return w
	}
	for _, name := range v.Names {
		origTyp := w.file.Pkg.TypeOf(name)
		// Look for time.Duration or *time.Duration;
		// the latter is common when using flag.Duration.
		typ := origTyp
		if pt, ok := typ.(*types.Pointer); ok {
			typ = pt.Elem()
		}
		if !isNamedType(typ, "time", "Duration") {
			continue
		}
		suffix := ""
		for _, suf := range timeSuffixes {
			if strings.HasSuffix(name.Name, suf) {
				suffix = suf
				break
			}
		}
		if suffix == "" {
			continue
		}
		w.onFailure(lint.Failure{
			Category:   "time",
			Confidence: 0.9,
			Node:       v,
			Failure:    fmt.Sprintf("var %s is of type %v; don't use unit-specific suffix %q", name.Name, origTyp, suffix),
		})
	}
	return w
}

// timeSuffixes is a list of name suffixes that imply a time unit.
// This is not an exhaustive list.
var timeSuffixes = []string{
	"Hour", "Hours",
	"Min", "Mins", "Minutes", "Minute",
	"Sec", "Secs", "Seconds", "Second",
	"Msec", "Msecs",
	"Milli", "Millis", "Milliseconds", "Millisecond",
	"Usec", "Usecs", "Microseconds", "Microsecond",
	"MS", "Ms",
}

func isNamedType(typ types.Type, importPath, name string) bool {
	n, ok := typ.(*types.Named)
	if !ok {
		return false
	}
	tn := n.Obj()
	return tn != nil && tn.Pkg() != nil && tn.Pkg().Path() == importPath && tn.Name() == name
}
