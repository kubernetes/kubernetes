package rule

import (
	"go/ast"
	"go/token"
	"strconv"
	"unicode"
	"unicode/utf8"

	"github.com/mgechev/revive/lint"
)

// ErrorStringsRule lints given else constructs.
type ErrorStringsRule struct{}

// Apply applies the rule to given file.
func (r *ErrorStringsRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	var errorFunctions = map[string]map[string]struct{}{
		"fmt": {
			"Errorf": {},
		},
		"errors": {
			"Errorf":       {},
			"WithMessage":  {},
			"Wrap":         {},
			"New":          {},
			"WithMessagef": {},
			"Wrapf":        {},
		},
	}

	fileAst := file.AST
	walker := lintErrorStrings{
		file:           file,
		fileAst:        fileAst,
		errorFunctions: errorFunctions,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	ast.Walk(walker, fileAst)

	return failures
}

// Name returns the rule name.
func (r *ErrorStringsRule) Name() string {
	return "error-strings"
}

type lintErrorStrings struct {
	file           *lint.File
	fileAst        *ast.File
	errorFunctions map[string]map[string]struct{}
	onFailure      func(lint.Failure)
}

// Visit browses the AST
func (w lintErrorStrings) Visit(n ast.Node) ast.Visitor {
	ce, ok := n.(*ast.CallExpr)
	if !ok {
		return w
	}

	if len(ce.Args) < 1 {
		return w
	}

	// expression matches the known pkg.function
	ok = w.match(ce)
	if !ok {
		return w
	}

	str, ok := w.getMessage(ce)
	if !ok {
		return w
	}
	s, _ := strconv.Unquote(str.Value) // can assume well-formed Go
	if s == "" {
		return w
	}
	clean, conf := lintErrorString(s)
	if clean {
		return w
	}
	w.onFailure(lint.Failure{
		Node:       str,
		Confidence: conf,
		Category:   "errors",
		Failure:    "error strings should not be capitalized or end with punctuation or a newline",
	})
	return w
}

// match returns true if the expression corresponds to the known pkg.function
// i.e.: errors.Wrap
func (w lintErrorStrings) match(expr *ast.CallExpr) bool {
	sel, ok := expr.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	// retrieve the package
	id, ok := sel.X.(*ast.Ident)
	if !ok {
		return false
	}
	functions, ok := w.errorFunctions[id.Name]
	if !ok {
		return false
	}
	// retrieve the function
	_, ok = functions[sel.Sel.Name]
	return ok
}

// getMessage returns the message depending on its position
// returns false if the cast is unsuccessful
func (w lintErrorStrings) getMessage(expr *ast.CallExpr) (s *ast.BasicLit, success bool) {
	str, ok := w.checkArg(expr, 0)
	if ok {
		return str, true
	}
	if len(expr.Args) < 2 {
		return s, false
	}
	str, ok = w.checkArg(expr, 1)
	if !ok {
		return s, false
	}
	return str, true
}

func (lintErrorStrings) checkArg(expr *ast.CallExpr, arg int) (s *ast.BasicLit, success bool) {
	str, ok := expr.Args[arg].(*ast.BasicLit)
	if !ok {
		return s, false
	}
	if str.Kind != token.STRING {
		return s, false
	}
	return str, true
}

func lintErrorString(s string) (isClean bool, conf float64) {
	const basicConfidence = 0.8
	const capConfidence = basicConfidence - 0.2
	first, firstN := utf8.DecodeRuneInString(s)
	last, _ := utf8.DecodeLastRuneInString(s)
	if last == '.' || last == ':' || last == '!' || last == '\n' {
		return false, basicConfidence
	}
	if unicode.IsUpper(first) {
		// People use proper nouns and exported Go identifiers in error strings,
		// so decrease the confidence of warnings for capitalization.
		if len(s) <= firstN {
			return false, capConfidence
		}
		// Flag strings starting with something that doesn't look like an initialism.
		if second, _ := utf8.DecodeRuneInString(s[firstN:]); !unicode.IsUpper(second) {
			return false, capConfidence
		}
	}
	return true, 0
}
