package rule

import (
	"fmt"
	"go/ast"
	"strings"

	"github.com/mgechev/revive/lint"
)

// BannedCharsRule checks if a file contains banned characters.
type BannedCharsRule struct {
	bannedCharList []string
}

const bannedCharsRuleName = "banned-characters"

// Apply applied the rule to the given file.
func (r *BannedCharsRule) Apply(file *lint.File, arguments lint.Arguments) []lint.Failure {
	if r.bannedCharList == nil {
		checkNumberOfArguments(1, arguments, bannedCharsRuleName)
		r.bannedCharList = r.getBannedCharsList(arguments)
	}

	var failures []lint.Failure
	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintBannedCharsRule{
		bannedChars: r.bannedCharList,
		onFailure:   onFailure,
	}
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name
func (r *BannedCharsRule) Name() string {
	return bannedCharsRuleName
}

// getBannedCharsList converts arguments into the banned characters list
func (r *BannedCharsRule) getBannedCharsList(args lint.Arguments) []string {
	var bannedChars []string
	for _, char := range args {
		charStr, ok := char.(string)
		if !ok {
			panic(fmt.Sprintf("Invalid argument for the %s rule: expecting a string, got %T", r.Name(), char))
		}
		bannedChars = append(bannedChars, charStr)
	}

	return bannedChars
}

type lintBannedCharsRule struct {
	bannedChars []string
	onFailure   func(lint.Failure)
}

// Visit checks for each node if an identifier contains banned characters
func (w lintBannedCharsRule) Visit(node ast.Node) ast.Visitor {
	n, ok := node.(*ast.Ident)
	if !ok {
		return w
	}
	for _, c := range w.bannedChars {
		ok := strings.Contains(n.Name, c)
		if ok {
			w.onFailure(lint.Failure{
				Confidence: 1,
				Failure:    fmt.Sprintf("banned character found: %s", c),
				RuleName:   bannedCharsRuleName,
				Node:       n,
			})
		}
	}

	return w
}
