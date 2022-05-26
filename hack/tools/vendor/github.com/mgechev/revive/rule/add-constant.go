package rule

import (
	"fmt"
	"go/ast"
	"strconv"
	"strings"

	"github.com/mgechev/revive/lint"
)

const (
	defaultStrLitLimit = 2
	kindFLOAT          = "FLOAT"
	kindINT            = "INT"
	kindSTRING         = "STRING"
)

type whiteList map[string]map[string]bool

func newWhiteList() whiteList {
	return map[string]map[string]bool{kindINT: {}, kindFLOAT: {}, kindSTRING: {}}
}

func (wl whiteList) add(kind string, list string) {
	elems := strings.Split(list, ",")
	for _, e := range elems {
		wl[kind][e] = true
	}
}

// AddConstantRule lints unused params in functions.
type AddConstantRule struct {
	whiteList   whiteList
	strLitLimit int
}

// Apply applies the rule to given file.
func (r *AddConstantRule) Apply(file *lint.File, arguments lint.Arguments) []lint.Failure {
	if r.whiteList == nil {
		r.strLitLimit = defaultStrLitLimit
		r.whiteList = newWhiteList()
		if len(arguments) > 0 {
			args, ok := arguments[0].(map[string]interface{})
			if !ok {
				panic(fmt.Sprintf("Invalid argument to the add-constant rule. Expecting a k,v map, got %T", arguments[0]))
			}
			for k, v := range args {
				kind := ""
				switch k {
				case "allowFloats":
					kind = kindFLOAT
					fallthrough
				case "allowInts":
					if kind == "" {
						kind = kindINT
					}
					fallthrough
				case "allowStrs":
					if kind == "" {
						kind = kindSTRING
					}
					list, ok := v.(string)
					if !ok {
						panic(fmt.Sprintf("Invalid argument to the add-constant rule, string expected. Got '%v' (%T)", v, v))
					}
					r.whiteList.add(kind, list)
				case "maxLitCount":
					sl, ok := v.(string)
					if !ok {
						panic(fmt.Sprintf("Invalid argument to the add-constant rule, expecting string representation of an integer. Got '%v' (%T)", v, v))
					}

					limit, err := strconv.Atoi(sl)
					if err != nil {
						panic(fmt.Sprintf("Invalid argument to the add-constant rule, expecting string representation of an integer. Got '%v'", v))
					}
					r.strLitLimit = limit
				}
			}
		}
	}

	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintAddConstantRule{onFailure: onFailure, strLits: make(map[string]int), strLitLimit: r.strLitLimit, whiteLst: r.whiteList}

	ast.Walk(w, file.AST)

	return failures
}

// Name returns the rule name.
func (r *AddConstantRule) Name() string {
	return "add-constant"
}

type lintAddConstantRule struct {
	onFailure   func(lint.Failure)
	strLits     map[string]int
	strLitLimit int
	whiteLst    whiteList
}

func (w lintAddConstantRule) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.GenDecl:
		return nil // skip declarations
	case *ast.BasicLit:
		switch kind := n.Kind.String(); kind {
		case kindFLOAT, kindINT:
			w.checkNumLit(kind, n)
		case kindSTRING:
			w.checkStrLit(n)
		}
	}

	return w

}

func (w lintAddConstantRule) checkStrLit(n *ast.BasicLit) {
	if w.whiteLst[kindSTRING][n.Value] {
		return
	}

	count := w.strLits[n.Value]
	if count >= 0 {
		w.strLits[n.Value] = count + 1
		if w.strLits[n.Value] > w.strLitLimit {
			w.onFailure(lint.Failure{
				Confidence: 1,
				Node:       n,
				Category:   "style",
				Failure:    fmt.Sprintf("string literal %s appears, at least, %d times, create a named constant for it", n.Value, w.strLits[n.Value]),
			})
			w.strLits[n.Value] = -1 // mark it to avoid failing again on the same literal
		}
	}
}

func (w lintAddConstantRule) checkNumLit(kind string, n *ast.BasicLit) {
	if w.whiteLst[kind][n.Value] {
		return
	}

	w.onFailure(lint.Failure{
		Confidence: 1,
		Node:       n,
		Category:   "style",
		Failure:    fmt.Sprintf("avoid magic numbers like '%s', create a named constant for it", n.Value),
	})
}
