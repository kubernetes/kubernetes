package decorder

import (
	"go/ast"
	"go/token"
	"strings"

	"golang.org/x/tools/go/analysis"
)

type (
	decNumChecker struct {
		tokenMap    map[string]token.Token
		tokenCounts map[token.Token]int
		decOrder    []string
		funcPoss    []funcPos
	}

	funcPos struct {
		start token.Pos
		end   token.Pos
	}
)

const (
	Name = "decorder"

	FlagDo    = "dec-order"
	FlagDdnc  = "disable-dec-num-check"
	FlagDdoc  = "disable-dec-order-check"
	FlagDiffc = "disable-init-func-first-check"
)

var (
	Analyzer = &analysis.Analyzer{
		Name: Name,
		Doc:  "check declaration order and count of types, constants, variables and functions",
		Run:  run,
	}

	decOrder                  string
	disableDecNumCheck        bool
	disableDecOrderCheck      bool
	disableInitFuncFirstCheck bool

	tokens = []token.Token{token.TYPE, token.CONST, token.VAR, token.FUNC}
)

//nolint:lll
func init() {
	Analyzer.Flags.StringVar(&decOrder, FlagDo, "type,const,var,func", "define the required order of types, constants, variables and functions declarations inside a file")
	Analyzer.Flags.BoolVar(&disableDecNumCheck, FlagDdnc, false, "option to disable check for number of e.g. var declarations inside file")
	Analyzer.Flags.BoolVar(&disableDecOrderCheck, FlagDdoc, false, "option to disable check for order of declarations inside file")
	Analyzer.Flags.BoolVar(&disableInitFuncFirstCheck, FlagDiffc, false, "option to disable check that init function is always first function in file")
}

func run(pass *analysis.Pass) (interface{}, error) {
	for _, f := range pass.Files {
		ast.Inspect(f, runDeclNumAndDecOrderCheck(pass))

		if !disableInitFuncFirstCheck {
			ast.Inspect(f, runInitFuncFirstCheck(pass))
		}
	}

	return nil, nil
}

func runInitFuncFirstCheck(pass *analysis.Pass) func(ast.Node) bool {
	nonInitFound := false

	return func(n ast.Node) bool {
		dec, ok := n.(*ast.FuncDecl)
		if !ok {
			return true
		}

		if dec.Name.Name == "init" {
			if nonInitFound {
				pass.Reportf(dec.Pos(), "init func must be the first function in file")
			}
		} else {
			nonInitFound = true
		}

		return true
	}
}

func runDeclNumAndDecOrderCheck(pass *analysis.Pass) func(ast.Node) bool {
	dnc := newDecNumChecker()

	return func(n ast.Node) bool {
		fd, ok := n.(*ast.FuncDecl)
		if ok {
			return dnc.handleFuncDec(fd, pass)
		}

		gd, ok := n.(*ast.GenDecl)
		if !ok {
			return true
		}

		if dnc.isInsideFunction(gd) {
			return true
		}

		if !disableDecNumCheck {
			dnc.handleDecNumCheck(gd, pass)
		}

		if !disableDecOrderCheck {
			dnc.handleDecOrderCheck(gd, pass)
		}

		return true
	}
}

func newDecNumChecker() decNumChecker {
	dnc := decNumChecker{
		tokenMap:    map[string]token.Token{},
		tokenCounts: map[token.Token]int{},
		decOrder:    []string{},
		funcPoss:    []funcPos{},
	}

	for _, t := range tokens {
		dnc.tokenCounts[t] = 0
		dnc.tokenMap[t.String()] = t
	}

	for _, do := range strings.Split(decOrder, ",") {
		dnc.decOrder = append(dnc.decOrder, strings.TrimSpace(do))
	}

	return dnc
}

func (dnc decNumChecker) isToLate(t token.Token) (string, bool) {
	for i, do := range dnc.decOrder {
		if do == t.String() {
			for j := i + 1; j < len(dnc.decOrder); j++ {
				if dnc.tokenCounts[dnc.tokenMap[dnc.decOrder[j]]] > 0 {
					return dnc.decOrder[j], false
				}
			}
			return "", true
		}
	}

	return "", true
}

func (dnc *decNumChecker) handleDecNumCheck(gd *ast.GenDecl, pass *analysis.Pass) {
	for _, t := range tokens {
		if gd.Tok == t {
			dnc.tokenCounts[t]++

			if dnc.tokenCounts[t] > 1 {
				pass.Reportf(gd.Pos(), "multiple \"%s\" declarations are not allowed; use parentheses instead", t.String())
			}
		}
	}
}

func (dnc decNumChecker) handleDecOrderCheck(gd *ast.GenDecl, pass *analysis.Pass) {
	l, c := dnc.isToLate(gd.Tok)
	if !c {
		pass.Reportf(gd.Pos(), "%s must not be placed after %s", gd.Tok.String(), l)
	}
}

func (dnc decNumChecker) isInsideFunction(dn *ast.GenDecl) bool {
	for _, poss := range dnc.funcPoss {
		if poss.start < dn.Pos() && poss.end > dn.Pos() {
			return true
		}
	}
	return false
}

func (dnc *decNumChecker) handleFuncDec(fd *ast.FuncDecl, pass *analysis.Pass) bool {
	dnc.funcPoss = append(dnc.funcPoss, funcPos{start: fd.Pos(), end: fd.End()})

	dnc.tokenCounts[token.FUNC]++

	if !disableDecOrderCheck {
		l, c := dnc.isToLate(token.FUNC)
		if !c {
			pass.Reportf(fd.Pos(), "%s must not be placed after %s", token.FUNC.String(), l)
		}
	}

	return true
}
