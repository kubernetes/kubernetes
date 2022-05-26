package errorlint

import (
	"flag"
	"go/ast"
	"go/types"
	"sort"

	"golang.org/x/tools/go/analysis"
)

func NewAnalyzer() *analysis.Analyzer {
	return &analysis.Analyzer{
		Name:  "errorlint",
		Doc:   "Source code linter for Go software that can be used to find code that will cause problems with the error wrapping scheme introduced in Go 1.13.",
		Run:   run,
		Flags: flagSet,
	}
}

var (
	flagSet         flag.FlagSet
	checkComparison bool
	checkAsserts    bool
	checkErrorf     bool
)

func init() {
	flagSet.BoolVar(&checkComparison, "comparison", true, "Check for plain error comparisons")
	flagSet.BoolVar(&checkAsserts, "asserts", true, "Check for plain type assertions and type switches")
	flagSet.BoolVar(&checkErrorf, "errorf", false, "Check whether fmt.Errorf uses the %w verb for formatting errors. See the readme for caveats")
}

func run(pass *analysis.Pass) (interface{}, error) {
	lints := []Lint{}
	extInfo := newTypesInfoExt(pass.TypesInfo)
	if checkComparison {
		l := LintErrorComparisons(pass.Fset, extInfo)
		lints = append(lints, l...)
	}
	if checkAsserts {
		l := LintErrorTypeAssertions(pass.Fset, *pass.TypesInfo)
		lints = append(lints, l...)
	}
	if checkErrorf {
		l := LintFmtErrorfCalls(pass.Fset, *pass.TypesInfo)
		lints = append(lints, l...)
	}
	sort.Sort(ByPosition(lints))

	for _, l := range lints {
		pass.Report(analysis.Diagnostic{Pos: l.Pos, Message: l.Message})
	}
	return nil, nil
}

type TypesInfoExt struct {
	types.Info

	// Maps AST nodes back to the node they are contain within.
	NodeParent map[ast.Node]ast.Node

	// Maps an object back to all identifiers to refer to it.
	IdentifiersForObject map[types.Object][]*ast.Ident
}

func newTypesInfoExt(info *types.Info) *TypesInfoExt {
	nodeParent := map[ast.Node]ast.Node{}
	for node := range info.Scopes {
		file, ok := node.(*ast.File)
		if !ok {
			continue
		}
		stack := []ast.Node{file}
		ast.Inspect(file, func(n ast.Node) bool {
			nodeParent[n] = stack[len(stack)-1]
			if n == nil {
				stack = stack[:len(stack)-1]
			} else {
				stack = append(stack, n)
			}
			return true
		})
	}

	identifiersForObject := map[types.Object][]*ast.Ident{}
	for node, obj := range info.Defs {
		identifiersForObject[obj] = append(identifiersForObject[obj], node)
	}
	for node, obj := range info.Uses {
		identifiersForObject[obj] = append(identifiersForObject[obj], node)
	}

	return &TypesInfoExt{
		Info:                 *info,
		NodeParent:           nodeParent,
		IdentifiersForObject: identifiersForObject,
	}
}
