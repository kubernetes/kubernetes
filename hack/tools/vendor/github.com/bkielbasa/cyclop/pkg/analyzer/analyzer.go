package analyzer

import (
	"flag"
	"go/ast"
	"go/token"
	"strings"

	"golang.org/x/tools/go/analysis"
)

var (
	flagSet flag.FlagSet
)

var maxComplexity int
var packageAverage float64
var skipTests bool

func NewAnalyzer() *analysis.Analyzer {
	return &analysis.Analyzer{
		Name:  "cyclop",
		Doc:   "calculates cyclomatic complexity",
		Run:   run,
		Flags: flagSet,
	}
}

func init() {
	flagSet.IntVar(&maxComplexity, "maxComplexity", 10, "max complexity the function can have")
	flagSet.Float64Var(&packageAverage, "packageAverage", 0, "max avarage complexity in package")
	flagSet.BoolVar(&skipTests, "skipTests", false, "should the linter execute on test files as well")
}

func run(pass *analysis.Pass) (interface{}, error) {
	var sum, count float64
	var pkgName string
	var pkgPos token.Pos

	for _, f := range pass.Files {
		ast.Inspect(f, func(node ast.Node) bool {
			f, ok := node.(*ast.FuncDecl)
			if !ok {
				if node == nil {
					return true
				}
				if file, ok := node.(*ast.File); ok {
					pkgName = file.Name.Name
					pkgPos = node.Pos()
				}
				// we check function by function
				return true
			}

			if skipTests && testFunc(f) {
				return true
			}

			count++
			comp := complexity(f)
			sum += float64(comp)
			if comp > maxComplexity {
				pass.Reportf(node.Pos(), "calculated cyclomatic complexity for function %s is %d, max is %d", f.Name.Name, comp, maxComplexity)
			}

			return true
		})
	}

	if packageAverage > 0 {
		avg := sum / count
		if avg > packageAverage {
			pass.Reportf(pkgPos, "the avarage complexity for the package %s is %f, max is %f", pkgName, avg, packageAverage)
		}
	}

	return nil, nil
}

func testFunc(f *ast.FuncDecl) bool {
	return strings.HasPrefix(f.Name.Name, "Test")
}

func complexity(fn *ast.FuncDecl) int {
	v := complexityVisitor{}
	ast.Walk(&v, fn)
	return v.Complexity
}

type complexityVisitor struct {
	Complexity int
}

func (v *complexityVisitor) Visit(n ast.Node) ast.Visitor {
	switch n := n.(type) {
	case *ast.FuncDecl, *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt, *ast.CaseClause, *ast.CommClause:
		v.Complexity++
	case *ast.BinaryExpr:
		if n.Op == token.LAND || n.Op == token.LOR {
			v.Complexity++
		}
	}
	return v
}
