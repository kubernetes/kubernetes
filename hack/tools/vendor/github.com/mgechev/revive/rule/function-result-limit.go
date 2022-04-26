package rule

import (
	"fmt"
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// FunctionResultsLimitRule lints given else constructs.
type FunctionResultsLimitRule struct {
	max int
}

// Apply applies the rule to given file.
func (r *FunctionResultsLimitRule) Apply(file *lint.File, arguments lint.Arguments) []lint.Failure {
	if r.max == 0 {
		checkNumberOfArguments(1, arguments, r.Name())

		max, ok := arguments[0].(int64) // Alt. non panicking version
		if !ok {
			panic(fmt.Sprintf(`invalid value passed as return results number to the "function-result-limit" rule; need int64 but got %T`, arguments[0]))
		}
		if max < 0 {
			panic(`the value passed as return results number to the "function-result-limit" rule cannot be negative`)
		}
		r.max = int(max)
	}

	var failures []lint.Failure

	walker := lintFunctionResultsNum{
		max: r.max,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	ast.Walk(walker, file.AST)

	return failures
}

// Name returns the rule name.
func (r *FunctionResultsLimitRule) Name() string {
	return "function-result-limit"
}

type lintFunctionResultsNum struct {
	max       int
	onFailure func(lint.Failure)
}

func (w lintFunctionResultsNum) Visit(n ast.Node) ast.Visitor {
	node, ok := n.(*ast.FuncDecl)
	if ok {
		num := 0
		if node.Type.Results != nil {
			num = node.Type.Results.NumFields()
		}
		if num > w.max {
			w.onFailure(lint.Failure{
				Confidence: 1,
				Failure:    fmt.Sprintf("maximum number of return results per function exceeded; max %d but got %d", w.max, num),
				Node:       node.Type,
			})
			return w
		}
	}
	return w
}
