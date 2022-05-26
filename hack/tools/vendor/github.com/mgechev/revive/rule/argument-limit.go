package rule

import (
	"fmt"
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// ArgumentsLimitRule lints given else constructs.
type ArgumentsLimitRule struct {
	total int
}

// Apply applies the rule to given file.
func (r *ArgumentsLimitRule) Apply(file *lint.File, arguments lint.Arguments) []lint.Failure {
	if r.total == 0 {
		checkNumberOfArguments(1, arguments, r.Name())

		total, ok := arguments[0].(int64) // Alt. non panicking version
		if !ok {
			panic(`invalid value passed as argument number to the "argument-limit" rule`)
		}
		r.total = int(total)
	}

	var failures []lint.Failure

	walker := lintArgsNum{
		total: r.total,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	ast.Walk(walker, file.AST)

	return failures
}

// Name returns the rule name.
func (r *ArgumentsLimitRule) Name() string {
	return "argument-limit"
}

type lintArgsNum struct {
	total     int
	onFailure func(lint.Failure)
}

func (w lintArgsNum) Visit(n ast.Node) ast.Visitor {
	node, ok := n.(*ast.FuncDecl)
	if ok {
		num := 0
		for _, l := range node.Type.Params.List {
			for range l.Names {
				num++
			}
		}
		if num > w.total {
			w.onFailure(lint.Failure{
				Confidence: 1,
				Failure:    fmt.Sprintf("maximum number of arguments per function exceeded; max %d but got %d", w.total, num),
				Node:       node.Type,
			})
			return w
		}
	}
	return w
}
