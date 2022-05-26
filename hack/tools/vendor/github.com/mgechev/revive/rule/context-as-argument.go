package rule

import (
	"fmt"
	"go/ast"
	"strings"

	"github.com/mgechev/revive/lint"
)

// ContextAsArgumentRule lints given else constructs.
type ContextAsArgumentRule struct {
	allowTypesLUT map[string]struct{}
}

// Apply applies the rule to given file.
func (r *ContextAsArgumentRule) Apply(file *lint.File, args lint.Arguments) []lint.Failure {

	if r.allowTypesLUT == nil {
		r.allowTypesLUT = getAllowTypesFromArguments(args)
	}

	var failures []lint.Failure
	walker := lintContextArguments{
		allowTypesLUT: r.allowTypesLUT,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	ast.Walk(walker, file.AST)

	return failures
}

// Name returns the rule name.
func (r *ContextAsArgumentRule) Name() string {
	return "context-as-argument"
}

type lintContextArguments struct {
	allowTypesLUT map[string]struct{}
	onFailure     func(lint.Failure)
}

func (w lintContextArguments) Visit(n ast.Node) ast.Visitor {
	fn, ok := n.(*ast.FuncDecl)
	if !ok || len(fn.Type.Params.List) <= 1 {
		return w
	}

	fnArgs := fn.Type.Params.List

	// A context.Context should be the first parameter of a function.
	// Flag any that show up after the first.
	isCtxStillAllowed := true
	for _, arg := range fnArgs {
		argIsCtx := isPkgDot(arg.Type, "context", "Context")
		if argIsCtx && !isCtxStillAllowed {
			w.onFailure(lint.Failure{
				Node:       arg,
				Category:   "arg-order",
				Failure:    "context.Context should be the first parameter of a function",
				Confidence: 0.9,
			})
			break // only flag one
		}

		typeName := gofmt(arg.Type)
		// a parameter of type context.Context is still allowed if the current arg type is in the LUT
		_, isCtxStillAllowed = w.allowTypesLUT[typeName]
	}

	return nil // avoid visiting the function body
}

func getAllowTypesFromArguments(args lint.Arguments) map[string]struct{} {
	allowTypesBefore := []string{}
	if len(args) >= 1 {
		argKV, ok := args[0].(map[string]interface{})
		if !ok {
			panic(fmt.Sprintf("Invalid argument to the context-as-argument rule. Expecting a k,v map, got %T", args[0]))
		}
		for k, v := range argKV {
			switch k {
			case "allowTypesBefore":
				typesBefore, ok := v.(string)
				if !ok {
					panic(fmt.Sprintf("Invalid argument to the context-as-argument.allowTypesBefore rule. Expecting a string, got %T", v))
				}
				allowTypesBefore = append(allowTypesBefore, strings.Split(typesBefore, ",")...)
			default:
				panic(fmt.Sprintf("Invalid argument to the context-as-argument rule. Unrecognized key %s", k))
			}
		}
	}

	result := make(map[string]struct{}, len(allowTypesBefore))
	for _, v := range allowTypesBefore {
		result[v] = struct{}{}
	}

	result["context.Context"] = struct{}{} // context.Context is always allowed before another context.Context
	return result
}
