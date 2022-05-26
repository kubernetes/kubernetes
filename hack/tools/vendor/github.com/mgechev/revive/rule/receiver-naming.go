package rule

import (
	"fmt"
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// ReceiverNamingRule lints given else constructs.
type ReceiverNamingRule struct{}

// Apply applies the rule to given file.
func (r *ReceiverNamingRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	fileAst := file.AST
	walker := lintReceiverName{
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
		typeReceiver: map[string]string{},
	}

	ast.Walk(walker, fileAst)

	return failures
}

// Name returns the rule name.
func (r *ReceiverNamingRule) Name() string {
	return "receiver-naming"
}

type lintReceiverName struct {
	onFailure    func(lint.Failure)
	typeReceiver map[string]string
}

func (w lintReceiverName) Visit(n ast.Node) ast.Visitor {
	fn, ok := n.(*ast.FuncDecl)
	if !ok || fn.Recv == nil || len(fn.Recv.List) == 0 {
		return w
	}
	names := fn.Recv.List[0].Names
	if len(names) < 1 {
		return w
	}
	name := names[0].Name
	const ref = styleGuideBase + "#receiver-names"
	if name == "_" {
		w.onFailure(lint.Failure{
			Node:       n,
			Confidence: 1,
			Category:   "naming",
			Failure:    "receiver name should not be an underscore, omit the name if it is unused",
		})
		return w
	}
	if name == "this" || name == "self" {
		w.onFailure(lint.Failure{
			Node:       n,
			Confidence: 1,
			Category:   "naming",
			Failure:    `receiver name should be a reflection of its identity; don't use generic names such as "this" or "self"`,
		})
		return w
	}
	recv := receiverType(fn)
	if prev, ok := w.typeReceiver[recv]; ok && prev != name {
		w.onFailure(lint.Failure{
			Node:       n,
			Confidence: 1,
			Category:   "naming",
			Failure:    fmt.Sprintf("receiver name %s should be consistent with previous receiver name %s for %s", name, prev, recv),
		})
		return w
	}
	w.typeReceiver[recv] = name
	return w
}
