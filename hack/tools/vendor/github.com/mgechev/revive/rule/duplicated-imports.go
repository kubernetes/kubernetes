package rule

import (
	"fmt"

	"github.com/mgechev/revive/lint"
)

// DuplicatedImportsRule lints given else constructs.
type DuplicatedImportsRule struct{}

// Apply applies the rule to given file.
func (r *DuplicatedImportsRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	impPaths := map[string]struct{}{}
	for _, imp := range file.AST.Imports {
		path := imp.Path.Value
		_, ok := impPaths[path]
		if ok {
			failures = append(failures, lint.Failure{
				Confidence: 1,
				Failure:    fmt.Sprintf("Package %s already imported", path),
				Node:       imp,
				Category:   "imports",
			})
			continue
		}

		impPaths[path] = struct{}{}
	}

	return failures
}

// Name returns the rule name.
func (r *DuplicatedImportsRule) Name() string {
	return "duplicated-imports"
}
