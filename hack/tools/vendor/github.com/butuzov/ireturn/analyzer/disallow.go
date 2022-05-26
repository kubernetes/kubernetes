package analyzer

import (
	"go/ast"
	"strings"
)

const nolintPrefix = "//nolint"

func hasDisallowDirective(cg *ast.CommentGroup) bool {
	if cg == nil {
		return false
	}

	return directiveFound(cg)
}

func directiveFound(cg *ast.CommentGroup) bool {
	for i := len(cg.List) - 1; i >= 0; i-- {
		comment := cg.List[i]
		if !strings.HasPrefix(comment.Text, nolintPrefix) {
			continue
		}

		startingIdx := len(nolintPrefix)
		for {
			idx := strings.Index(comment.Text[startingIdx:], name)
			if idx == -1 {
				break
			}

			if len(comment.Text[startingIdx+idx:]) == len(name) {
				return true
			}

			c := comment.Text[startingIdx+idx+len(name)]
			if c == '.' || c == ',' || c == ' ' || c == '	' {
				return true
			}
			startingIdx += idx + 1
		}
	}

	return false
}
