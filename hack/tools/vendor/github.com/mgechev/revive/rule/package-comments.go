package rule

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"

	"github.com/mgechev/revive/lint"
)

// PackageCommentsRule lints the package comments. It complains if
// there is no package comment, or if it is not of the right form.
// This has a notable false positive in that a package comment
// could rightfully appear in a different file of the same package,
// but that's not easy to fix since this linter is file-oriented.
type PackageCommentsRule struct{}

// Apply applies the rule to given file.
func (r *PackageCommentsRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	if isTest(file) {
		return failures
	}

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	fileAst := file.AST
	w := &lintPackageComments{fileAst, file, onFailure}
	ast.Walk(w, fileAst)
	return failures
}

// Name returns the rule name.
func (r *PackageCommentsRule) Name() string {
	return "package-comments"
}

type lintPackageComments struct {
	fileAst   *ast.File
	file      *lint.File
	onFailure func(lint.Failure)
}

func (l *lintPackageComments) Visit(_ ast.Node) ast.Visitor {
	if l.file.IsTest() {
		return nil
	}

	const ref = styleGuideBase + "#package-comments"
	prefix := "Package " + l.fileAst.Name.Name + " "

	// Look for a detached package comment.
	// First, scan for the last comment that occurs before the "package" keyword.
	var lastCG *ast.CommentGroup
	for _, cg := range l.fileAst.Comments {
		if cg.Pos() > l.fileAst.Package {
			// Gone past "package" keyword.
			break
		}
		lastCG = cg
	}
	if lastCG != nil && strings.HasPrefix(lastCG.Text(), prefix) {
		endPos := l.file.ToPosition(lastCG.End())
		pkgPos := l.file.ToPosition(l.fileAst.Package)
		if endPos.Line+1 < pkgPos.Line {
			// There isn't a great place to anchor this error;
			// the start of the blank lines between the doc and the package statement
			// is at least pointing at the location of the problem.
			pos := token.Position{
				Filename: endPos.Filename,
				// Offset not set; it is non-trivial, and doesn't appear to be needed.
				Line:   endPos.Line + 1,
				Column: 1,
			}
			l.onFailure(lint.Failure{
				Category: "comments",
				Position: lint.FailurePosition{
					Start: pos,
					End:   pos,
				},
				Confidence: 0.9,
				Failure:    "package comment is detached; there should be no blank lines between it and the package statement",
			})
			return nil
		}
	}

	if l.fileAst.Doc == nil {
		l.onFailure(lint.Failure{
			Category:   "comments",
			Node:       l.fileAst,
			Confidence: 0.2,
			Failure:    "should have a package comment, unless it's in another file for this package",
		})
		return nil
	}
	s := l.fileAst.Doc.Text()
	if ts := strings.TrimLeft(s, " \t"); ts != s {
		l.onFailure(lint.Failure{
			Category:   "comments",
			Node:       l.fileAst.Doc,
			Confidence: 1,
			Failure:    "package comment should not have leading space",
		})
		s = ts
	}
	// Only non-main packages need to keep to this form.
	if !l.file.Pkg.IsMain() && !strings.HasPrefix(s, prefix) {
		l.onFailure(lint.Failure{
			Category:   "comments",
			Node:       l.fileAst.Doc,
			Confidence: 1,
			Failure:    fmt.Sprintf(`package comment should be of the form "%s..."`, prefix),
		})
	}
	return nil
}
