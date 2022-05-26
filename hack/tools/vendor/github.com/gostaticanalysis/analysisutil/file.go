package analysisutil

import (
	"go/ast"
	"go/token"
	"regexp"

	"golang.org/x/tools/go/analysis"
)

// File finds *ast.File in pass.Files by pos.
func File(pass *analysis.Pass, pos token.Pos) *ast.File {
	for _, f := range pass.Files {
		if f.Pos() <= pos && pos <= f.End() {
			return f
		}
	}
	return nil
}

var genCommentRegexp = regexp.MustCompile(`^// Code generated .* DO NOT EDIT\.$`)

// IsGeneratedFile reports whether the file has been generated automatically.
// If file is nil, IsGeneratedFile will return false.
func IsGeneratedFile(file *ast.File) bool {
	if file == nil || len(file.Comments) == 0 {
		return false
	}
	return genCommentRegexp.MatchString(file.Comments[0].List[0].Text)
}
