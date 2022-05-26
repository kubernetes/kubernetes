// Package godot checks if comments contain a period at the end of the last
// sentence if needed.
package godot

import (
	"fmt"
	"go/ast"
	"go/token"
	"io/ioutil"
	"os"
	"regexp"
	"sort"
	"strings"
)

// NOTE: Line and column indexes are 1-based.

// NOTE: Errors `invalid line number inside comment...` should never happen.
// Their goal is to prevent panic, if there's a bug with array indexes.

// Issue contains a description of linting error and a recommended replacement.
type Issue struct {
	Pos         token.Position
	Message     string
	Replacement string
}

// position is a position inside a comment (might be multiline comment).
type position struct {
	line   int // starts at 1
	column int // starts at 1, byte count
}

// comment is an internal representation of AST comment entity with additional
// data attached. The latter is used for creating a full replacement for
// the line with issues.
type comment struct {
	lines []string       // unmodified lines from file
	text  string         // concatenated `lines` with special parts excluded
	start token.Position // position of the first symbol in comment
	decl  bool           // whether comment is a declaration comment
}

// Run runs this linter on the provided code.
func Run(file *ast.File, fset *token.FileSet, settings Settings) ([]Issue, error) {
	pf, err := newParsedFile(file, fset)
	if err == errEmptyInput || err == errUnsuitableInput {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("parse input file: %v", err)
	}

	exclude := make([]*regexp.Regexp, len(settings.Exclude))
	for i := 0; i < len(settings.Exclude); i++ {
		exclude[i], err = regexp.Compile(settings.Exclude[i])
		if err != nil {
			return nil, fmt.Errorf("invalid regexp: %v", err)
		}
	}

	comments := pf.getComments(settings.Scope, exclude)
	issues := checkComments(comments, settings)
	sortIssues(issues)

	return issues, nil
}

// Fix fixes all issues and returns new version of file content.
func Fix(path string, file *ast.File, fset *token.FileSet, settings Settings) ([]byte, error) {
	// Read file
	content, err := ioutil.ReadFile(path) // nolint: gosec
	if err != nil {
		return nil, fmt.Errorf("read file: %v", err)
	}
	if len(content) == 0 {
		return nil, nil
	}

	issues, err := Run(file, fset, settings)
	if err != nil {
		return nil, fmt.Errorf("run linter: %v", err)
	}

	// slice -> map
	m := map[int]Issue{}
	for _, iss := range issues {
		m[iss.Pos.Line] = iss
	}

	// Replace lines from issues
	fixed := make([]byte, 0, len(content))
	for i, line := range strings.Split(string(content), "\n") {
		newline := line
		if iss, ok := m[i+1]; ok {
			newline = iss.Replacement
		}
		fixed = append(fixed, []byte(newline+"\n")...)
	}
	fixed = fixed[:len(fixed)-1] // trim last "\n"

	return fixed, nil
}

// Replace rewrites original file with it's fixed version.
func Replace(path string, file *ast.File, fset *token.FileSet, settings Settings) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("check file: %v", err)
	}
	mode := info.Mode()

	fixed, err := Fix(path, file, fset, settings)
	if err != nil {
		return fmt.Errorf("fix issues: %v", err)
	}

	if err := ioutil.WriteFile(path, fixed, mode); err != nil {
		return fmt.Errorf("write file: %v", err)
	}
	return nil
}

// sortIssues sorts by filename, line and column.
func sortIssues(iss []Issue) {
	sort.Slice(iss, func(i, j int) bool {
		if iss[i].Pos.Filename != iss[j].Pos.Filename {
			return iss[i].Pos.Filename < iss[j].Pos.Filename
		}
		if iss[i].Pos.Line != iss[j].Pos.Line {
			return iss[i].Pos.Line < iss[j].Pos.Line
		}
		return iss[i].Pos.Column < iss[j].Pos.Column
	})
}
