package git

import "strings"

const defaultDotGitPath = ".git"

// countLines returns the number of lines in a string à la git, this is
// The newline character is assumed to be '\n'.  The empty string
// contains 0 lines.  If the last line of the string doesn't end with a
// newline, it will still be considered a line.
func countLines(s string) int {
	if s == "" {
		return 0
	}

	nEOL := strings.Count(s, "\n")
	if strings.HasSuffix(s, "\n") {
		return nEOL
	}

	return nEOL + 1
}
