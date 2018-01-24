package githttp

import (
	"errors"
	"io"
	"regexp"
)

// GitReader scans for errors in the output of a git command
type GitReader struct {
	// Underlying reader (to relay calls to)
	io.Reader

	// Error
	GitError error
}

// Regex to detect errors
var (
	gitErrorRegex = regexp.MustCompile("error: (.*)")
)

// Implement the io.Reader interface
func (g *GitReader) Read(p []byte) (n int, err error) {
	// Relay call
	n, err = g.Reader.Read(p)

	// Scan for errors
	g.scan(p)

	return n, err
}

func (g *GitReader) scan(data []byte) {
	// Already got an error
	// the main error will be the first error line
	if g.GitError != nil {
		return
	}

	matches := gitErrorRegex.FindSubmatch(data)

	// Skip, no matches found
	if matches == nil {
		return
	}

	g.GitError = errors.New(string(matches[1][:]))
}
