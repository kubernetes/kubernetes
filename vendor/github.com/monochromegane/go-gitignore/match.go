package gitignore

import "path/filepath"

type pathMatcher interface {
	match(path string) bool
}

type simpleMatcher struct {
	path string
}

func (m simpleMatcher) match(path string) bool {
	return m.path == path
}

type filepathMatcher struct {
	path string
}

func (m filepathMatcher) match(path string) bool {
	match, _ := filepath.Match(m.path, path)
	return match
}
