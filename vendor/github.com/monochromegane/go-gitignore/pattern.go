package gitignore

import (
	"path/filepath"
	"strings"
)

var Separator = string(filepath.Separator)

type pattern struct {
	hasRootPrefix     bool
	hasDirSuffix      bool
	pathDepth         int
	matcher           pathMatcher
	onlyEqualizedPath bool
}

func newPattern(path string) pattern {
	hasRootPrefix := path[0] == '/'
	hasDirSuffix := path[len(path)-1] == '/'

	var pathDepth int
	if !hasRootPrefix {
		pathDepth = strings.Count(path, "/")
	}

	var matcher pathMatcher
	matchingPath := strings.Trim(path, "/")
	if hasMeta(path) {
		matcher = filepathMatcher{path: matchingPath}
	} else {
		matcher = simpleMatcher{path: matchingPath}
	}

	return pattern{
		hasRootPrefix: hasRootPrefix,
		hasDirSuffix:  hasDirSuffix,
		pathDepth:     pathDepth,
		matcher:       matcher,
	}
}

func newPatternForEqualizedPath(path string) pattern {
	pattern := newPattern(path)
	pattern.onlyEqualizedPath = true
	return pattern
}

func (p pattern) match(path string, isDir bool) bool {

	if p.hasDirSuffix && !isDir {
		return false
	}

	var targetPath string
	if p.hasRootPrefix || p.onlyEqualizedPath {
		// absolute pattern or only equalized path mode
		targetPath = path
	} else {
		// relative pattern
		targetPath = p.equalizeDepth(path)
	}
	return p.matcher.match(targetPath)
}

func (p pattern) equalizeDepth(path string) string {
	equalizedPath, _ := cutLastN(path, p.pathDepth+1)
	return equalizedPath
}
