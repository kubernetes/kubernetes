package gitignore

import "strings"

// Only benchmark use
type fullScanPatterns struct {
	absolute patterns
	relative patterns
}

func newFullScanPatterns() *fullScanPatterns {
	return &fullScanPatterns{
		absolute: patterns{},
		relative: patterns{},
	}
}

func (ps *fullScanPatterns) add(pattern string) {
	if strings.HasPrefix(pattern, "/") {
		ps.absolute.add(newPattern(pattern))
	} else {
		ps.relative.add(newPattern(pattern))
	}
}

func (ps fullScanPatterns) match(path string, isDir bool) bool {
	if ps.absolute.match(path, isDir) {
		return true
	}
	return ps.relative.match(path, isDir)
}
