package gitignore

import "strings"

const initials = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ."

type initialPatternHolder struct {
	patterns      initialPatterns
	otherPatterns *patterns
}

func newInitialPatternHolder() initialPatternHolder {
	return initialPatternHolder{
		patterns:      initialPatterns{m: map[byte]*patterns{}},
		otherPatterns: &patterns{},
	}
}

func (h *initialPatternHolder) add(pattern string) {
	trimedPattern := strings.TrimPrefix(pattern, "/")
	if strings.IndexAny(trimedPattern[0:1], initials) != -1 {
		h.patterns.set(trimedPattern[0], newPatternForEqualizedPath(pattern))
	} else {
		h.otherPatterns.add(newPatternForEqualizedPath(pattern))
	}
}

func (h initialPatternHolder) match(path string, isDir bool) bool {
	if h.patterns.size() == 0 && h.otherPatterns.size() == 0 {
		return false
	}
	if patterns, ok := h.patterns.get(path[0]); ok {
		if patterns.match(path, isDir) {
			return true
		}
	}
	return h.otherPatterns.match(path, isDir)
}

type initialPatterns struct {
	m map[byte]*patterns
}

func (p *initialPatterns) set(initial byte, pattern pattern) {
	if ps, ok := p.m[initial]; ok {
		ps.add(pattern)
	} else {
		patterns := &patterns{}
		patterns.add(pattern)
		p.m[initial] = patterns

	}
}

func (p initialPatterns) get(initial byte) (*patterns, bool) {
	patterns, ok := p.m[initial]
	return patterns, ok
}

func (p initialPatterns) size() int {
	return len(p.m)
}
