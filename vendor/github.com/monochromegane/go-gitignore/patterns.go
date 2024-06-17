package gitignore

type patterns struct {
	patterns []pattern
}

func (ps *patterns) add(pattern pattern) {
	ps.patterns = append(ps.patterns, pattern)
}

func (ps *patterns) size() int {
	return len(ps.patterns)
}

func (ps patterns) match(path string, isDir bool) bool {
	for _, p := range ps.patterns {
		if match := p.match(path, isDir); match {
			return true
		}
	}
	return false
}
