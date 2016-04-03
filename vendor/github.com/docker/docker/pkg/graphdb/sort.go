package graphdb

import "sort"

type pathSorter struct {
	paths []string
	by    func(i, j string) bool
}

func sortByDepth(paths []string) {
	s := &pathSorter{paths, func(i, j string) bool {
		return PathDepth(i) > PathDepth(j)
	}}
	sort.Sort(s)
}

func (s *pathSorter) Len() int {
	return len(s.paths)
}

func (s *pathSorter) Swap(i, j int) {
	s.paths[i], s.paths[j] = s.paths[j], s.paths[i]
}

func (s *pathSorter) Less(i, j int) bool {
	return s.by(s.paths[i], s.paths[j])
}
