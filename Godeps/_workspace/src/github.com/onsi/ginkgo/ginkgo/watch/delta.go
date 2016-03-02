package watch

import "sort"

type Delta struct {
	ModifiedPackages []string

	NewSuites      []*Suite
	RemovedSuites  []*Suite
	modifiedSuites []*Suite
}

type DescendingByDelta []*Suite

func (a DescendingByDelta) Len() int           { return len(a) }
func (a DescendingByDelta) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a DescendingByDelta) Less(i, j int) bool { return a[i].Delta() > a[j].Delta() }

func (d Delta) ModifiedSuites() []*Suite {
	sort.Sort(DescendingByDelta(d.modifiedSuites))
	return d.modifiedSuites
}
