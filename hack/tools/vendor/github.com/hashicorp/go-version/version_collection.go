package version

// Collection is a type that implements the sort.Interface interface
// so that versions can be sorted.
type Collection []*Version

func (v Collection) Len() int {
	return len(v)
}

func (v Collection) Less(i, j int) bool {
	return v[i].LessThan(v[j])
}

func (v Collection) Swap(i, j int) {
	v[i], v[j] = v[j], v[i]
}
