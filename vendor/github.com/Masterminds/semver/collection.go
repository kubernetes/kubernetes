package semver

// Collection is a collection of Version instances and implements the sort
// interface. See the sort package for more details.
// https://golang.org/pkg/sort/
type Collection []Version

// Len returns the length of a collection. The number of Version instances
// on the slice.
func (c Collection) Len() int {
	return len(c)
}

// Less is needed for the sort interface to compare two Version objects on the
// slice. If checks if one is less than the other.
func (c Collection) Less(i, j int) bool {
	return c[i].LessThan(c[j])
}

// Swap is needed for the sort interface to replace the Version objects
// at two different positions in the slice.
func (c Collection) Swap(i, j int) {
	c[i], c[j] = c[j], c[i]
}
