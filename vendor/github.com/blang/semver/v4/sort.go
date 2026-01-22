package semver

import (
	"sort"
)

// Versions represents multiple versions.
type Versions []Version

// Len returns length of version collection
func (s Versions) Len() int {
	return len(s)
}

// Swap swaps two versions inside the collection by its indices
func (s Versions) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Less checks if version at index i is less than version at index j
func (s Versions) Less(i, j int) bool {
	return s[i].LT(s[j])
}

// Sort sorts a slice of versions
func Sort(versions []Version) {
	sort.Sort(Versions(versions))
}
