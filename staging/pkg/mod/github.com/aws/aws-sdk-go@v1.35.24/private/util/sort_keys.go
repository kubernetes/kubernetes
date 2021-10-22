package util

import "sort"

// SortedKeys returns a sorted slice of keys of a map.
func SortedKeys(m map[string]interface{}) []string {
	i, sorted := 0, make([]string, len(m))
	for k := range m {
		sorted[i] = k
		i++
	}
	sort.Strings(sorted)
	return sorted
}
