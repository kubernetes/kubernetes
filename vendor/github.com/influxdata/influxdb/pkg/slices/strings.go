package slices // import "github.com/influxdata/influxdb/pkg/slices"

import "strings"

// Union combines two string sets
func Union(setA, setB []string, ignoreCase bool) []string {
	for _, b := range setB {
		if ignoreCase {
			if !ExistsIgnoreCase(setA, b) {
				setA = append(setA, b)
			}
			continue
		}
		if !Exists(setA, b) {
			setA = append(setA, b)
		}
	}
	return setA
}

// Exists checks if a string is in a set
func Exists(set []string, find string) bool {
	for _, s := range set {
		if s == find {
			return true
		}
	}
	return false
}

// ExistsIgnoreCase checks if a string is in a set but ignores its case
func ExistsIgnoreCase(set []string, find string) bool {
	find = strings.ToLower(find)
	for _, s := range set {
		if strings.ToLower(s) == find {
			return true
		}
	}
	return false
}
