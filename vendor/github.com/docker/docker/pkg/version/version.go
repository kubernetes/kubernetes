package version

import (
	"strconv"
	"strings"
)

// Version provides utility methods for comparing versions.
type Version string

func (v Version) compareTo(other Version) int {
	var (
		currTab  = strings.Split(string(v), ".")
		otherTab = strings.Split(string(other), ".")
	)

	max := len(currTab)
	if len(otherTab) > max {
		max = len(otherTab)
	}
	for i := 0; i < max; i++ {
		var currInt, otherInt int

		if len(currTab) > i {
			currInt, _ = strconv.Atoi(currTab[i])
		}
		if len(otherTab) > i {
			otherInt, _ = strconv.Atoi(otherTab[i])
		}
		if currInt > otherInt {
			return 1
		}
		if otherInt > currInt {
			return -1
		}
	}
	return 0
}

// LessThan checks if a version is less than another
func (v Version) LessThan(other Version) bool {
	return v.compareTo(other) == -1
}

// LessThanOrEqualTo checks if a version is less than or equal to another
func (v Version) LessThanOrEqualTo(other Version) bool {
	return v.compareTo(other) <= 0
}

// GreaterThan checks if a version is greater than another
func (v Version) GreaterThan(other Version) bool {
	return v.compareTo(other) == 1
}

// GreaterThanOrEqualTo checks if a version is greater than or equal to another
func (v Version) GreaterThanOrEqualTo(other Version) bool {
	return v.compareTo(other) >= 0
}

// Equal checks if a version is equal to another
func (v Version) Equal(other Version) bool {
	return v.compareTo(other) == 0
}
