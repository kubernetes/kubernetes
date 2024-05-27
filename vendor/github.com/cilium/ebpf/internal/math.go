package internal

import "golang.org/x/exp/constraints"

// Align returns 'n' updated to 'alignment' boundary.
func Align[I constraints.Integer](n, alignment I) I {
	return (n + alignment - 1) / alignment * alignment
}

// IsPow returns true if n is a power of two.
func IsPow[I constraints.Integer](n I) bool {
	return n != 0 && (n&(n-1)) == 0
}
