package internal

import "golang.org/x/exp/constraints"

// Align returns 'n' updated to 'alignment' boundary.
func Align[I constraints.Integer](n, alignment I) I {
	return (n + alignment - 1) / alignment * alignment
}
