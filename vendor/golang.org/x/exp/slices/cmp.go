// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slices

import "golang.org/x/exp/constraints"

// min is a version of the predeclared function from the Go 1.21 release.
func min[T constraints.Ordered](a, b T) T {
	if a < b || isNaN(a) {
		return a
	}
	return b
}

// max is a version of the predeclared function from the Go 1.21 release.
func max[T constraints.Ordered](a, b T) T {
	if a > b || isNaN(a) {
		return a
	}
	return b
}

// cmpLess is a copy of cmp.Less from the Go 1.21 release.
func cmpLess[T constraints.Ordered](x, y T) bool {
	return (isNaN(x) && !isNaN(y)) || x < y
}

// cmpCompare is a copy of cmp.Compare from the Go 1.21 release.
func cmpCompare[T constraints.Ordered](x, y T) int {
	xNaN := isNaN(x)
	yNaN := isNaN(y)
	if xNaN && yNaN {
		return 0
	}
	if xNaN || x < y {
		return -1
	}
	if yNaN || x > y {
		return +1
	}
	return 0
}
