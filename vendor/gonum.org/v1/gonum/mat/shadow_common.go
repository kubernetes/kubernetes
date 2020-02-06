// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

const (
	// regionOverlap is the panic string used for the general case
	// of a matrix region overlap between a source and destination.
	regionOverlap = "mat: bad region: overlap"

	// regionIdentity is the panic string used for the specific
	// case of complete agreement between a source and a destination.
	regionIdentity = "mat: bad region: identical"

	// mismatchedStrides is the panic string used for overlapping
	// data slices with differing strides.
	mismatchedStrides = "mat: bad region: different strides"
)

// rectanglesOverlap returns whether the strided rectangles a and b overlap
// when b is offset by off elements after a but has at least one element before
// the end of a. off must be positive. a and b have aCols and bCols respectively.
//
// rectanglesOverlap works by shifting both matrices left such that the left
// column of a is at 0. The column indexes are flattened by obtaining the shifted
// relative left and right column positions modulo the common stride. This allows
// direct comparison of the column offsets when the matrix backing data slices
// are known to overlap.
func rectanglesOverlap(off, aCols, bCols, stride int) bool {
	if stride == 1 {
		// Unit stride means overlapping data
		// slices must overlap as matrices.
		return true
	}

	// Flatten the shifted matrix column positions
	// so a starts at 0, modulo the common stride.
	aTo := aCols
	// The mod stride operations here make the from
	// and to indexes comparable between a and b when
	// the data slices of a and b overlap.
	bFrom := off % stride
	bTo := (bFrom + bCols) % stride

	if bTo == 0 || bFrom < bTo {
		// b matrix is not wrapped: compare for
		// simple overlap.
		return bFrom < aTo
	}

	// b strictly wraps and so must overlap with a.
	return true
}
