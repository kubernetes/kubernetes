// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

// TriKind represents the triangularity of the matrix.
type TriKind bool

const (
	// Upper specifies an upper triangular matrix.
	Upper TriKind = true
	// Lower specifies a lower triangular matrix.
	Lower TriKind = false
)
