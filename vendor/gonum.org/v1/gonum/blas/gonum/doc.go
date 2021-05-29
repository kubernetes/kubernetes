// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure changes made to blas/native are reflected in blas/cgo where relevant.

/*
Package gonum is a Go implementation of the BLAS API. This implementation
panics when the input arguments are invalid as per the standard, for example
if a vector increment is zero. Note that the treatment of NaN values
is not specified, and differs among the BLAS implementations.
gonum.org/v1/gonum/blas/blas64 provides helpful wrapper functions to the BLAS
interface. The rest of this text describes the layout of the data for the input types.

Note that in the function documentation, x[i] refers to the i^th element
of the vector, which will be different from the i^th element of the slice if
incX != 1.

See http://www.netlib.org/lapack/explore-html/d4/de1/_l_i_c_e_n_s_e_source.html
for more license information.

Vector arguments are effectively strided slices. They have two input arguments,
a number of elements, n, and an increment, incX. The increment specifies the
distance between elements of the vector. The actual Go slice may be longer
than necessary.
The increment may be positive or negative, except in functions with only
a single vector argument where the increment may only be positive. If the increment
is negative, s[0] is the last element in the slice. Note that this is not the same
as counting backward from the end of the slice, as len(s) may be longer than
necessary. So, for example, if n = 5 and incX = 3, the elements of s are
	[0 * * 1 * * 2 * * 3 * * 4 * * * ...]
where ∗ elements are never accessed. If incX = -3, the same elements are
accessed, just in reverse order (4, 3, 2, 1, 0).

Dense matrices are specified by a number of rows, a number of columns, and a stride.
The stride specifies the number of entries in the slice between the first element
of successive rows. The stride must be at least as large as the number of columns
but may be longer.
	[a00 ... a0n a0* ... a1stride-1 a21 ... amn am* ... amstride-1]
Thus, dense[i*ld + j] refers to the {i, j}th element of the matrix.

Symmetric and triangular matrices (non-packed) are stored identically to Dense,
except that only elements in one triangle of the matrix are accessed.

Packed symmetric and packed triangular matrices are laid out with the entries
condensed such that all of the unreferenced elements are removed. So, the upper triangular
matrix
  [
    1  2  3
    0  4  5
    0  0  6
  ]
and the lower-triangular matrix
  [
    1  0  0
    2  3  0
    4  5  6
  ]
will both be compacted as [1 2 3 4 5 6]. The (i, j) element of the original
dense matrix can be found at element i*n - (i-1)*i/2 + j for upper triangular,
and at element i * (i+1) /2 + j for lower triangular.

Banded matrices are laid out in a compact format, constructed by removing the
zeros in the rows and aligning the diagonals. For example, the matrix
  [
    1  2  3  0  0  0
    4  5  6  7  0  0
    0  8  9 10 11  0
    0  0 12 13 14 15
    0  0  0 16 17 18
    0  0  0  0 19 20
  ]

implicitly becomes (∗ entries are never accessed)
  [
     *  1  2  3
     4  5  6  7
     8  9 10 11
    12 13 14 15
    16 17 18  *
    19 20  *  *
  ]
which is given to the BLAS routine as [∗ 1 2 3 4 ...].

See http://www.crest.iu.edu/research/mtl/reference/html/banded.html
for more information
*/
package gonum // import "gonum.org/v1/gonum/blas/gonum"
