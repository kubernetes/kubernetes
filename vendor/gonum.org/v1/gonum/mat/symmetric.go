// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

var (
	symDense *SymDense

	_ Matrix           = symDense
	_ Symmetric        = symDense
	_ RawSymmetricer   = symDense
	_ MutableSymmetric = symDense
)

const (
	badSymTriangle = "mat: blas64.Symmetric not upper"
	badSymCap      = "mat: bad capacity for SymDense"
)

// SymDense is a symmetric matrix that uses dense storage. SymDense
// matrices are stored in the upper triangle.
type SymDense struct {
	mat blas64.Symmetric
	cap int
}

// Symmetric represents a symmetric matrix (where the element at {i, j} equals
// the element at {j, i}). Symmetric matrices are always square.
type Symmetric interface {
	Matrix
	// Symmetric returns the number of rows/columns in the matrix.
	Symmetric() int
}

// A RawSymmetricer can return a view of itself as a BLAS Symmetric matrix.
type RawSymmetricer interface {
	RawSymmetric() blas64.Symmetric
}

// A MutableSymmetric can set elements of a symmetric matrix.
type MutableSymmetric interface {
	Symmetric
	SetSym(i, j int, v float64)
}

// NewSymDense creates a new Symmetric matrix with n rows and columns. If data == nil,
// a new slice is allocated for the backing slice. If len(data) == n*n, data is
// used as the backing slice, and changes to the elements of the returned SymDense
// will be reflected in data. If neither of these is true, NewSymDense will panic.
//
// The data must be arranged in row-major order, i.e. the (i*c + j)-th
// element in the data slice is the {i, j}-th element in the matrix.
// Only the values in the upper triangular portion of the matrix are used.
func NewSymDense(n int, data []float64) *SymDense {
	if n < 0 {
		panic("mat: negative dimension")
	}
	if data != nil && n*n != len(data) {
		panic(ErrShape)
	}
	if data == nil {
		data = make([]float64, n*n)
	}
	return &SymDense{
		mat: blas64.Symmetric{
			N:      n,
			Stride: n,
			Data:   data,
			Uplo:   blas.Upper,
		},
		cap: n,
	}
}

// Dims returns the number of rows and columns in the matrix.
func (s *SymDense) Dims() (r, c int) {
	return s.mat.N, s.mat.N
}

// Caps returns the number of rows and columns in the backing matrix.
func (s *SymDense) Caps() (r, c int) {
	return s.cap, s.cap
}

// T implements the Matrix interface. Symmetric matrices, by definition, are
// equal to their transpose, and this is a no-op.
func (s *SymDense) T() Matrix {
	return s
}

func (s *SymDense) Symmetric() int {
	return s.mat.N
}

// RawSymmetric returns the matrix as a blas64.Symmetric. The returned
// value must be stored in upper triangular format.
func (s *SymDense) RawSymmetric() blas64.Symmetric {
	return s.mat
}

// SetRawSymmetric sets the underlying blas64.Symmetric used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in b. SetRawSymmetric will panic if b is not an upper-encoded symmetric
// matrix.
func (s *SymDense) SetRawSymmetric(b blas64.Symmetric) {
	if b.Uplo != blas.Upper {
		panic(badSymTriangle)
	}
	s.mat = b
}

// Reset zeros the dimensions of the matrix so that it can be reused as the
// receiver of a dimensionally restricted operation.
//
// See the Reseter interface for more information.
func (s *SymDense) Reset() {
	// N and Stride must be zeroed in unison.
	s.mat.N, s.mat.Stride = 0, 0
	s.mat.Data = s.mat.Data[:0]
}

// IsZero returns whether the receiver is zero-sized. Zero-sized matrices can be the
// receiver for size-restricted operations. SymDense matrices can be zeroed using Reset.
func (s *SymDense) IsZero() bool {
	// It must be the case that m.Dims() returns
	// zeros in this case. See comment in Reset().
	return s.mat.N == 0
}

// reuseAs resizes an empty matrix to a n×n matrix,
// or checks that a non-empty matrix is n×n.
func (s *SymDense) reuseAs(n int) {
	if n == 0 {
		panic(ErrZeroLength)
	}
	if s.mat.N > s.cap {
		panic(badSymCap)
	}
	if s.IsZero() {
		s.mat = blas64.Symmetric{
			N:      n,
			Stride: n,
			Data:   use(s.mat.Data, n*n),
			Uplo:   blas.Upper,
		}
		s.cap = n
		return
	}
	if s.mat.Uplo != blas.Upper {
		panic(badSymTriangle)
	}
	if s.mat.N != n {
		panic(ErrShape)
	}
}

func (s *SymDense) isolatedWorkspace(a Symmetric) (w *SymDense, restore func()) {
	n := a.Symmetric()
	if n == 0 {
		panic(ErrZeroLength)
	}
	w = getWorkspaceSym(n, false)
	return w, func() {
		s.CopySym(w)
		putWorkspaceSym(w)
	}
}

func (s *SymDense) AddSym(a, b Symmetric) {
	n := a.Symmetric()
	if n != b.Symmetric() {
		panic(ErrShape)
	}
	s.reuseAs(n)

	if a, ok := a.(RawSymmetricer); ok {
		if b, ok := b.(RawSymmetricer); ok {
			amat, bmat := a.RawSymmetric(), b.RawSymmetric()
			if s != a {
				s.checkOverlap(generalFromSymmetric(amat))
			}
			if s != b {
				s.checkOverlap(generalFromSymmetric(bmat))
			}
			for i := 0; i < n; i++ {
				btmp := bmat.Data[i*bmat.Stride+i : i*bmat.Stride+n]
				stmp := s.mat.Data[i*s.mat.Stride+i : i*s.mat.Stride+n]
				for j, v := range amat.Data[i*amat.Stride+i : i*amat.Stride+n] {
					stmp[j] = v + btmp[j]
				}
			}
			return
		}
	}

	s.checkOverlapMatrix(a)
	s.checkOverlapMatrix(b)
	for i := 0; i < n; i++ {
		stmp := s.mat.Data[i*s.mat.Stride : i*s.mat.Stride+n]
		for j := i; j < n; j++ {
			stmp[j] = a.At(i, j) + b.At(i, j)
		}
	}
}

func (s *SymDense) CopySym(a Symmetric) int {
	n := a.Symmetric()
	n = min(n, s.mat.N)
	if n == 0 {
		return 0
	}
	switch a := a.(type) {
	case RawSymmetricer:
		amat := a.RawSymmetric()
		if amat.Uplo != blas.Upper {
			panic(badSymTriangle)
		}
		for i := 0; i < n; i++ {
			copy(s.mat.Data[i*s.mat.Stride+i:i*s.mat.Stride+n], amat.Data[i*amat.Stride+i:i*amat.Stride+n])
		}
	default:
		for i := 0; i < n; i++ {
			stmp := s.mat.Data[i*s.mat.Stride : i*s.mat.Stride+n]
			for j := i; j < n; j++ {
				stmp[j] = a.At(i, j)
			}
		}
	}
	return n
}

// SymRankOne performs a symetric rank-one update to the matrix a and stores
// the result in the receiver
//  s = a + alpha * x * x'
func (s *SymDense) SymRankOne(a Symmetric, alpha float64, x Vector) {
	n, c := x.Dims()
	if a.Symmetric() != n || c != 1 {
		panic(ErrShape)
	}
	s.reuseAs(n)

	if s != a {
		if rs, ok := a.(RawSymmetricer); ok {
			s.checkOverlap(generalFromSymmetric(rs.RawSymmetric()))
		}
		s.CopySym(a)
	}

	xU, _ := untranspose(x)
	if rv, ok := xU.(RawVectorer); ok {
		xmat := rv.RawVector()
		s.checkOverlap((&VecDense{mat: xmat, n: n}).asGeneral())
		blas64.Syr(alpha, xmat, s.mat)
		return
	}

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			s.set(i, j, s.at(i, j)+alpha*x.AtVec(i)*x.AtVec(j))
		}
	}
}

// SymRankK performs a symmetric rank-k update to the matrix a and stores the
// result into the receiver. If a is zero, see SymOuterK.
//  s = a + alpha * x * x'
func (s *SymDense) SymRankK(a Symmetric, alpha float64, x Matrix) {
	n := a.Symmetric()
	r, _ := x.Dims()
	if r != n {
		panic(ErrShape)
	}
	xMat, aTrans := untranspose(x)
	var g blas64.General
	if rm, ok := xMat.(RawMatrixer); ok {
		g = rm.RawMatrix()
	} else {
		g = DenseCopyOf(x).mat
		aTrans = false
	}
	if a != s {
		if rs, ok := a.(RawSymmetricer); ok {
			s.checkOverlap(generalFromSymmetric(rs.RawSymmetric()))
		}
		s.reuseAs(n)
		s.CopySym(a)
	}
	t := blas.NoTrans
	if aTrans {
		t = blas.Trans
	}
	blas64.Syrk(t, alpha, g, 1, s.mat)
}

// SymOuterK calculates the outer product of x with itself and stores
// the result into the receiver. It is equivalent to the matrix
// multiplication
//  s = alpha * x * x'.
// In order to update an existing matrix, see SymRankOne.
func (s *SymDense) SymOuterK(alpha float64, x Matrix) {
	n, _ := x.Dims()
	switch {
	case s.IsZero():
		s.mat = blas64.Symmetric{
			N:      n,
			Stride: n,
			Data:   useZeroed(s.mat.Data, n*n),
			Uplo:   blas.Upper,
		}
		s.cap = n
		s.SymRankK(s, alpha, x)
	case s.mat.Uplo != blas.Upper:
		panic(badSymTriangle)
	case s.mat.N == n:
		if s == x {
			w := getWorkspaceSym(n, true)
			w.SymRankK(w, alpha, x)
			s.CopySym(w)
			putWorkspaceSym(w)
		} else {
			switch r := x.(type) {
			case RawMatrixer:
				s.checkOverlap(r.RawMatrix())
			case RawSymmetricer:
				s.checkOverlap(generalFromSymmetric(r.RawSymmetric()))
			case RawTriangular:
				s.checkOverlap(generalFromTriangular(r.RawTriangular()))
			}
			// Only zero the upper triangle.
			for i := 0; i < n; i++ {
				ri := i * s.mat.Stride
				zero(s.mat.Data[ri+i : ri+n])
			}
			s.SymRankK(s, alpha, x)
		}
	default:
		panic(ErrShape)
	}
}

// RankTwo performs a symmmetric rank-two update to the matrix a and stores
// the result in the receiver
//  m = a + alpha * (x * y' + y * x')
func (s *SymDense) RankTwo(a Symmetric, alpha float64, x, y Vector) {
	n := s.mat.N
	xr, xc := x.Dims()
	if xr != n || xc != 1 {
		panic(ErrShape)
	}
	yr, yc := y.Dims()
	if yr != n || yc != 1 {
		panic(ErrShape)
	}

	if s != a {
		if rs, ok := a.(RawSymmetricer); ok {
			s.checkOverlap(generalFromSymmetric(rs.RawSymmetric()))
		}
	}

	var xmat, ymat blas64.Vector
	fast := true
	xU, _ := untranspose(x)
	if rv, ok := xU.(RawVectorer); ok {
		xmat = rv.RawVector()
		s.checkOverlap((&VecDense{mat: xmat, n: x.Len()}).asGeneral())
	} else {
		fast = false
	}
	yU, _ := untranspose(y)
	if rv, ok := yU.(RawVectorer); ok {
		ymat = rv.RawVector()
		s.checkOverlap((&VecDense{mat: ymat, n: y.Len()}).asGeneral())
	} else {
		fast = false
	}

	if s != a {
		if rs, ok := a.(RawSymmetricer); ok {
			s.checkOverlap(generalFromSymmetric(rs.RawSymmetric()))
		}
		s.reuseAs(n)
		s.CopySym(a)
	}

	if fast {
		if s != a {
			s.reuseAs(n)
			s.CopySym(a)
		}
		blas64.Syr2(alpha, xmat, ymat, s.mat)
		return
	}

	for i := 0; i < n; i++ {
		s.reuseAs(n)
		for j := i; j < n; j++ {
			s.set(i, j, a.At(i, j)+alpha*(x.AtVec(i)*y.AtVec(j)+y.AtVec(i)*x.AtVec(j)))
		}
	}
}

// ScaleSym multiplies the elements of a by f, placing the result in the receiver.
func (s *SymDense) ScaleSym(f float64, a Symmetric) {
	n := a.Symmetric()
	s.reuseAs(n)
	if a, ok := a.(RawSymmetricer); ok {
		amat := a.RawSymmetric()
		if s != a {
			s.checkOverlap(generalFromSymmetric(amat))
		}
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				s.mat.Data[i*s.mat.Stride+j] = f * amat.Data[i*amat.Stride+j]
			}
		}
		return
	}
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			s.mat.Data[i*s.mat.Stride+j] = f * a.At(i, j)
		}
	}
}

// SubsetSym extracts a subset of the rows and columns of the matrix a and stores
// the result in-place into the receiver. The resulting matrix size is
// len(set)×len(set). Specifically, at the conclusion of SubsetSym,
// s.At(i, j) equals a.At(set[i], set[j]). Note that the supplied set does not
// have to be a strict subset, dimension repeats are allowed.
func (s *SymDense) SubsetSym(a Symmetric, set []int) {
	n := len(set)
	na := a.Symmetric()
	s.reuseAs(n)
	var restore func()
	if a == s {
		s, restore = s.isolatedWorkspace(a)
		defer restore()
	}

	if a, ok := a.(RawSymmetricer); ok {
		raw := a.RawSymmetric()
		if s != a {
			s.checkOverlap(generalFromSymmetric(raw))
		}
		for i := 0; i < n; i++ {
			ssub := s.mat.Data[i*s.mat.Stride : i*s.mat.Stride+n]
			r := set[i]
			rsub := raw.Data[r*raw.Stride : r*raw.Stride+na]
			for j := i; j < n; j++ {
				c := set[j]
				if r <= c {
					ssub[j] = rsub[c]
				} else {
					ssub[j] = raw.Data[c*raw.Stride+r]
				}
			}
		}
		return
	}
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			s.mat.Data[i*s.mat.Stride+j] = a.At(set[i], set[j])
		}
	}
}

// SliceSquare returns a new Matrix that shares backing data with the receiver.
// The returned matrix starts at {i,i} of the receiver and extends k-i rows
// and columns. The final row and column in the resulting matrix is k-1.
// SliceSquare panics with ErrIndexOutOfRange if the slice is outside the capacity
// of the receiver.
func (s *SymDense) SliceSquare(i, k int) Matrix {
	sz := s.cap
	if i < 0 || sz < i || k < i || sz < k {
		panic(ErrIndexOutOfRange)
	}
	v := *s
	v.mat.Data = s.mat.Data[i*s.mat.Stride+i : (k-1)*s.mat.Stride+k]
	v.mat.N = k - i
	v.cap = s.cap - i
	return &v
}

// GrowSquare returns the receiver expanded by n rows and n columns. If the
// dimensions of the expanded matrix are outside the capacity of the receiver
// a new allocation is made, otherwise not. Note that the receiver itself is
// not modified during the call to GrowSquare.
func (s *SymDense) GrowSquare(n int) Matrix {
	if n < 0 {
		panic(ErrIndexOutOfRange)
	}
	if n == 0 {
		return s
	}
	var v SymDense
	n += s.mat.N
	if n > s.cap {
		v.mat = blas64.Symmetric{
			N:      n,
			Stride: n,
			Uplo:   blas.Upper,
			Data:   make([]float64, n*n),
		}
		v.cap = n
		// Copy elements, including those not currently visible. Use a temporary
		// structure to avoid modifying the receiver.
		var tmp SymDense
		tmp.mat = blas64.Symmetric{
			N:      s.cap,
			Stride: s.mat.Stride,
			Data:   s.mat.Data,
			Uplo:   s.mat.Uplo,
		}
		tmp.cap = s.cap
		v.CopySym(&tmp)
		return &v
	}
	v.mat = blas64.Symmetric{
		N:      n,
		Stride: s.mat.Stride,
		Uplo:   blas.Upper,
		Data:   s.mat.Data[:(n-1)*s.mat.Stride+n],
	}
	v.cap = s.cap
	return &v
}

// PowPSD computes a^pow where a is a positive symmetric definite matrix.
//
// PowPSD returns an error if the matrix is not  not positive symmetric definite
// or the Eigendecomposition is not successful.
func (s *SymDense) PowPSD(a Symmetric, pow float64) error {
	dim := a.Symmetric()
	s.reuseAs(dim)

	var eigen EigenSym
	ok := eigen.Factorize(a, true)
	if !ok {
		return ErrFailedEigen
	}
	values := eigen.Values(nil)
	for i, v := range values {
		if v <= 0 {
			return ErrNotPSD
		}
		values[i] = math.Pow(v, pow)
	}
	var u Dense
	u.EigenvectorsSym(&eigen)

	s.SymOuterK(values[0], u.ColView(0))

	var v VecDense
	for i := 1; i < dim; i++ {
		v.ColViewOf(&u, i)
		s.SymRankOne(s, values[i], &v)
	}
	return nil
}
