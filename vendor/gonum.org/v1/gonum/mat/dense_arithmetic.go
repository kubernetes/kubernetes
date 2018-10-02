// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack/lapack64"
)

// Add adds a and b element-wise, placing the result in the receiver. Add
// will panic if the two matrices do not have the same shape.
func (m *Dense) Add(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		panic(ErrShape)
	}

	aU, _ := untranspose(a)
	bU, _ := untranspose(b)
	m.reuseAs(ar, ac)

	if arm, ok := a.(RawMatrixer); ok {
		if brm, ok := b.(RawMatrixer); ok {
			amat, bmat := arm.RawMatrix(), brm.RawMatrix()
			if m != aU {
				m.checkOverlap(amat)
			}
			if m != bU {
				m.checkOverlap(bmat)
			}
			for ja, jb, jm := 0, 0, 0; ja < ar*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v + bmat.Data[i+jb]
				}
			}
			return
		}
	}

	m.checkOverlapMatrix(aU)
	m.checkOverlapMatrix(bU)
	var restore func()
	if m == aU {
		m, restore = m.isolatedWorkspace(aU)
		defer restore()
	} else if m == bU {
		m, restore = m.isolatedWorkspace(bU)
		defer restore()
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, a.At(r, c)+b.At(r, c))
		}
	}
}

// Sub subtracts the matrix b from a, placing the result in the receiver. Sub
// will panic if the two matrices do not have the same shape.
func (m *Dense) Sub(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		panic(ErrShape)
	}

	aU, _ := untranspose(a)
	bU, _ := untranspose(b)
	m.reuseAs(ar, ac)

	if arm, ok := a.(RawMatrixer); ok {
		if brm, ok := b.(RawMatrixer); ok {
			amat, bmat := arm.RawMatrix(), brm.RawMatrix()
			if m != aU {
				m.checkOverlap(amat)
			}
			if m != bU {
				m.checkOverlap(bmat)
			}
			for ja, jb, jm := 0, 0, 0; ja < ar*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v - bmat.Data[i+jb]
				}
			}
			return
		}
	}

	m.checkOverlapMatrix(aU)
	m.checkOverlapMatrix(bU)
	var restore func()
	if m == aU {
		m, restore = m.isolatedWorkspace(aU)
		defer restore()
	} else if m == bU {
		m, restore = m.isolatedWorkspace(bU)
		defer restore()
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, a.At(r, c)-b.At(r, c))
		}
	}
}

// MulElem performs element-wise multiplication of a and b, placing the result
// in the receiver. MulElem will panic if the two matrices do not have the same
// shape.
func (m *Dense) MulElem(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		panic(ErrShape)
	}

	aU, _ := untranspose(a)
	bU, _ := untranspose(b)
	m.reuseAs(ar, ac)

	if arm, ok := a.(RawMatrixer); ok {
		if brm, ok := b.(RawMatrixer); ok {
			amat, bmat := arm.RawMatrix(), brm.RawMatrix()
			if m != aU {
				m.checkOverlap(amat)
			}
			if m != bU {
				m.checkOverlap(bmat)
			}
			for ja, jb, jm := 0, 0, 0; ja < ar*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v * bmat.Data[i+jb]
				}
			}
			return
		}
	}

	m.checkOverlapMatrix(aU)
	m.checkOverlapMatrix(bU)
	var restore func()
	if m == aU {
		m, restore = m.isolatedWorkspace(aU)
		defer restore()
	} else if m == bU {
		m, restore = m.isolatedWorkspace(bU)
		defer restore()
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, a.At(r, c)*b.At(r, c))
		}
	}
}

// DivElem performs element-wise division of a by b, placing the result
// in the receiver. DivElem will panic if the two matrices do not have the same
// shape.
func (m *Dense) DivElem(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		panic(ErrShape)
	}

	aU, _ := untranspose(a)
	bU, _ := untranspose(b)
	m.reuseAs(ar, ac)

	if arm, ok := a.(RawMatrixer); ok {
		if brm, ok := b.(RawMatrixer); ok {
			amat, bmat := arm.RawMatrix(), brm.RawMatrix()
			if m != aU {
				m.checkOverlap(amat)
			}
			if m != bU {
				m.checkOverlap(bmat)
			}
			for ja, jb, jm := 0, 0, 0; ja < ar*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v / bmat.Data[i+jb]
				}
			}
			return
		}
	}

	m.checkOverlapMatrix(aU)
	m.checkOverlapMatrix(bU)
	var restore func()
	if m == aU {
		m, restore = m.isolatedWorkspace(aU)
		defer restore()
	} else if m == bU {
		m, restore = m.isolatedWorkspace(bU)
		defer restore()
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, a.At(r, c)/b.At(r, c))
		}
	}
}

// Inverse computes the inverse of the matrix a, storing the result into the
// receiver. If a is ill-conditioned, a Condition error will be returned.
// Note that matrix inversion is numerically unstable, and should generally
// be avoided where possible, for example by using the Solve routines.
func (m *Dense) Inverse(a Matrix) error {
	// TODO(btracey): Special case for RawTriangular, etc.
	r, c := a.Dims()
	if r != c {
		panic(ErrSquare)
	}
	m.reuseAs(a.Dims())
	aU, aTrans := untranspose(a)
	switch rm := aU.(type) {
	case RawMatrixer:
		if m != aU || aTrans {
			if m == aU || m.checkOverlap(rm.RawMatrix()) {
				tmp := getWorkspace(r, c, false)
				tmp.Copy(a)
				m.Copy(tmp)
				putWorkspace(tmp)
				break
			}
			m.Copy(a)
		}
	default:
		m.Copy(a)
	}
	ipiv := getInts(r, false)
	defer putInts(ipiv)
	ok := lapack64.Getrf(m.mat, ipiv)
	if !ok {
		return Condition(math.Inf(1))
	}
	work := getFloats(4*r, false) // must be at least 4*r for cond.
	lapack64.Getri(m.mat, ipiv, work, -1)
	if int(work[0]) > 4*r {
		l := int(work[0])
		putFloats(work)
		work = getFloats(l, false)
	} else {
		work = work[:4*r]
	}
	defer putFloats(work)
	lapack64.Getri(m.mat, ipiv, work, len(work))
	norm := lapack64.Lange(CondNorm, m.mat, work)
	rcond := lapack64.Gecon(CondNorm, m.mat, norm, work, ipiv) // reuse ipiv
	if rcond == 0 {
		return Condition(math.Inf(1))
	}
	cond := 1 / rcond
	if cond > ConditionTolerance {
		return Condition(cond)
	}
	return nil
}

// Mul takes the matrix product of a and b, placing the result in the receiver.
// If the number of columns in a does not equal the number of rows in b, Mul will panic.
func (m *Dense) Mul(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ac != br {
		panic(ErrShape)
	}

	aU, aTrans := untranspose(a)
	bU, bTrans := untranspose(b)
	m.reuseAs(ar, bc)
	var restore func()
	if m == aU {
		m, restore = m.isolatedWorkspace(aU)
		defer restore()
	} else if m == bU {
		m, restore = m.isolatedWorkspace(bU)
		defer restore()
	}
	aT := blas.NoTrans
	if aTrans {
		aT = blas.Trans
	}
	bT := blas.NoTrans
	if bTrans {
		bT = blas.Trans
	}

	// Some of the cases do not have a transpose option, so create
	// temporary memory.
	// C = A^T * B = (B^T * A)^T
	// C^T = B^T * A.
	if aUrm, ok := aU.(RawMatrixer); ok {
		amat := aUrm.RawMatrix()
		if restore == nil {
			m.checkOverlap(amat)
		}
		if bUrm, ok := bU.(RawMatrixer); ok {
			bmat := bUrm.RawMatrix()
			if restore == nil {
				m.checkOverlap(bmat)
			}
			blas64.Gemm(aT, bT, 1, amat, bmat, 0, m.mat)
			return
		}
		if bU, ok := bU.(RawSymmetricer); ok {
			bmat := bU.RawSymmetric()
			if aTrans {
				c := getWorkspace(ac, ar, false)
				blas64.Symm(blas.Left, 1, bmat, amat, 0, c.mat)
				strictCopy(m, c.T())
				putWorkspace(c)
				return
			}
			blas64.Symm(blas.Right, 1, bmat, amat, 0, m.mat)
			return
		}
		if bU, ok := bU.(RawTriangular); ok {
			// Trmm updates in place, so copy aU first.
			bmat := bU.RawTriangular()
			if aTrans {
				c := getWorkspace(ac, ar, false)
				var tmp Dense
				tmp.SetRawMatrix(amat)
				c.Copy(&tmp)
				bT := blas.Trans
				if bTrans {
					bT = blas.NoTrans
				}
				blas64.Trmm(blas.Left, bT, 1, bmat, c.mat)
				strictCopy(m, c.T())
				putWorkspace(c)
				return
			}
			m.Copy(a)
			blas64.Trmm(blas.Right, bT, 1, bmat, m.mat)
			return
		}
		if bU, ok := bU.(*VecDense); ok {
			m.checkOverlap(bU.asGeneral())
			bvec := bU.RawVector()
			if bTrans {
				// {ar,1} x {1,bc}, which is not a vector.
				// Instead, construct B as a General.
				bmat := blas64.General{
					Rows:   bc,
					Cols:   1,
					Stride: bvec.Inc,
					Data:   bvec.Data,
				}
				blas64.Gemm(aT, bT, 1, amat, bmat, 0, m.mat)
				return
			}
			cvec := blas64.Vector{
				Inc:  m.mat.Stride,
				Data: m.mat.Data,
			}
			blas64.Gemv(aT, 1, amat, bvec, 0, cvec)
			return
		}
	}
	if bUrm, ok := bU.(RawMatrixer); ok {
		bmat := bUrm.RawMatrix()
		if restore == nil {
			m.checkOverlap(bmat)
		}
		if aU, ok := aU.(RawSymmetricer); ok {
			amat := aU.RawSymmetric()
			if bTrans {
				c := getWorkspace(bc, br, false)
				blas64.Symm(blas.Right, 1, amat, bmat, 0, c.mat)
				strictCopy(m, c.T())
				putWorkspace(c)
				return
			}
			blas64.Symm(blas.Left, 1, amat, bmat, 0, m.mat)
			return
		}
		if aU, ok := aU.(RawTriangular); ok {
			// Trmm updates in place, so copy bU first.
			amat := aU.RawTriangular()
			if bTrans {
				c := getWorkspace(bc, br, false)
				var tmp Dense
				tmp.SetRawMatrix(bmat)
				c.Copy(&tmp)
				aT := blas.Trans
				if aTrans {
					aT = blas.NoTrans
				}
				blas64.Trmm(blas.Right, aT, 1, amat, c.mat)
				strictCopy(m, c.T())
				putWorkspace(c)
				return
			}
			m.Copy(b)
			blas64.Trmm(blas.Left, aT, 1, amat, m.mat)
			return
		}
		if aU, ok := aU.(*VecDense); ok {
			m.checkOverlap(aU.asGeneral())
			avec := aU.RawVector()
			if aTrans {
				// {1,ac} x {ac, bc}
				// Transpose B so that the vector is on the right.
				cvec := blas64.Vector{
					Inc:  1,
					Data: m.mat.Data,
				}
				bT := blas.Trans
				if bTrans {
					bT = blas.NoTrans
				}
				blas64.Gemv(bT, 1, bmat, avec, 0, cvec)
				return
			}
			// {ar,1} x {1,bc} which is not a vector result.
			// Instead, construct A as a General.
			amat := blas64.General{
				Rows:   ar,
				Cols:   1,
				Stride: avec.Inc,
				Data:   avec.Data,
			}
			blas64.Gemm(aT, bT, 1, amat, bmat, 0, m.mat)
			return
		}
	}

	m.checkOverlapMatrix(aU)
	m.checkOverlapMatrix(bU)
	row := getFloats(ac, false)
	defer putFloats(row)
	for r := 0; r < ar; r++ {
		for i := range row {
			row[i] = a.At(r, i)
		}
		for c := 0; c < bc; c++ {
			var v float64
			for i, e := range row {
				v += e * b.At(i, c)
			}
			m.mat.Data[r*m.mat.Stride+c] = v
		}
	}
}

// strictCopy copies a into m panicking if the shape of a and m differ.
func strictCopy(m *Dense, a Matrix) {
	r, c := m.Copy(a)
	if r != m.mat.Rows || c != m.mat.Cols {
		// Panic with a string since this
		// is not a user-facing panic.
		panic(ErrShape.Error())
	}
}

// Exp calculates the exponential of the matrix a, e^a, placing the result
// in the receiver. Exp will panic with matrix.ErrShape if a is not square.
func (m *Dense) Exp(a Matrix) {
	// The implementation used here is from Functions of Matrices: Theory and Computation
	// Chapter 10, Algorithm 10.20. https://doi.org/10.1137/1.9780898717778.ch10

	r, c := a.Dims()
	if r != c {
		panic(ErrShape)
	}

	m.reuseAs(r, r)
	if r == 1 {
		m.mat.Data[0] = math.Exp(a.At(0, 0))
		return
	}

	pade := []struct {
		theta float64
		b     []float64
	}{
		{theta: 0.015, b: []float64{
			120, 60, 12, 1,
		}},
		{theta: 0.25, b: []float64{
			30240, 15120, 3360, 420, 30, 1,
		}},
		{theta: 0.95, b: []float64{
			17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1,
		}},
		{theta: 2.1, b: []float64{
			17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1,
		}},
	}

	a1 := m
	a1.Copy(a)
	v := getWorkspace(r, r, true)
	vraw := v.RawMatrix()
	vvec := blas64.Vector{Inc: 1, Data: vraw.Data}
	defer putWorkspace(v)

	u := getWorkspace(r, r, true)
	uraw := u.RawMatrix()
	uvec := blas64.Vector{Inc: 1, Data: uraw.Data}
	defer putWorkspace(u)

	a2 := getWorkspace(r, r, false)
	defer putWorkspace(a2)

	n1 := Norm(a, 1)
	for i, t := range pade {
		if n1 > t.theta {
			continue
		}

		// This loop only executes once, so
		// this is not as horrible as it looks.
		p := getWorkspace(r, r, true)
		praw := p.RawMatrix()
		pvec := blas64.Vector{Inc: 1, Data: praw.Data}
		defer putWorkspace(p)

		for k := 0; k < r; k++ {
			p.set(k, k, 1)
			v.set(k, k, t.b[0])
			u.set(k, k, t.b[1])
		}

		a2.Mul(a1, a1)
		for j := 0; j <= i; j++ {
			p.Mul(p, a2)
			blas64.Axpy(r*r, t.b[2*j+2], pvec, vvec)
			blas64.Axpy(r*r, t.b[2*j+3], pvec, uvec)
		}
		u.Mul(a1, u)

		// Use p as a workspace here and
		// rename u for the second call's
		// receiver.
		vmu, vpu := u, p
		vpu.Add(v, u)
		vmu.Sub(v, u)

		m.Solve(vmu, vpu)
		return
	}

	// Remaining Padé table line.
	const theta13 = 5.4
	b := [...]float64{
		64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800,
		129060195264000, 10559470521600, 670442572800, 33522128640,
		1323241920, 40840800, 960960, 16380, 182, 1,
	}

	s := math.Log2(n1 / theta13)
	if s >= 0 {
		s = math.Ceil(s)
		a1.Scale(1/math.Pow(2, s), a1)
	}
	a2.Mul(a1, a1)

	i := getWorkspace(r, r, true)
	for j := 0; j < r; j++ {
		i.set(j, j, 1)
	}
	iraw := i.RawMatrix()
	ivec := blas64.Vector{Inc: 1, Data: iraw.Data}
	defer putWorkspace(i)

	a2raw := a2.RawMatrix()
	a2vec := blas64.Vector{Inc: 1, Data: a2raw.Data}

	a4 := getWorkspace(r, r, false)
	a4raw := a4.RawMatrix()
	a4vec := blas64.Vector{Inc: 1, Data: a4raw.Data}
	defer putWorkspace(a4)
	a4.Mul(a2, a2)

	a6 := getWorkspace(r, r, false)
	a6raw := a6.RawMatrix()
	a6vec := blas64.Vector{Inc: 1, Data: a6raw.Data}
	defer putWorkspace(a6)
	a6.Mul(a2, a4)

	// V = A_6(b_12*A_6 + b_10*A_4 + b_8*A_2) + b_6*A_6 + b_4*A_4 + b_2*A_2 +b_0*I
	blas64.Axpy(r*r, b[12], a6vec, vvec)
	blas64.Axpy(r*r, b[10], a4vec, vvec)
	blas64.Axpy(r*r, b[8], a2vec, vvec)
	v.Mul(v, a6)
	blas64.Axpy(r*r, b[6], a6vec, vvec)
	blas64.Axpy(r*r, b[4], a4vec, vvec)
	blas64.Axpy(r*r, b[2], a2vec, vvec)
	blas64.Axpy(r*r, b[0], ivec, vvec)

	// U = A(A_6(b_13*A_6 + b_11*A_4 + b_9*A_2) + b_7*A_6 + b_5*A_4 + b_2*A_3 +b_1*I)
	blas64.Axpy(r*r, b[13], a6vec, uvec)
	blas64.Axpy(r*r, b[11], a4vec, uvec)
	blas64.Axpy(r*r, b[9], a2vec, uvec)
	u.Mul(u, a6)
	blas64.Axpy(r*r, b[7], a6vec, uvec)
	blas64.Axpy(r*r, b[5], a4vec, uvec)
	blas64.Axpy(r*r, b[3], a2vec, uvec)
	blas64.Axpy(r*r, b[1], ivec, uvec)
	u.Mul(u, a1)

	// Use i as a workspace here and
	// rename u for the second call's
	// receiver.
	vmu, vpu := u, i
	vpu.Add(v, u)
	vmu.Sub(v, u)

	m.Solve(vmu, vpu)

	for ; s > 0; s-- {
		m.Mul(m, m)
	}
}

// Pow calculates the integral power of the matrix a to n, placing the result
// in the receiver. Pow will panic if n is negative or if a is not square.
func (m *Dense) Pow(a Matrix, n int) {
	if n < 0 {
		panic("matrix: illegal power")
	}
	r, c := a.Dims()
	if r != c {
		panic(ErrShape)
	}

	m.reuseAs(r, c)

	// Take possible fast paths.
	switch n {
	case 0:
		for i := 0; i < r; i++ {
			zero(m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+c])
			m.mat.Data[i*m.mat.Stride+i] = 1
		}
		return
	case 1:
		m.Copy(a)
		return
	case 2:
		m.Mul(a, a)
		return
	}

	// Perform iterative exponentiation by squaring in work space.
	w := getWorkspace(r, r, false)
	w.Copy(a)
	s := getWorkspace(r, r, false)
	s.Copy(a)
	x := getWorkspace(r, r, false)
	for n--; n > 0; n >>= 1 {
		if n&1 != 0 {
			x.Mul(w, s)
			w, x = x, w
		}
		if n != 1 {
			x.Mul(s, s)
			s, x = x, s
		}
	}
	m.Copy(w)
	putWorkspace(w)
	putWorkspace(s)
	putWorkspace(x)
}

// Scale multiplies the elements of a by f, placing the result in the receiver.
//
// See the Scaler interface for more information.
func (m *Dense) Scale(f float64, a Matrix) {
	ar, ac := a.Dims()

	m.reuseAs(ar, ac)

	aU, aTrans := untranspose(a)
	if rm, ok := aU.(RawMatrixer); ok {
		amat := rm.RawMatrix()
		if m == aU || m.checkOverlap(amat) {
			var restore func()
			m, restore = m.isolatedWorkspace(a)
			defer restore()
		}
		if !aTrans {
			for ja, jm := 0, 0; ja < ar*amat.Stride; ja, jm = ja+amat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v * f
				}
			}
		} else {
			for ja, jm := 0, 0; ja < ac*amat.Stride; ja, jm = ja+amat.Stride, jm+1 {
				for i, v := range amat.Data[ja : ja+ar] {
					m.mat.Data[i*m.mat.Stride+jm] = v * f
				}
			}
		}
		return
	}

	m.checkOverlapMatrix(a)
	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, f*a.At(r, c))
		}
	}
}

// Apply applies the function fn to each of the elements of a, placing the
// resulting matrix in the receiver. The function fn takes a row/column
// index and element value and returns some function of that tuple.
func (m *Dense) Apply(fn func(i, j int, v float64) float64, a Matrix) {
	ar, ac := a.Dims()

	m.reuseAs(ar, ac)

	aU, aTrans := untranspose(a)
	if rm, ok := aU.(RawMatrixer); ok {
		amat := rm.RawMatrix()
		if m == aU || m.checkOverlap(amat) {
			var restore func()
			m, restore = m.isolatedWorkspace(a)
			defer restore()
		}
		if !aTrans {
			for j, ja, jm := 0, 0, 0; ja < ar*amat.Stride; j, ja, jm = j+1, ja+amat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = fn(j, i, v)
				}
			}
		} else {
			for j, ja, jm := 0, 0, 0; ja < ac*amat.Stride; j, ja, jm = j+1, ja+amat.Stride, jm+1 {
				for i, v := range amat.Data[ja : ja+ar] {
					m.mat.Data[i*m.mat.Stride+jm] = fn(i, j, v)
				}
			}
		}
		return
	}

	m.checkOverlapMatrix(a)
	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, fn(r, c, a.At(r, c)))
		}
	}
}

// RankOne performs a rank-one update to the matrix a and stores the result
// in the receiver. If a is zero, see Outer.
//  m = a + alpha * x * y'
func (m *Dense) RankOne(a Matrix, alpha float64, x, y Vector) {
	ar, ac := a.Dims()
	xr, xc := x.Dims()
	if xr != ar || xc != 1 {
		panic(ErrShape)
	}
	yr, yc := y.Dims()
	if yr != ac || yc != 1 {
		panic(ErrShape)
	}

	if a != m {
		aU, _ := untranspose(a)
		if rm, ok := aU.(RawMatrixer); ok {
			m.checkOverlap(rm.RawMatrix())
		}
	}

	var xmat, ymat blas64.Vector
	fast := true
	xU, _ := untranspose(x)
	if rv, ok := xU.(RawVectorer); ok {
		xmat = rv.RawVector()
		m.checkOverlap((&VecDense{mat: xmat, n: x.Len()}).asGeneral())
	} else {
		fast = false
	}
	yU, _ := untranspose(y)
	if rv, ok := yU.(RawVectorer); ok {
		ymat = rv.RawVector()
		m.checkOverlap((&VecDense{mat: ymat, n: y.Len()}).asGeneral())
	} else {
		fast = false
	}

	if fast {
		if m != a {
			m.reuseAs(ar, ac)
			m.Copy(a)
		}
		blas64.Ger(alpha, xmat, ymat, m.mat)
		return
	}

	m.reuseAs(ar, ac)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			m.set(i, j, a.At(i, j)+alpha*x.AtVec(i)*y.AtVec(j))
		}
	}
}

// Outer calculates the outer product of the column vectors x and y,
// and stores the result in the receiver.
//  m = alpha * x * y'
// In order to update an existing matrix, see RankOne.
func (m *Dense) Outer(alpha float64, x, y Vector) {
	xr, xc := x.Dims()
	if xc != 1 {
		panic(ErrShape)
	}
	yr, yc := y.Dims()
	if yc != 1 {
		panic(ErrShape)
	}

	r := xr
	c := yr

	// Copied from reuseAs with use replaced by useZeroed
	// and a final zero of the matrix elements if we pass
	// the shape checks.
	// TODO(kortschak): Factor out into reuseZeroedAs if
	// we find another case that needs it.
	if m.mat.Rows > m.capRows || m.mat.Cols > m.capCols {
		// Panic as a string, not a mat.Error.
		panic("mat: caps not correctly set")
	}
	if m.IsZero() {
		m.mat = blas64.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			Data:   useZeroed(m.mat.Data, r*c),
		}
		m.capRows = r
		m.capCols = c
	} else if r != m.mat.Rows || c != m.mat.Cols {
		panic(ErrShape)
	}

	var xmat, ymat blas64.Vector
	fast := true
	xU, _ := untranspose(x)
	if rv, ok := xU.(RawVectorer); ok {
		xmat = rv.RawVector()
		m.checkOverlap((&VecDense{mat: xmat, n: x.Len()}).asGeneral())

	} else {
		fast = false
	}
	yU, _ := untranspose(y)
	if rv, ok := yU.(RawVectorer); ok {
		ymat = rv.RawVector()
		m.checkOverlap((&VecDense{mat: ymat, n: y.Len()}).asGeneral())
	} else {
		fast = false
	}

	if fast {
		for i := 0; i < r; i++ {
			zero(m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+c])
		}
		blas64.Ger(alpha, xmat, ymat, m.mat)
		return
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.set(i, j, alpha*x.AtVec(i)*y.AtVec(j))
		}
	}
}
