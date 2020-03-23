// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file must be kept in sync with index_bound_checks.go.

// +build !bounds

package mat

// At returns the element at row i, column j.
func (m *Dense) At(i, j int) float64 {
	if uint(i) >= uint(m.mat.Rows) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(m.mat.Cols) {
		panic(ErrColAccess)
	}
	return m.at(i, j)
}

func (m *Dense) at(i, j int) float64 {
	return m.mat.Data[i*m.mat.Stride+j]
}

// Set sets the element at row i, column j to the value v.
func (m *Dense) Set(i, j int, v float64) {
	if uint(i) >= uint(m.mat.Rows) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(m.mat.Cols) {
		panic(ErrColAccess)
	}
	m.set(i, j, v)
}

func (m *Dense) set(i, j int, v float64) {
	m.mat.Data[i*m.mat.Stride+j] = v
}

// At returns the element at row i, column j.
func (m *CDense) At(i, j int) complex128 {
	if uint(i) >= uint(m.mat.Rows) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(m.mat.Cols) {
		panic(ErrColAccess)
	}
	return m.at(i, j)
}

func (m *CDense) at(i, j int) complex128 {
	return m.mat.Data[i*m.mat.Stride+j]
}

// Set sets the element at row i, column j to the value v.
func (m *CDense) Set(i, j int, v complex128) {
	if uint(i) >= uint(m.mat.Rows) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(m.mat.Cols) {
		panic(ErrColAccess)
	}
	m.set(i, j, v)
}

func (m *CDense) set(i, j int, v complex128) {
	m.mat.Data[i*m.mat.Stride+j] = v
}

// At returns the element at row i.
// It panics if i is out of bounds or if j is not zero.
func (v *VecDense) At(i, j int) float64 {
	if uint(i) >= uint(v.mat.N) {
		panic(ErrRowAccess)
	}
	if j != 0 {
		panic(ErrColAccess)
	}
	return v.at(i)
}

// AtVec returns the element at row i.
// It panics if i is out of bounds.
func (v *VecDense) AtVec(i int) float64 {
	if uint(i) >= uint(v.mat.N) {
		panic(ErrRowAccess)
	}
	return v.at(i)
}

func (v *VecDense) at(i int) float64 {
	return v.mat.Data[i*v.mat.Inc]
}

// SetVec sets the element at row i to the value val.
// It panics if i is out of bounds.
func (v *VecDense) SetVec(i int, val float64) {
	if uint(i) >= uint(v.mat.N) {
		panic(ErrVectorAccess)
	}
	v.setVec(i, val)
}

func (v *VecDense) setVec(i int, val float64) {
	v.mat.Data[i*v.mat.Inc] = val
}

// At returns the element at row i and column j.
func (s *SymDense) At(i, j int) float64 {
	if uint(i) >= uint(s.mat.N) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(s.mat.N) {
		panic(ErrColAccess)
	}
	return s.at(i, j)
}

func (s *SymDense) at(i, j int) float64 {
	if i > j {
		i, j = j, i
	}
	return s.mat.Data[i*s.mat.Stride+j]
}

// SetSym sets the elements at (i,j) and (j,i) to the value v.
func (s *SymDense) SetSym(i, j int, v float64) {
	if uint(i) >= uint(s.mat.N) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(s.mat.N) {
		panic(ErrColAccess)
	}
	s.set(i, j, v)
}

func (s *SymDense) set(i, j int, v float64) {
	if i > j {
		i, j = j, i
	}
	s.mat.Data[i*s.mat.Stride+j] = v
}

// At returns the element at row i, column j.
func (t *TriDense) At(i, j int) float64 {
	if uint(i) >= uint(t.mat.N) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(t.mat.N) {
		panic(ErrColAccess)
	}
	return t.at(i, j)
}

func (t *TriDense) at(i, j int) float64 {
	isUpper := t.triKind()
	if (isUpper && i > j) || (!isUpper && i < j) {
		return 0
	}
	return t.mat.Data[i*t.mat.Stride+j]
}

// SetTri sets the element at row i, column j to the value v.
// It panics if the location is outside the appropriate half of the matrix.
func (t *TriDense) SetTri(i, j int, v float64) {
	if uint(i) >= uint(t.mat.N) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(t.mat.N) {
		panic(ErrColAccess)
	}
	isUpper := t.isUpper()
	if (isUpper && i > j) || (!isUpper && i < j) {
		panic(ErrTriangleSet)
	}
	t.set(i, j, v)
}

func (t *TriDense) set(i, j int, v float64) {
	t.mat.Data[i*t.mat.Stride+j] = v
}

// At returns the element at row i, column j.
func (b *BandDense) At(i, j int) float64 {
	if uint(i) >= uint(b.mat.Rows) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(b.mat.Cols) {
		panic(ErrColAccess)
	}
	return b.at(i, j)
}

func (b *BandDense) at(i, j int) float64 {
	pj := j + b.mat.KL - i
	if pj < 0 || b.mat.KL+b.mat.KU+1 <= pj {
		return 0
	}
	return b.mat.Data[i*b.mat.Stride+pj]
}

// SetBand sets the element at row i, column j to the value v.
// It panics if the location is outside the appropriate region of the matrix.
func (b *BandDense) SetBand(i, j int, v float64) {
	if uint(i) >= uint(b.mat.Rows) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(b.mat.Cols) {
		panic(ErrColAccess)
	}
	pj := j + b.mat.KL - i
	if pj < 0 || b.mat.KL+b.mat.KU+1 <= pj {
		panic(ErrBandSet)
	}
	b.set(i, j, v)
}

func (b *BandDense) set(i, j int, v float64) {
	pj := j + b.mat.KL - i
	b.mat.Data[i*b.mat.Stride+pj] = v
}

// At returns the element at row i, column j.
func (s *SymBandDense) At(i, j int) float64 {
	if uint(i) >= uint(s.mat.N) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(s.mat.N) {
		panic(ErrColAccess)
	}
	return s.at(i, j)
}

func (s *SymBandDense) at(i, j int) float64 {
	if i > j {
		i, j = j, i
	}
	pj := j - i
	if s.mat.K+1 <= pj {
		return 0
	}
	return s.mat.Data[i*s.mat.Stride+pj]
}

// SetSymBand sets the element at row i, column j to the value v.
// It panics if the location is outside the appropriate region of the matrix.
func (s *SymBandDense) SetSymBand(i, j int, v float64) {
	if uint(i) >= uint(s.mat.N) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(s.mat.N) {
		panic(ErrColAccess)
	}
	s.set(i, j, v)
}

func (s *SymBandDense) set(i, j int, v float64) {
	if i > j {
		i, j = j, i
	}
	pj := j - i
	if s.mat.K+1 <= pj {
		panic(ErrBandSet)
	}
	s.mat.Data[i*s.mat.Stride+pj] = v
}

func (t *TriBandDense) At(i, j int) float64 {
	if uint(i) >= uint(t.mat.N) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(t.mat.N) {
		panic(ErrColAccess)
	}
	return t.at(i, j)
}

func (t *TriBandDense) at(i, j int) float64 {
	// TODO(btracey): Support Diag field, see #692.
	isUpper := t.isUpper()
	if (isUpper && i > j) || (!isUpper && i < j) {
		return 0
	}
	kl := t.mat.K
	ku := 0
	if isUpper {
		ku = t.mat.K
		kl = 0
	}
	pj := j + kl - i
	if pj < 0 || kl+ku+1 <= pj {
		return 0
	}
	return t.mat.Data[i*t.mat.Stride+pj]
}

func (t *TriBandDense) SetTriBand(i, j int, v float64) {
	if uint(i) >= uint(t.mat.N) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(t.mat.N) {
		panic(ErrColAccess)
	}
	isUpper := t.isUpper()
	if (isUpper && i > j) || (!isUpper && i < j) {
		panic(ErrTriangleSet)
	}
	kl, ku := t.mat.K, 0
	if isUpper {
		kl, ku = 0, t.mat.K
	}
	pj := j + kl - i
	if pj < 0 || kl+ku+1 <= pj {
		panic(ErrBandSet)
	}
	// TODO(btracey): Support Diag field, see #692.
	t.mat.Data[i*t.mat.Stride+pj] = v
}

func (t *TriBandDense) setTriBand(i, j int, v float64) {
	var kl int
	if !t.isUpper() {
		kl = t.mat.K
	}
	pj := j + kl - i
	t.mat.Data[i*t.mat.Stride+pj] = v
}

// At returns the element at row i, column j.
func (d *DiagDense) At(i, j int) float64 {
	if uint(i) >= uint(d.mat.N) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(d.mat.N) {
		panic(ErrColAccess)
	}
	return d.at(i, j)
}

func (d *DiagDense) at(i, j int) float64 {
	if i != j {
		return 0
	}
	return d.mat.Data[i*d.mat.Inc]
}

// SetDiag sets the element at row i, column i to the value v.
// It panics if the location is outside the appropriate region of the matrix.
func (d *DiagDense) SetDiag(i int, v float64) {
	if uint(i) >= uint(d.mat.N) {
		panic(ErrRowAccess)
	}
	d.setDiag(i, v)
}

func (d *DiagDense) setDiag(i int, v float64) {
	d.mat.Data[i*d.mat.Inc] = v
}
