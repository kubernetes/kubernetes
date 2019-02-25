// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"sync"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

var tab64 = [64]byte{
	0x3f, 0x00, 0x3a, 0x01, 0x3b, 0x2f, 0x35, 0x02,
	0x3c, 0x27, 0x30, 0x1b, 0x36, 0x21, 0x2a, 0x03,
	0x3d, 0x33, 0x25, 0x28, 0x31, 0x12, 0x1c, 0x14,
	0x37, 0x1e, 0x22, 0x0b, 0x2b, 0x0e, 0x16, 0x04,
	0x3e, 0x39, 0x2e, 0x34, 0x26, 0x1a, 0x20, 0x29,
	0x32, 0x24, 0x11, 0x13, 0x1d, 0x0a, 0x0d, 0x15,
	0x38, 0x2d, 0x19, 0x1f, 0x23, 0x10, 0x09, 0x0c,
	0x2c, 0x18, 0x0f, 0x08, 0x17, 0x07, 0x06, 0x05,
}

// bits returns the ceiling of base 2 log of v.
// Approach based on http://stackoverflow.com/a/11398748.
func bits(v uint64) byte {
	if v == 0 {
		return 0
	}
	v <<= 2
	v--
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v |= v >> 32
	return tab64[((v-(v>>1))*0x07EDD5E59A4E28C2)>>58] - 1
}

var (
	// pool contains size stratified workspace Dense pools.
	// Each pool element i returns sized matrices with a data
	// slice capped at 1<<i.
	pool [63]sync.Pool

	// poolSym is the SymDense equivalent of pool.
	poolSym [63]sync.Pool

	// poolTri is the TriDense equivalent of pool.
	poolTri [63]sync.Pool

	// poolVec is the VecDense equivalent of pool.
	poolVec [63]sync.Pool

	// poolFloats is the []float64 equivalent of pool.
	poolFloats [63]sync.Pool

	// poolInts is the []int equivalent of pool.
	poolInts [63]sync.Pool
)

func init() {
	for i := range pool {
		l := 1 << uint(i)
		pool[i].New = func() interface{} {
			return &Dense{mat: blas64.General{
				Data: make([]float64, l),
			}}
		}
		poolSym[i].New = func() interface{} {
			return &SymDense{mat: blas64.Symmetric{
				Uplo: blas.Upper,
				Data: make([]float64, l),
			}}
		}
		poolTri[i].New = func() interface{} {
			return &TriDense{mat: blas64.Triangular{
				Data: make([]float64, l),
			}}
		}
		poolVec[i].New = func() interface{} {
			return &VecDense{mat: blas64.Vector{
				Inc:  1,
				Data: make([]float64, l),
			}}
		}
		poolFloats[i].New = func() interface{} {
			return make([]float64, l)
		}
		poolInts[i].New = func() interface{} {
			return make([]int, l)
		}
	}
}

// getWorkspace returns a *Dense of size r×c and a data slice
// with a cap that is less than 2*r*c. If clear is true, the
// data slice visible through the Matrix interface is zeroed.
func getWorkspace(r, c int, clear bool) *Dense {
	l := uint64(r * c)
	w := pool[bits(l)].Get().(*Dense)
	w.mat.Data = w.mat.Data[:l]
	if clear {
		zero(w.mat.Data)
	}
	w.mat.Rows = r
	w.mat.Cols = c
	w.mat.Stride = c
	w.capRows = r
	w.capCols = c
	return w
}

// putWorkspace replaces a used *Dense into the appropriate size
// workspace pool. putWorkspace must not be called with a matrix
// where references to the underlying data slice have been kept.
func putWorkspace(w *Dense) {
	pool[bits(uint64(cap(w.mat.Data)))].Put(w)
}

// getWorkspaceSym returns a *SymDense of size n and a cap that
// is less than 2*n. If clear is true, the data slice visible
// through the Matrix interface is zeroed.
func getWorkspaceSym(n int, clear bool) *SymDense {
	l := uint64(n)
	l *= l
	s := poolSym[bits(l)].Get().(*SymDense)
	s.mat.Data = s.mat.Data[:l]
	if clear {
		zero(s.mat.Data)
	}
	s.mat.N = n
	s.mat.Stride = n
	s.cap = n
	return s
}

// putWorkspaceSym replaces a used *SymDense into the appropriate size
// workspace pool. putWorkspaceSym must not be called with a matrix
// where references to the underlying data slice have been kept.
func putWorkspaceSym(s *SymDense) {
	poolSym[bits(uint64(cap(s.mat.Data)))].Put(s)
}

// getWorkspaceTri returns a *TriDense of size n and a cap that
// is less than 2*n. If clear is true, the data slice visible
// through the Matrix interface is zeroed.
func getWorkspaceTri(n int, kind TriKind, clear bool) *TriDense {
	l := uint64(n)
	l *= l
	t := poolTri[bits(l)].Get().(*TriDense)
	t.mat.Data = t.mat.Data[:l]
	if clear {
		zero(t.mat.Data)
	}
	t.mat.N = n
	t.mat.Stride = n
	if kind == Upper {
		t.mat.Uplo = blas.Upper
	} else if kind == Lower {
		t.mat.Uplo = blas.Lower
	} else {
		panic(ErrTriangle)
	}
	t.mat.Diag = blas.NonUnit
	t.cap = n
	return t
}

// putWorkspaceTri replaces a used *TriDense into the appropriate size
// workspace pool. putWorkspaceTri must not be called with a matrix
// where references to the underlying data slice have been kept.
func putWorkspaceTri(t *TriDense) {
	poolTri[bits(uint64(cap(t.mat.Data)))].Put(t)
}

// getWorkspaceVec returns a *VecDense of length n and a cap that
// is less than 2*n. If clear is true, the data slice visible
// through the Matrix interface is zeroed.
func getWorkspaceVec(n int, clear bool) *VecDense {
	l := uint64(n)
	v := poolVec[bits(l)].Get().(*VecDense)
	v.mat.Data = v.mat.Data[:l]
	if clear {
		zero(v.mat.Data)
	}
	v.n = n
	return v
}

// putWorkspaceVec replaces a used *VecDense into the appropriate size
// workspace pool. putWorkspaceVec must not be called with a matrix
// where references to the underlying data slice have been kept.
func putWorkspaceVec(v *VecDense) {
	poolVec[bits(uint64(cap(v.mat.Data)))].Put(v)
}

// getFloats returns a []float64 of length l and a cap that is
// less than 2*l. If clear is true, the slice visible is zeroed.
func getFloats(l int, clear bool) []float64 {
	w := poolFloats[bits(uint64(l))].Get().([]float64)
	w = w[:l]
	if clear {
		zero(w)
	}
	return w
}

// putFloats replaces a used []float64 into the appropriate size
// workspace pool. putFloats must not be called with a slice
// where references to the underlying data have been kept.
func putFloats(w []float64) {
	poolFloats[bits(uint64(cap(w)))].Put(w)
}

// getInts returns a []ints of length l and a cap that is
// less than 2*l. If clear is true, the slice visible is zeroed.
func getInts(l int, clear bool) []int {
	w := poolInts[bits(uint64(l))].Get().([]int)
	w = w[:l]
	if clear {
		for i := range w {
			w[i] = 0
		}
	}
	return w
}

// putInts replaces a used []int into the appropriate size
// workspace pool. putInts must not be called with a slice
// where references to the underlying data have been kept.
func putInts(w []int) {
	poolInts[bits(uint64(cap(w)))].Put(w)
}
