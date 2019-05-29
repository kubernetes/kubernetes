// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mat provides implementations of float64 and complex128 matrix
// structures and linear algebra operations on them.
//
// Overview
//
// This section provides a quick overview of the mat package. The following
// sections provide more in depth commentary.
//
// mat provides:
//  - Interfaces for Matrix classes (Matrix, Symmetric, Triangular)
//  - Concrete implementations (Dense, SymDense, TriDense)
//  - Methods and functions for using matrix data (Add, Trace, SymRankOne)
//  - Types for constructing and using matrix factorizations (QR, LU)
//  - The complementary types for complex matrices, CMatrix, CSymDense, etc.
//
// A matrix may be constructed through the corresponding New function. If no
// backing array is provided the matrix will be initialized to all zeros.
//  // Allocate a zeroed real matrix of size 3×5
//  zero := mat.NewDense(3, 5, nil)
// If a backing data slice is provided, the matrix will have those elements.
// Matrices are all stored in row-major format.
//  // Generate a 6×6 matrix of random values.
//  data := make([]float64, 36)
//  for i := range data {
//  	data[i] = rand.NormFloat64()
//  }
//  a := mat.NewDense(6, 6, data)
// Operations involving matrix data are implemented as functions when the values
// of the matrix remain unchanged
//  tr := mat.Trace(a)
// and are implemented as methods when the operation modifies the receiver.
//  zero.Copy(a)
//
// Receivers must be the correct size for the matrix operations, otherwise the
// operation will panic. As a special case for convenience, a zero-value matrix
// will be modified to have the correct size, allocating data if necessary.
//  var c mat.Dense // construct a new zero-sized matrix
//  c.Mul(a, a)     // c is automatically adjusted to be 6×6
//
// Zero-value of a matrix
//
// A zero-value matrix is either the Go language definition of a zero-value or
// is a zero-sized matrix with zero-length stride. Matrix implementations may have
// a Reset method to revert the receiver into a zero-valued matrix and an IsZero
// method that returns whether the matrix is zero-valued.
// So the following will all result in a zero-value matrix.
//  - var a mat.Dense
//  - a := NewDense(0, 0, make([]float64, 0, 100))
//  - a.Reset()
// A zero-value matrix can not be sliced even if it does have an adequately sized
// backing data slice, but can be expanded using its Grow method if it exists.
//
// The Matrix Interfaces
//
// The Matrix interface is the common link between the concrete types of real
// matrices, The Matrix interface is defined by three functions: Dims, which
// returns the dimensions of the Matrix, At, which returns the element in the
// specified location, and T for returning a Transpose (discussed later). All of
// the concrete types can perform these behaviors and so implement the interface.
// Methods and functions are designed to use this interface, so in particular the method
//  func (m *Dense) Mul(a, b Matrix)
// constructs a *Dense from the result of a multiplication with any Matrix types,
// not just *Dense. Where more restrictive requirements must be met, there are also the
// Symmetric and Triangular interfaces. For example, in
//  func (s *SymDense) AddSym(a, b Symmetric)
// the Symmetric interface guarantees a symmetric result.
//
// The CMatrix interface plays the same role for complex matrices. The difference
// is that the CMatrix type has the H method instead T, for returning the conjugate
// transpose.
//
// (Conjugate) Transposes
//
// The T method is used for transposition on real matrices, and H is used for
// conjugate transposition on complex matrices. For example, c.Mul(a.T(), b) computes
// c = a^T * b. The mat types implement this method implicitly —
// see the Transpose and Conjugate types for more details. Note that some
// operations have a transpose as part of their definition, as in *SymDense.SymOuterK.
//
// Matrix Factorization
//
// Matrix factorizations, such as the LU decomposition, typically have their own
// specific data storage, and so are each implemented as a specific type. The
// factorization can be computed through a call to Factorize
//  var lu mat.LU
//  lu.Factorize(a)
// The elements of the factorization can be extracted through methods on the
// factorized type, i.e. *LU.UTo. The factorization types can also be used directly,
// as in *Dense.SolveCholesky. Some factorizations can be updated directly,
// without needing to update the original matrix and refactorize,
// as in *LU.RankOne.
//
// BLAS and LAPACK
//
// BLAS and LAPACK are the standard APIs for linear algebra routines. Many
// operations in mat are implemented using calls to the wrapper functions
// in gonum/blas/blas64 and gonum/lapack/lapack64 and their complex equivalents.
// By default, blas64 and lapack64 call the native Go implementations of the
// routines. Alternatively, it is possible to use C-based implementations of the
// APIs through the respective cgo packages and "Use" functions. The Go
// implementation of LAPACK (used by default) makes calls
// through blas64, so if a cgo BLAS implementation is registered, the lapack64
// calls will be partially executed in Go and partially executed in C.
//
// Type Switching
//
// The Matrix abstraction enables efficiency as well as interoperability. Go's
// type reflection capabilities are used to choose the most efficient routine
// given the specific concrete types. For example, in
//  c.Mul(a, b)
// if a and b both implement RawMatrixer, that is, they can be represented as a
// blas64.General, blas64.Gemm (general matrix multiplication) is called, while
// instead if b is a RawSymmetricer blas64.Symm is used (general-symmetric
// multiplication), and if b is a *VecDense blas64.Gemv is used.
//
// There are many possible type combinations and special cases. No specific guarantees
// are made about the performance of any method, and in particular, note that an
// abstract matrix type may be copied into a concrete type of the corresponding
// value. If there are specific special cases that are needed, please submit a
// pull-request or file an issue.
//
// Invariants
//
// Matrix input arguments to functions are never directly modified. If an operation
// changes Matrix data, the mutated matrix will be the receiver of a function.
//
// For convenience, a matrix may be used as both a receiver and as an input, e.g.
//  a.Pow(a, 6)
//  v.SolveVec(a.T(), v)
// though in many cases this will cause an allocation (see Element Aliasing).
// An exception to this rule is Copy, which does not allow a.Copy(a.T()).
//
// Element Aliasing
//
// Most methods in mat modify receiver data. It is forbidden for the modified
// data region of the receiver to overlap the used data area of the input
// arguments. The exception to this rule is when the method receiver is equal to one
// of the input arguments, as in the a.Pow(a, 6) call above, or its implicit transpose.
//
// This prohibition is to help avoid subtle mistakes when the method needs to read
// from and write to the same data region. There are ways to make mistakes using the
// mat API, and mat functions will detect and complain about those.
// There are many ways to make mistakes by excursion from the mat API via
// interaction with raw matrix values.
//
// If you need to read the rest of this section to understand the behavior of
// your program, you are being clever. Don't be clever. If you must be clever,
// blas64 and lapack64 may be used to call the behavior directly.
//
// mat will use the following rules to detect overlap between the receiver and one
// of the inputs:
//  - the input implements one of the Raw methods, and
//  - the address ranges of the backing data slices overlap, and
//  - the strides differ or there is an overlap in the used data elements.
// If such an overlap is detected, the method will panic.
//
// The following cases will not panic:
//  - the data slices do not overlap,
//  - there is pointer identity between the receiver and input values after
//    the value has been untransposed if necessary.
//
// mat will not attempt to detect element overlap if the input does not implement a
// Raw method. Method behavior is undefined if there is undetected overlap.
//
package mat // import "gonum.org/v1/gonum/mat"
