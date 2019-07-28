// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"runtime"

	"gonum.org/v1/gonum/lapack"
)

// Condition is the condition number of a matrix. The condition
// number is defined as |A| * |A^-1|.
//
// One important use of Condition is during linear solve routines (finding x such
// that A * x = b). The condition number of A indicates the accuracy of
// the computed solution. A Condition error will be returned if the condition
// number of A is sufficiently large. If A is exactly singular to working precision,
// Condition == ∞, and the solve algorithm may have completed early. If Condition
// is large and finite the solve algorithm will be performed, but the computed
// solution may be innacurate. Due to the nature of finite precision arithmetic,
// the value of Condition is only an approximate test of singularity.
type Condition float64

func (c Condition) Error() string {
	return fmt.Sprintf("matrix singular or near-singular with condition number %.4e", c)
}

// ConditionTolerance is the tolerance limit of the condition number. If the
// condition number is above this value, the matrix is considered singular.
const ConditionTolerance = 1e16

const (
	// CondNorm is the matrix norm used for computing the condition number by routines
	// in the matrix packages.
	CondNorm = lapack.MaxRowSum

	// CondNormTrans is the norm used to compute on A^T to get the same result as
	// computing CondNorm on A.
	CondNormTrans = lapack.MaxColumnSum
)

const stackTraceBufferSize = 1 << 20

// Maybe will recover a panic with a type mat.Error from fn, and return this error
// as the Err field of an ErrorStack. The stack trace for the panicking function will be
// recovered and placed in the StackTrace field. Any other error is re-panicked.
func Maybe(fn func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(Error); ok {
				if e.string == "" {
					panic("mat: invalid error")
				}
				buf := make([]byte, stackTraceBufferSize)
				n := runtime.Stack(buf, false)
				err = ErrorStack{Err: e, StackTrace: string(buf[:n])}
				return
			}
			panic(r)
		}
	}()
	fn()
	return
}

// MaybeFloat will recover a panic with a type mat.Error from fn, and return this error
// as the Err field of an ErrorStack. The stack trace for the panicking function will be
// recovered and placed in the StackTrace field. Any other error is re-panicked.
func MaybeFloat(fn func() float64) (f float64, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(Error); ok {
				if e.string == "" {
					panic("mat: invalid error")
				}
				buf := make([]byte, stackTraceBufferSize)
				n := runtime.Stack(buf, false)
				err = ErrorStack{Err: e, StackTrace: string(buf[:n])}
				return
			}
			panic(r)
		}
	}()
	return fn(), nil
}

// MaybeComplex will recover a panic with a type mat.Error from fn, and return this error
// as the Err field of an ErrorStack. The stack trace for the panicking function will be
// recovered and placed in the StackTrace field. Any other error is re-panicked.
func MaybeComplex(fn func() complex128) (f complex128, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(Error); ok {
				if e.string == "" {
					panic("mat: invalid error")
				}
				buf := make([]byte, stackTraceBufferSize)
				n := runtime.Stack(buf, false)
				err = ErrorStack{Err: e, StackTrace: string(buf[:n])}
				return
			}
			panic(r)
		}
	}()
	return fn(), nil
}

// Error represents matrix handling errors. These errors can be recovered by Maybe wrappers.
type Error struct{ string }

func (err Error) Error() string { return err.string }

var (
	ErrIndexOutOfRange     = Error{"matrix: index out of range"}
	ErrRowAccess           = Error{"matrix: row index out of range"}
	ErrColAccess           = Error{"matrix: column index out of range"}
	ErrVectorAccess        = Error{"matrix: vector index out of range"}
	ErrZeroLength          = Error{"matrix: zero length in matrix dimension"}
	ErrRowLength           = Error{"matrix: row length mismatch"}
	ErrColLength           = Error{"matrix: col length mismatch"}
	ErrSquare              = Error{"matrix: expect square matrix"}
	ErrNormOrder           = Error{"matrix: invalid norm order for matrix"}
	ErrSingular            = Error{"matrix: matrix is singular"}
	ErrShape               = Error{"matrix: dimension mismatch"}
	ErrIllegalStride       = Error{"matrix: illegal stride"}
	ErrPivot               = Error{"matrix: malformed pivot list"}
	ErrTriangle            = Error{"matrix: triangular storage mismatch"}
	ErrTriangleSet         = Error{"matrix: triangular set out of bounds"}
	ErrBandSet             = Error{"matrix: band set out of bounds"}
	ErrDiagSet             = Error{"matrix: diagonal set out of bounds"}
	ErrSliceLengthMismatch = Error{"matrix: input slice length mismatch"}
	ErrNotPSD              = Error{"matrix: input not positive symmetric definite"}
	ErrFailedEigen         = Error{"matrix: eigendecomposition not successful"}
)

// ErrorStack represents matrix handling errors that have been recovered by Maybe wrappers.
type ErrorStack struct {
	Err error

	// StackTrace is the stack trace
	// recovered by Maybe, MaybeFloat
	// or MaybeComplex.
	StackTrace string
}

func (err ErrorStack) Error() string { return err.Err.Error() }
