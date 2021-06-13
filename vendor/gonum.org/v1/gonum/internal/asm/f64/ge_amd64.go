// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

package f64

// Ger performs the rank-one operation
//  A += alpha * x * yᵀ
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func Ger(m, n uintptr, alpha float64, x []float64, incX uintptr, y []float64, incY uintptr, a []float64, lda uintptr)

// GemvN computes
//  y = alpha * A * x + beta * y
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func GemvN(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr)

// GemvT computes
//  y = alpha * Aᵀ * x + beta * y
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func GemvT(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr)
