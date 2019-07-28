// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gonum is a pure-go implementation of the LAPACK API. The LAPACK API defines
// a set of algorithms for advanced matrix operations.
//
// The function definitions and implementations follow that of the netlib reference
// implementation. See http://www.netlib.org/lapack/explore-html/ for more
// information, and http://www.netlib.org/lapack/explore-html/d4/de1/_l_i_c_e_n_s_e_source.html
// for more license information.
//
// Slice function arguments frequently represent vectors and matrices. The data
// layout is identical to that found in https://godoc.org/gonum.org/v1/gonum/blas/gonum.
//
// Most LAPACK functions are built on top the routines defined in the BLAS API,
// and as such the computation time for many LAPACK functions is
// dominated by BLAS calls. Here, BLAS is accessed through the
// blas64 package (https://godoc.org/golang.org/v1/gonum/blas/blas64). In particular,
// this implies that an external BLAS library will be used if it is
// registered in blas64.
//
// The full LAPACK capability has not been implemented at present. The full
// API is very large, containing approximately 200 functions for double precision
// alone. Future additions will be focused on supporting the gonum matrix
// package (https://godoc.org/github.com/gonum/matrix/mat64), though pull requests
// with implementations and tests for LAPACK function are encouraged.
package gonum // import "gonum.org/v1/gonum/lapack/gonum"
