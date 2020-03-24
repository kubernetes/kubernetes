// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package xerrors implements functions to manipulate errors.
//
// This package supports transitioning to the Go 2 proposal for error values:
//   https://golang.org/design/29934-error-values
//
// Most of the functions and types in this package will be incorporated into the
// standard library's errors package in Go 1.13; the behavior of this package's
// Errorf function will be incorporated into the standard library's fmt.Errorf.
// Use this package to get equivalent behavior in all supported Go versions. For
// example, create errors using
//
//    xerrors.New("write failed")
//
// or
//
//    xerrors.Errorf("while reading: %v", err)
//
// If you want your error type to participate in the new formatting
// implementation for %v and %+v, provide it with a Format method that calls
// xerrors.FormatError, as shown in the example for FormatError.
package xerrors // import "golang.org/x/xerrors"
