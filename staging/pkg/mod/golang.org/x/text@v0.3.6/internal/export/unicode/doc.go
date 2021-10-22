// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unicode generates the Unicode tables in core.
package unicode

// This package is defined here, instead of core, as Go does not allow any
// standard packages to have non-standard imports, even if imported in files
// with a build ignore tag.

//go:generate go run gen.go -tables=all
//go:generate mv tables.go $GOROOT/src/unicode
