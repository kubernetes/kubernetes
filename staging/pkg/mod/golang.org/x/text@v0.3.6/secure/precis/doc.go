// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package precis contains types and functions for the preparation,
// enforcement, and comparison of internationalized strings ("PRECIS") as
// defined in RFC 8264. It also contains several pre-defined profiles for
// passwords, nicknames, and usernames as defined in RFC 8265 and RFC 8266.
//
// BE ADVISED: This package is under construction and the API may change in
// backwards incompatible ways and without notice.
package precis // import "golang.org/x/text/secure/precis"

//go:generate go run gen.go gen_trieval.go
