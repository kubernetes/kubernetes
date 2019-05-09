// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

// This file contains code common to the maketables.go and the package code.

// AliasType is the type of an alias in AliasMap.
type AliasType int8

const (
	Deprecated AliasType = iota
	Macro
	Legacy

	AliasTypeUnknown AliasType = -1
)
