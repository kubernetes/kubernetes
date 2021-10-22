// Copyright 2020, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package foo is deliberately named differently than the parent directory.
// It contain declarations that have ambiguity in their short names,
// relative to a different package also called foo.
package foo

type Bar struct{ S string }
