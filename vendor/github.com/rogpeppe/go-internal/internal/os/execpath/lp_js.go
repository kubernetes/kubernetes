// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package execpath

// Look searches for an executable named file, using getenv to look up
// environment variables. If getenv is nil, os.Getenv will be used. If file
// contains a slash, it is tried directly and getenv will not be called.  The
// result may be an absolute path or a path relative to the current directory.
func Look(file string, getenv func(string) string) (string, error) {
	// Wasm can not execute processes, so act as if there are no executables at all.
	return "", &Error{file, ErrNotFound}
}
