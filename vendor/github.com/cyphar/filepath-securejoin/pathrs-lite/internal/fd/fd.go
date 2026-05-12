// SPDX-License-Identifier: MPL-2.0

// Copyright (C) 2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Package fd provides a drop-in interface-based replacement of [*os.File] that
// allows for things like noop-Close wrappers to be used.
//
// [*os.File]: https://pkg.go.dev/os#File
package fd

import (
	"io"
	"os"
)

// Fd is an interface that mirrors most of the API of [*os.File], allowing you
// to create wrappers that can be used in place of [*os.File].
//
// [*os.File]: https://pkg.go.dev/os#File
type Fd interface {
	io.Closer
	Name() string
	Fd() uintptr
}

// Compile-time interface checks.
var (
	_ Fd = (*os.File)(nil)
	_ Fd = noClose{}
)

type noClose struct{ inner Fd }

func (f noClose) Name() string { return f.inner.Name() }
func (f noClose) Fd() uintptr  { return f.inner.Fd() }

func (f noClose) Close() error { return nil }

// NopCloser returns an [*os.File]-like object where the [Close] method is now
// a no-op.
//
// Note that for [*os.File] and similar objects, the Go garbage collector will
// still call [Close] on the underlying file unless you use
// [runtime.SetFinalizer] to disable this behaviour. This is up to the caller
// to do (if necessary).
//
// [*os.File]: https://pkg.go.dev/os#File
// [Close]: https://pkg.go.dev/io#Closer
// [runtime.SetFinalizer]: https://pkg.go.dev/runtime#SetFinalizer
func NopCloser(f Fd) Fd { return noClose{inner: f} }
