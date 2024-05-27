// Copyright 2023 The Bazel Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import _ "unsafe" // for linkname

// FileOptions specifies various per-file options that affect static
// aspects of an individual file such as parsing, name resolution, and
// code generation. (Options that affect global dynamics are typically
// controlled through [go.starlark.net/starlark.Thread].)
//
// The zero value of FileOptions is the default behavior.
//
// Many functions in this package come in two versions: the legacy
// standalone function (such as [Parse]) uses [LegacyFileOptions],
// whereas the more recent method (such as [FileOptions.Parse]) honors the
// provided options. The second form is preferred. In other packages,
// the modern version is a standalone function with a leading
// FileOptions parameter and the name suffix "Options", such as
// [go.starlark.net/starlark.ExecFileOptions].
type FileOptions struct {
	// resolver
	Set               bool // allow references to the 'set' built-in function
	While             bool // allow 'while' statements
	TopLevelControl   bool // allow if/for/while statements at top-level
	GlobalReassign    bool // allow reassignment to top-level names
	LoadBindsGlobally bool // load creates global not file-local bindings (deprecated)

	// compiler
	Recursion bool // disable recursion check for functions in this file
}

// TODO(adonovan): provide a canonical flag parser for FileOptions.
// (And use it in the testdata "options:" strings.)

// LegacyFileOptions returns a new FileOptions containing the current
// values of the resolver package's legacy global variables such as
// [resolve.AllowRecursion], etc.
// These variables may be associated with command-line flags.
func LegacyFileOptions() *FileOptions {
	return &FileOptions{
		Set:               resolverAllowSet,
		While:             resolverAllowGlobalReassign,
		TopLevelControl:   resolverAllowGlobalReassign,
		GlobalReassign:    resolverAllowGlobalReassign,
		Recursion:         resolverAllowRecursion,
		LoadBindsGlobally: resolverLoadBindsGlobally,
	}
}

// Access resolver (legacy) flags, if they are linked in; false otherwise.
var (
	//go:linkname resolverAllowSet go.starlark.net/resolve.AllowSet
	resolverAllowSet bool
	//go:linkname resolverAllowGlobalReassign go.starlark.net/resolve.AllowGlobalReassign
	resolverAllowGlobalReassign bool
	//go:linkname resolverAllowRecursion go.starlark.net/resolve.AllowRecursion
	resolverAllowRecursion bool
	//go:linkname resolverLoadBindsGlobally go.starlark.net/resolve.LoadBindsGlobally
	resolverLoadBindsGlobally bool
)
