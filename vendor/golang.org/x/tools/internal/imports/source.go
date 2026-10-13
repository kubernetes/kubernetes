// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import "context"

// These types document the APIs below.
//
// TODO(rfindley): consider making these defined types rather than aliases.
type (
	ImportPath  = string
	PackageName = string
	Symbol      = string

	// References is set of References found in a Go file. The first map key is the
	// left hand side of a selector expression, the second key is the right hand
	// side, and the value should always be true.
	References = map[PackageName]map[Symbol]bool
)

// A Result satisfies a missing import.
//
// The Import field describes the missing import spec, and the Package field
// summarizes the package exports.
type Result struct {
	Import  *ImportInfo
	Package *PackageInfo
}

// An ImportInfo represents a single import statement.
type ImportInfo struct {
	ImportPath string // import path, e.g. "crypto/rand".
	Name       string // import name, e.g. "crand", or "" if none.
}

// A PackageInfo represents what's known about a package.
type PackageInfo struct {
	Name    string          // package name in the package declaration, if known
	Exports map[string]bool // set of names of known package level sortSymbols
}

// A Source provides imports to satisfy unresolved references in the file being
// fixed.
type Source interface {
	// LoadPackageNames queries PackageName information for the requested import
	// paths, when operating from the provided srcDir.
	//
	// TODO(rfindley): try to refactor to remove this operation.
	LoadPackageNames(ctx context.Context, srcDir string, paths []ImportPath) (map[ImportPath]PackageName, error)

	// ResolveReferences asks the Source for the best package name to satisfy
	// each of the missing references, in the context of fixing the given
	// filename.
	//
	// Returns a map from package name to a [Result] for that package name that
	// provides the required symbols. Keys may be omitted in the map if no
	// candidates satisfy all missing references for that package name. It is up
	// to each data source to select the best result for each entry in the
	// missing map.
	ResolveReferences(ctx context.Context, filename string, missing References) ([]*Result, error)
}
