// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package swag

import "github.com/go-openapi/swag/fileutils"

// GOPATHKey represents the env key for gopath
//
// Deprecated: use [fileutils.GOPATHKey] instead.
const GOPATHKey = fileutils.GOPATHKey

// File represents an uploaded file.
//
// Deprecated: use [fileutils.File] instead.
type File = fileutils.File

// FindInSearchPath finds a package in a provided lists of paths.
//
// Deprecated: use [fileutils.FindInSearchPath] instead.
func FindInSearchPath(searchPath, pkg string) string {
	return fileutils.FindInSearchPath(searchPath, pkg)
}

// FindInGoSearchPath finds a package in the $GOPATH:$GOROOT
//
// Deprecated: use [fileutils.FindInGoSearchPath] instead.
func FindInGoSearchPath(pkg string) string { return fileutils.FindInGoSearchPath(pkg) }

// FullGoSearchPath gets the search paths for finding packages
//
// Deprecated: use [fileutils.FullGoSearchPath] instead.
func FullGoSearchPath() string { return fileutils.FullGoSearchPath() }
