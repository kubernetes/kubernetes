// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

// Package fileutils exposes utilities to deal with files and paths.
//
// It provides:
//
//   - [File], an abstraction of an uploaded file.
//     It is used by [github.com/go-openapi/runtime.File].
//   - Implementations of [fs.FS]: [OsFS] and [GlobOsFS] wrap the os package,
//     [MapFS] serves files held in memory, [OverlayFS] stacks file systems on top of one another,
//     [OpaqueFS] lets a layer claim a directory for itself, and [FileReaderFS] adds a ReadFile
//     method to any [fs.FS].
//   - [MustSub], to re-root a file system inline when the directory is a constant of the program.
//   - path search utilities, to locate a package in the go search path.
package fileutils
