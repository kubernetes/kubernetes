//go:build linux

// SPDX-License-Identifier: MPL-2.0
/*
 * libpathrs: safe path resolution on Linux
 * Copyright (C) 2026 Aleksa Sarai <cyphar@cyphar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

package pathrs

import (
	"cyphar.com/go-pathrs/internal/libpathrs"
)

// LibraryVersionInfo contains information about the version and features
// supported by the underlying libpathrs.so library at runtime.
type LibraryVersionInfo = libpathrs.VersionInfo

// LibraryVersion returns information about the version and features supported
// by the underlying libpathrs.so library at runtime.
func LibraryVersion() (*LibraryVersionInfo, error) {
	return libpathrs.Version()
}
