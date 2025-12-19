//go:build linux

// TODO: Use "go:build unix" once we bump the minimum Go version 1.19.

// SPDX-License-Identifier: MPL-2.0
/*
 * libpathrs: safe path resolution on Linux
 * Copyright (C) 2019-2025 Aleksa Sarai <cyphar@cyphar.com>
 * Copyright (C) 2019-2025 SUSE LLC
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

package libpathrs

import (
	"syscall"
)

// Error represents an underlying libpathrs error.
type Error struct {
	description string
	errno       syscall.Errno
}

// Error returns a textual description of the error.
func (err *Error) Error() string {
	return err.description
}

// Unwrap returns the underlying error which was wrapped by this error (if
// applicable).
func (err *Error) Unwrap() error {
	if err.errno != 0 {
		return err.errno
	}
	return nil
}
