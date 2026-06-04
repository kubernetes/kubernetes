// SPDX-License-Identifier: MPL-2.0

//go:build linux

// Copyright (C) 2024-2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2024-2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

package gopathrs

import (
	"os"
)

// OpenatInRoot is equivalent to [OpenInRoot], except that the root is provided
// using an *[os.File] handle, to ensure that the correct root directory is used.
func OpenatInRoot(root *os.File, unsafePath string) (*os.File, error) {
	handle, err := completeLookupInRoot(root, unsafePath)
	if err != nil {
		return nil, &os.PathError{Op: "securejoin.OpenInRoot", Path: unsafePath, Err: err}
	}
	return handle, nil
}
