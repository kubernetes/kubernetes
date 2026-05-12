//go:build linux

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

package pathrs

import (
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

//nolint:cyclop // this function needs to handle a lot of cases
func toUnixMode(mode os.FileMode, needsType bool) (uint32, error) {
	sysMode := uint32(mode.Perm())
	switch mode & os.ModeType { //nolint:exhaustive // we only care about ModeType bits
	case 0:
		if needsType {
			sysMode |= unix.S_IFREG
		}
	case os.ModeDir:
		sysMode |= unix.S_IFDIR
	case os.ModeSymlink:
		sysMode |= unix.S_IFLNK
	case os.ModeCharDevice | os.ModeDevice:
		sysMode |= unix.S_IFCHR
	case os.ModeDevice:
		sysMode |= unix.S_IFBLK
	case os.ModeNamedPipe:
		sysMode |= unix.S_IFIFO
	case os.ModeSocket:
		sysMode |= unix.S_IFSOCK
	default:
		return 0, fmt.Errorf("invalid mode filetype %+o", mode)
	}
	if mode&os.ModeSetuid != 0 {
		sysMode |= unix.S_ISUID
	}
	if mode&os.ModeSetgid != 0 {
		sysMode |= unix.S_ISGID
	}
	if mode&os.ModeSticky != 0 {
		sysMode |= unix.S_ISVTX
	}
	return sysMode, nil
}
