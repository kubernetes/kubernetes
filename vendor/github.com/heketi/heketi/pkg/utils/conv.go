//
// Copyright (c) 2018 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package utils

// BoolToYN returns a "y" (yes) or "n" (no) for the passed bool.
func BoolToYN(b bool) string {
	if b {
		return "y"
	}
	return "n"
}
