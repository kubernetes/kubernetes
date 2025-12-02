// SPDX-License-Identifier: MPL-2.0

// Copyright (C) 2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Package assert provides some basic assertion helpers for Go.
package assert

import (
	"fmt"
)

// Assert panics if the predicate is false with the provided argument.
func Assert(predicate bool, msg any) {
	if !predicate {
		panic(msg)
	}
}

// Assertf panics if the predicate is false and formats the message using the
// same formatting as [fmt.Printf].
//
// [fmt.Printf]: https://pkg.go.dev/fmt#Printf
func Assertf(predicate bool, fmtMsg string, args ...any) {
	Assert(predicate, fmt.Sprintf(fmtMsg, args...))
}
