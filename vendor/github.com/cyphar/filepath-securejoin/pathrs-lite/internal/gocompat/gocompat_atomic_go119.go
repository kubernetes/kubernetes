// SPDX-License-Identifier: BSD-3-Clause

//go:build linux && go1.19

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gocompat

import (
	"sync/atomic"
)

// A Bool is an atomic boolean value.
// The zero value is false.
//
// Bool must not be copied after first use.
type Bool = atomic.Bool
