//go:build linux && go1.20

// Copyright (C) 2024 SUSE LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package securejoin

import (
	"fmt"
)

// wrapBaseError is a helper that is equivalent to fmt.Errorf("%w: %w"), except
// that on pre-1.20 Go versions only errors.Is() works properly (errors.Unwrap)
// is only guaranteed to give you baseErr.
func wrapBaseError(baseErr, extraErr error) error {
	return fmt.Errorf("%w: %w", extraErr, baseErr)
}
