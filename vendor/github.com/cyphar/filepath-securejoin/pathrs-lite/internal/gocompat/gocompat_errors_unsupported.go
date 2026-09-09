// SPDX-License-Identifier: BSD-3-Clause

//go:build linux && !go1.20

// Copyright (C) 2024 SUSE LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gocompat

import (
	"fmt"
)

type wrappedError struct {
	inner   error
	isError error
}

func (err wrappedError) Is(target error) bool {
	return err.isError == target
}

func (err wrappedError) Unwrap() error {
	return err.inner
}

func (err wrappedError) Error() string {
	return fmt.Sprintf("%v: %v", err.isError, err.inner)
}

// WrapBaseError is a helper that is equivalent to fmt.Errorf("%w: %w"), except
// that on pre-1.20 Go versions only errors.Is() works properly (errors.Unwrap)
// is only guaranteed to give you baseErr.
func WrapBaseError(baseErr, extraErr error) error {
	return wrappedError{
		inner:   baseErr,
		isError: extraErr,
	}
}
