// SPDX-License-Identifier: MPL-2.0

//go:build linux

// Copyright (C) 2024-2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2024-2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Package internal contains unexported common code for filepath-securejoin.
package internal

import (
	"errors"

	"golang.org/x/sys/unix"
)

type xdevErrorish struct {
	description string
}

func (err xdevErrorish) Error() string        { return err.description }
func (err xdevErrorish) Is(target error) bool { return target == unix.EXDEV }

var (
	// ErrPossibleAttack indicates that some attack was detected.
	ErrPossibleAttack error = xdevErrorish{"possible attack detected"}

	// ErrPossibleBreakout indicates that during an operation we ended up in a
	// state that could be a breakout but we detected it.
	ErrPossibleBreakout error = xdevErrorish{"possible breakout detected"}

	// ErrInvalidDirectory indicates an unlinked directory.
	ErrInvalidDirectory = errors.New("wandered into deleted directory")

	// ErrDeletedInode indicates an unlinked file (non-directory).
	ErrDeletedInode = errors.New("cannot verify path of deleted inode")
)
