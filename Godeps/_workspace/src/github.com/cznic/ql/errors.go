// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"errors"
)

var (
	errBeginTransNoCtx          = errors.New("BEGIN TRANSACTION: Must use R/W context, have nil")
	errCommitNotInTransaction   = errors.New("COMMIT: Not in transaction")
	errDivByZero                = errors.New("division by zero")
	errIncompatibleDBFormat     = errors.New("incompatible DB format")
	errNoDataForHandle          = errors.New("read: no data for handle")
	errRollbackNotInTransaction = errors.New("ROLLBACK: Not in transaction")
)
