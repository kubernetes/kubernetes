// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package utils

import (
	"fmt"
	"time"

	"github.com/pkg/errors"
)

type errTimeOut struct {
	duration time.Duration
	cmd      string
}

func NewErrTimeOut(d time.Duration, c string) errTimeOut {
	return errTimeOut{duration: d, cmd: c}
}

func (e errTimeOut) Error() string {
	return fmt.Sprintf("hit %s timeout running '%s'", e.duration, e.cmd)
}

func IsErrTimeout(err error) bool {
	if err == nil {
		return false
	}
	_, ok := err.(errTimeOut)
	if ok {
		return true
	}
	_, ok = errors.Cause(err).(errTimeOut)
	return ok
}
