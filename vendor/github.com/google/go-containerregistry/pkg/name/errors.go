// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package name

import (
	"errors"
	"fmt"
)

// ErrBadName is an error for when a bad docker name is supplied.
type ErrBadName struct {
	info string
}

func (e *ErrBadName) Error() string {
	return e.info
}

// Is reports whether target is an error of type ErrBadName
func (e *ErrBadName) Is(target error) bool {
	var berr *ErrBadName
	return errors.As(target, &berr)
}

// newErrBadName returns a ErrBadName which returns the given formatted string from Error().
func newErrBadName(fmtStr string, args ...any) *ErrBadName {
	return &ErrBadName{fmt.Sprintf(fmtStr, args...)}
}

// IsErrBadName returns true if the given error is an ErrBadName.
//
// Deprecated: Use errors.Is.
func IsErrBadName(err error) bool {
	var berr *ErrBadName
	return errors.As(err, &berr)
}
