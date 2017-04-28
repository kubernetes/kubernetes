/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package mergepatch

import (
	"errors"
	"fmt"
)

var (
	ErrBadJSONDoc                     = errors.New("Invalid JSON document")
	ErrNoListOfLists                  = errors.New("Lists of lists are not supported")
	ErrBadPatchFormatForPrimitiveList = errors.New("Invalid patch format of primitive list")
)

func ErrNoMergeKey(m map[string]interface{}, k string) error {
	return fmt.Errorf("map: %v does not contain declared merge key: %s", m, k)
}

func ErrBadArgType(expected, actual string) error {
	return fmt.Errorf("expected a %s, but received a %s", expected, actual)
}

func ErrBadPatchType(t interface{}, m map[string]interface{}) error {
	return fmt.Errorf("unknown patch type: %s in map: %v", t, m)
}

// IsPreconditionFailed returns true if the provided error indicates
// a precondition failed.
func IsPreconditionFailed(err error) bool {
	_, ok := err.(ErrPreconditionFailed)
	return ok
}

type ErrPreconditionFailed struct {
	message string
}

func NewErrPreconditionFailed(target map[string]interface{}) ErrPreconditionFailed {
	s := fmt.Sprintf("precondition failed for: %v", target)
	return ErrPreconditionFailed{s}
}

func (err ErrPreconditionFailed) Error() string {
	return err.message
}

type ErrConflict struct {
	message string
}

func NewErrConflict(patch, current string) ErrConflict {
	s := fmt.Sprintf("patch:\n%s\nconflicts with changes made from original to current:\n%s\n", patch, current)
	return ErrConflict{s}
}

func (err ErrConflict) Error() string {
	return err.message
}

// IsConflict returns true if the provided error indicates
// a conflict between the patch and the current configuration.
func IsConflict(err error) bool {
	_, ok := err.(ErrConflict)
	return ok
}
