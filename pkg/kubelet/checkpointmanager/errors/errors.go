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

package errors

import "fmt"

// ErrCorruptCheckpoint error is reported when checksum does not match.
// Check for it with:
//
//	var csErr *CorruptCheckpointError
//	if errors.As(err, &csErr) { ... }
//	if errors.Is(err, CorruptCheckpointError{}) { ... }
type CorruptCheckpointError struct {
	ActualCS, ExpectedCS uint64
}

func (err CorruptCheckpointError) Error() string {
	return "checkpoint is corrupted"
}

func (err CorruptCheckpointError) Is(target error) bool {
	switch target.(type) {
	case *CorruptCheckpointError, CorruptCheckpointError:
		return true
	default:
		return false
	}
}

// ErrCheckpointNotFound is reported when checkpoint is not found for a given key
var ErrCheckpointNotFound = fmt.Errorf("checkpoint is not found")
