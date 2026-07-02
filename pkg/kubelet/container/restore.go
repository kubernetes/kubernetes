/*
Copyright 2026 The Kubernetes Authors.

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

package container

import "errors"

// RestoreError wraps an error encountered while restoring a pod from a
// PodCheckpoint together with a stable Reason suitable for use as a Kubernetes
// event reason (e.g. "CheckpointNotReady", "CheckpointWrongNode",
// "PodSpecMismatch", "RestoreInProgress"). Callers that surface restore
// failures as events can extract the reason with RestoreErrorReason.
type RestoreError struct {
	// Reason is a stable, machine-readable event reason.
	Reason string
	// Err is the underlying error.
	Err error
}

func (e *RestoreError) Error() string {
	if e.Err == nil {
		return e.Reason
	}
	return e.Err.Error()
}

func (e *RestoreError) Unwrap() error {
	return e.Err
}

// RestoreErrorReason returns the Reason of the first RestoreError in err's
// chain, or "" if there is none.
func RestoreErrorReason(err error) string {
	var re *RestoreError
	if errors.As(err, &re) {
		return re.Reason
	}
	return ""
}
