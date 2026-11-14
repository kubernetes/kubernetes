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

package types

import "errors"

// StageRecoveryRequiredError asks the volume operation to retry device staging.
// It preserves the original publication error and its retry classification.
type StageRecoveryRequiredError struct{ err error }

func NewStageRecoveryRequiredError(err error) *StageRecoveryRequiredError {
	return &StageRecoveryRequiredError{err: err}
}
func (e *StageRecoveryRequiredError) Error() string {
	return "CSI stage recovery required: " + e.err.Error()
}
func (e *StageRecoveryRequiredError) Unwrap() error { return e.err }
func IsStageRecoveryRequiredError(err error) bool {
	var recovery *StageRecoveryRequiredError
	return errors.As(err, &recovery)
}
