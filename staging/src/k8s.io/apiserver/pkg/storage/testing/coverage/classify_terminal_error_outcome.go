/*
Copyright The Kubernetes Authors.

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

package coverage

import (
	"k8s.io/apiserver/pkg/storage"
)

// classifyTerminalErrorOutcome classifies the terminal error (or nil)
// returned by a storage.Interface call into an Outcome.
func classifyTerminalErrorOutcome(err error) Outcome {
	switch {
	case err == nil:
		return OutcomeSuccess
	case storage.IsNotFound(err):
		return OutcomeNotFound
	case storage.IsExist(err):
		return OutcomeExists
	case storage.IsConflict(err):
		return OutcomeConflict
	case storage.IsInvalidObj(err):
		return OutcomeInvalidObj
	case storage.IsUnreachable(err):
		return OutcomeUnreachable
	case storage.IsRequestTimeout(err):
		return OutcomeTimeout
	case storage.IsCorruptObject(err):
		return OutcomeCorruptObj
	default:
		return OutcomeOtherError
	}
}
