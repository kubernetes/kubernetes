/*
Copyright 2024 The Kubernetes Authors.

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

package validating

import (
	"strings"

	celmetrics "k8s.io/apiserver/pkg/admission/plugin/policy/validating/metrics"
)

// ErrorType decodes the error to determine the error type
// that the metrics understand.
func ErrorType(decision *PolicyDecision) celmetrics.ValidationErrorType {
	if decision.Evaluation == EvalAdmit {
		return celmetrics.ValidationNoError
	}
	if strings.HasPrefix(decision.Message, "compilation") {
		return celmetrics.ValidationCompileError
	}
	if strings.HasPrefix(decision.Message, "validation failed due to running out of cost budget") {
		return celmetrics.ValidatingOutOfBudget
	}
	return celmetrics.ValidatingInternalError
}
