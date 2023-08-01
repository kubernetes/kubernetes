/*
Copyright 2023 The Kubernetes Authors.

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

package cel

const (
	// PerCallLimit specify the actual cost limit per CEL validation call
	// current PerCallLimit gives roughly 0.1 second for each expression validation call
	PerCallLimit = 1000000

	// RuntimeCELCostBudget is the overall cost budget for runtime CEL validation cost per ValidatingAdmissionPolicyBinding or CustomResource
	// current RuntimeCELCostBudget gives roughly 1 seconds for the validation
	RuntimeCELCostBudget = 10000000

	// RuntimeCELCostBudgetMatchConditions is the overall cost budget for runtime CEL validation cost on matchConditions per object with matchConditions
	// this is per webhook for validatingwebhookconfigurations and mutatingwebhookconfigurations or per ValidatingAdmissionPolicyBinding
	// current RuntimeCELCostBudgetMatchConditions gives roughly 1/4 seconds for the validation
	RuntimeCELCostBudgetMatchConditions = 2500000

	// CheckFrequency configures the number of iterations within a comprehension to evaluate
	// before checking whether the function evaluation has been interrupted
	CheckFrequency = 100

	// MaxRequestSizeBytes is the maximum size of a request to the API server
	// TODO(DangerOnTheRanger): wire in MaxRequestBodyBytes from apiserver/pkg/server/options/server_run_options.go to make this configurable
	// Note that even if server_run_options.go becomes configurable in the future, this cost constant should be fixed and it should be the max allowed request size for the server
	MaxRequestSizeBytes = int64(3 * 1024 * 1024)

	// MaxEvaluatedMessageExpressionSizeBytes represents the largest-allowable string generated
	// by a messageExpression field
	MaxEvaluatedMessageExpressionSizeBytes = 5 * 1024
)
