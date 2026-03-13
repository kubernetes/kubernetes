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

package config

// ValidatingAdmissionPolicyStatusControllerConfiguration contains elements describing ValidatingAdmissionPolicyStatusController.
type ValidatingAdmissionPolicyStatusControllerConfiguration struct {
	// ConcurrentPolicySyncs is the number of policy objects that are
	// allowed to sync concurrently. Larger number = quicker type checking,
	// but more CPU (and network) load.
	// The default value is 5.
	ConcurrentPolicySyncs int32
}
