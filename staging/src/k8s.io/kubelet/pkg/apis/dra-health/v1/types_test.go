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

package v1

import (
	"testing"

	v1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
)

// TestServiceNameConstants verifies that the service name constants match
// the service names declared in the generated gRPC code. The constants are
// what plugins advertise during registration and what the kubelet matches
// against, so they must stay in sync with the gRPC service definitions.
func TestServiceNameConstants(t *testing.T) {
	if DRAResourceHealthService != DRAResourceHealth_ServiceDesc.ServiceName {
		t.Errorf("v1.DRAResourceHealthService = %q, want generated service name %q", DRAResourceHealthService, DRAResourceHealth_ServiceDesc.ServiceName)
	}
	if v1alpha1.DRAResourceHealthService != v1alpha1.DRAResourceHealth_ServiceDesc.ServiceName {
		t.Errorf("v1alpha1.DRAResourceHealthService = %q, want generated service name %q", v1alpha1.DRAResourceHealthService, v1alpha1.DRAResourceHealth_ServiceDesc.ServiceName)
	}
}
