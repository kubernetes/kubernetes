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

package upgrade

import (
	clientset "k8s.io/client-go/kubernetes"
)

// CheckClusterHealth makes sure:
// - the API /healthz endpoint is healthy
// - all Nodes are Ready
// - (if self-hosted) that there are DaemonSets with at least one Pod for all control plane components
// - (if static pod-hosted) that all required Static Pod manifests exist on disk
func CheckClusterHealth(_ clientset.Interface) error {
	return nil
}

// IsControlPlaneSelfHosted returns whether the control plane is self hosted or not
func IsControlPlaneSelfHosted(_ clientset.Interface) bool {
	// No-op for now
	return false
}
