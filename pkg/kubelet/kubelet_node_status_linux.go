/*
Copyright 2025 The Kubernetes Authors.

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

package kubelet

import (
	"github.com/moby/sys/userns"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func getOSSpecificLabels() (map[string]string, error) {
	return nil, nil
}

// runningInUserNS returns a pointer to true if the Kubelet is running in a user namespace.
func (kl *Kubelet) runningInUserNS() *bool {
	if !utilfeature.DefaultFeatureGate.Enabled(features.KubeletInUserNamespace) {
		return nil
	}
	return ptr.To(userns.RunningInUserNS())
}
