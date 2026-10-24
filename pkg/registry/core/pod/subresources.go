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

package pod

import "k8s.io/apimachinery/pkg/util/sets"

// DisableablePodSubresources returns the pod subresources that operators can
// disable via --disable-pod-subresources without breaking core pod lifecycle.
func DisableablePodSubresources() sets.Set[string] {
	return sets.New(
		"attach",
		"ephemeralcontainers",
		"eviction",
		"exec",
		"log",
		"portforward",
		"proxy",
		"resize",
	)
}

// NonDisableablePodSubresources returns the pod subresources required for core
// pod scheduling and kubelet status reporting, which --disable-pod-subresources
// must never disable.
func NonDisableablePodSubresources() sets.Set[string] {
	return sets.New(
		"binding",
		"status",
	)
}
