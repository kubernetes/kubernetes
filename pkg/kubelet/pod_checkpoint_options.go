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

package kubelet

import (
	"errors"
	"fmt"

	v1 "k8s.io/api/core/v1"
	nodeutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
)

func (kl *Kubelet) validatePodCheckpointOptions(pod *v1.Pod, options map[string]string, runtimeHandler string) error {
	if len(options) == 0 {
		return nil
	}
	if pod.Spec.RuntimeClassName == nil || *pod.Spec.RuntimeClassName == "" {
		return fmt.Errorf("checkpoint options require a RuntimeClass with allowedCheckpointOptions")
	}
	if kl.runtimeClassManager == nil {
		return fmt.Errorf("cannot validate checkpoint options without a RuntimeClass manager")
	}
	// Admission's policy may have been revoked since the checkpoint was created.
	// The cache can still lag a revocation, as can any read before the runtime
	// call, so a live read would only narrow that window.
	runtimeClass, err := kl.runtimeClassManager.GetRuntimeClass(*pod.Spec.RuntimeClassName)
	if errors.Is(err, runtimeclass.ErrNotSynced) {
		return fmt.Errorf("%w: get RuntimeClass %q: %w", errPodCheckpointPolicyUnavailable, *pod.Spec.RuntimeClassName, err)
	}
	if err != nil {
		return fmt.Errorf("get RuntimeClass %q for checkpoint options: %w", *pod.Spec.RuntimeClassName, err)
	}
	// A deleted and recreated RuntimeClass must not grant its new handler's
	// options to a sandbox that still runs under the previous handler.
	if runtimeClass.Handler != runtimeHandler {
		return fmt.Errorf("RuntimeClass %q handler %q does not match source sandbox handler %q", runtimeClass.Name, runtimeClass.Handler, runtimeHandler)
	}
	var allowed []string
	if runtimeClass.PodCheckpointOptions != nil {
		allowed = runtimeClass.PodCheckpointOptions.AllowedCheckpointOptions
	}
	return nodeutil.ValidateRuntimeOptions(options, allowed)
}
