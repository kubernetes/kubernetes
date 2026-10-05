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
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	nodeutil "k8s.io/kubernetes/pkg/apis/node/util"
)

func (kl *Kubelet) validatePodCheckpointOptions(ctx context.Context, pod *v1.Pod, options map[string]string, runtimeHandler string) error {
	if len(options) == 0 {
		return nil
	}
	if pod.Spec.RuntimeClassName == nil || *pod.Spec.RuntimeClassName == "" {
		return fmt.Errorf("checkpoint options require a RuntimeClass with allowedCheckpointOptions")
	}
	if kl.kubeClient == nil {
		return fmt.Errorf("cannot validate checkpoint options without an API client")
	}
	// Admission's policy may have been revoked since the checkpoint was created.
	runtimeClass, err := kl.kubeClient.NodeV1().RuntimeClasses().Get(ctx, *pod.Spec.RuntimeClassName, metav1.GetOptions{})
	if err != nil {
		if nodeutil.RuntimeOptionPolicyUnavailable(err) {
			return fmt.Errorf("%w: get RuntimeClass %q: %w", errPodCheckpointPolicyUnavailable, *pod.Spec.RuntimeClassName, err)
		}
		return fmt.Errorf("get RuntimeClass %q for checkpoint options: %w", *pod.Spec.RuntimeClassName, err)
	}
	// A deleted and recreated RuntimeClass must not grant its new handler's
	// options to a sandbox that still runs under the previous handler.
	if runtimeClass.Handler != runtimeHandler {
		return fmt.Errorf("RuntimeClass %q handler %q does not match source sandbox handler %q", runtimeClass.Name, runtimeClass.Handler, runtimeHandler)
	}
	var allowed []string
	if runtimeClass.PodCheckpoint != nil {
		allowed = runtimeClass.PodCheckpoint.AllowedCheckpointOptions
	}
	return nodeutil.ValidateRuntimeOptions(options, allowed)
}
