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

package kuberuntime

import (
	"context"

	v1 "k8s.io/api/core/v1"
	ref "k8s.io/client-go/tools/reference"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// EnsureSecurityProfiles pulls every OCI security profile referenced by the
// pod, once per unique reference. It runs on every pod sync, so that a
// profile evicted by the runtime is pulled again before containers that use
// it are created.
func (m *kubeGenericRuntimeManager) EnsureSecurityProfiles(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus, pullSecrets []v1.Secret) error {
	refs := kubecontainer.SecurityProfileOCIRefs(pod)
	if len(refs) == 0 {
		return nil
	}
	logger := klog.FromContext(ctx)

	var attempt uint32
	if podStatus != nil && len(podStatus.SandboxStatuses) > 0 {
		attempt = podStatus.SandboxStatuses[0].GetMetadata().GetAttempt()
	}
	// Pulls only need the identity of the pod, not the full sandbox config,
	// which this avoids generating on every sync.
	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Uid:       string(pod.UID),
			Attempt:   attempt,
		},
		Labels:      newPodLabels(pod),
		Annotations: newPodAnnotations(pod),
	}
	podRuntimeHandler, err := m.getPodRuntimeHandler(pod)
	if err != nil {
		return err
	}
	objRef, err := ref.GetReference(legacyscheme.Scheme, pod)
	if err != nil {
		// Events are best effort; pull without them.
		logger.V(4).Info("Failed to get pod reference for security profile events", "pod", klog.KObj(pod), "err", err)
		objRef = nil
	}

	for _, profileRef := range refs {
		if err := m.imagePuller.EnsureSecurityProfile(ctx, objRef, pod, profileRef, runtimeapi.SecurityProfileKind_Seccomp, pullSecrets, podSandboxConfig, podRuntimeHandler); err != nil {
			return err
		}
	}
	return nil
}
