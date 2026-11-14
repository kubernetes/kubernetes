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

package podcheckpoint

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	coreclient "k8s.io/client-go/kubernetes/typed/core/v1"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
)

type createStrategy struct {
	podCheckpointStrategy
	pods coreclient.PodsGetter
}

// NewCreateStrategy validates references against current API state. Update
// strategies must not require the source Pod to remain available.
func NewCreateStrategy(pods coreclient.PodsGetter) *createStrategy {
	return &createStrategy{Strategy, pods}
}

func (s *createStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	errs := s.podCheckpointStrategy.Validate(ctx, obj)
	if err := ctx.Err(); err != nil {
		return append(errs, field.InternalError(field.NewPath("spec", "sourcePod"), err))
	}
	pc := obj.(*nodeapi.PodCheckpoint)
	path := field.NewPath("spec", "sourcePod")
	if pc.Spec.SourcePod.Name == "" {
		return errs
	}
	pod, err := s.pods.Pods(pc.Namespace).Get(ctx, pc.Spec.SourcePod.Name, metav1.GetOptions{})
	if err != nil {
		if !apierrors.IsNotFound(err) {
			return append(errs, field.InternalError(path, fmt.Errorf("cannot read source Pod %q: %w", pc.Spec.SourcePod.Name, err)))
		}
		return append(errs, field.Forbidden(path, fmt.Sprintf("cannot read source Pod %q: %v", pc.Spec.SourcePod.Name, err)))
	}
	if err := validateSourcePod(pod, &pc.Spec.SourcePod); err != nil {
		return append(errs, field.Forbidden(path, err.Error()))
	}
	return errs
}

// validateSourcePod checks that the source kubelet can checkpoint pod.
func validateSourcePod(pod *corev1.Pod, ref *nodeapi.PodReference) error {
	if _, mirror := pod.Annotations[corev1.MirrorPodAnnotationKey]; mirror {
		return fmt.Errorf("source Pod %q is a mirror Pod; checkpointing static Pods is not supported", ref.Name)
	}
	if ref.UID != nil && *ref.UID != pod.UID {
		return fmt.Errorf("source Pod %q has UID %q, not requested UID %q", ref.Name, pod.UID, *ref.UID)
	}
	if pod.DeletionTimestamp != nil {
		return fmt.Errorf("source Pod %q is being deleted", ref.Name)
	}
	if pod.Spec.NodeName == "" || pod.Status.Phase != corev1.PodRunning {
		return fmt.Errorf("source Pod %q must be assigned to a node and Running before checkpointing", ref.Name)
	}
	return nil
}
