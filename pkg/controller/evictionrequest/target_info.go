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

package evictionrequest

import (
	"fmt"

	coordinationv1alpha1 "k8s.io/api/coordination/v1alpha1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

var _ metav1.ObjectMetaAccessor = &targetInfo{}

// targetInfo abstracts over the eviction target so that callers can ask
// semantic questions ("is the target valid?", "is it gone?") rather than
// reaching into pod-specific fields.
type targetInfo struct {
	spec coordinationv1alpha1.EvictionTarget
	pod  *v1.Pod
}

// newTargetInfo creates a targetInfo from the resolved target.
// The caller is responsible for looking up the target object beforehand;
// a nil pod with a non-nil spec.Pod means the pod was not found.
func newTargetInfo(spec coordinationv1alpha1.EvictionTarget, pod *v1.Pod) targetInfo {
	return targetInfo{spec: spec, pod: pod}
}

// GetObjectMeta returns the target's ObjectMeta, or nil if the target is unavailable.
func (t targetInfo) GetObjectMeta() metav1.Object {
	switch {
	case t.spec.Pod != nil:
		if t.pod != nil {
			return t.pod
		}
	}
	return nil
}

// isGone reports whether the original target no longer exists. This is true
// when the target is not found or when the found object has a different UID
// (i.e., the original was deleted and a new object with the same name was created).
func (t targetInfo) isGone() bool {
	switch {
	case t.spec.Pod != nil:
		if t.pod == nil {
			return true
		}
		return string(t.pod.UID) != t.spec.Pod.UID
	}
	return false
}

// isValidTarget checks all preconditions for processing an eviction target.
// It returns true if the target is valid. If invalid, it returns false and a
// message describing why validation failed.
func (t targetInfo) isValidTarget() (bool, string) {
	switch {
	case t.spec.Pod != nil:
		if t.pod == nil {
			return false, "Target pod not found"
		}
		if string(t.pod.UID) != t.spec.Pod.UID {
			return false, fmt.Sprintf("Pod UID mismatch: expected %s, got %s", t.spec.Pod.UID, string(t.pod.UID))
		}
		if t.hasWorkloadRef() {
			return false, fmt.Sprintf("Target pod %s is part of a Workload. Eviction of such pods is currently not supported.", t.pod.Name)
		}
	}
	return true, ""
}

// hasWorkloadRef reports whether the target belongs to a Workload.
func (t targetInfo) hasWorkloadRef() bool {
	switch {
	case t.pod != nil:
		return t.pod.Spec.WorkloadRef != nil
	}
	return false
}

// isTerminal reports whether the target has reached a terminal lifecycle state.
func (t targetInfo) isTerminal() bool {
	switch {
	case t.pod != nil:
		return podutil.IsPodTerminal(t.pod)
	}
	return false
}

// targetType returns the type of the eviction target (e.g. "pod").
func (t targetInfo) targetType() string {
	switch {
	case t.spec.Pod != nil:
		return "pod"
	}
	return "unknown"
}

// evictionInterceptors returns the interceptors declared on the target, or nil
// if the target is unavailable.
func (t targetInfo) evictionInterceptors() []v1.EvictionInterceptor {
	switch {
	case t.pod != nil:
		return t.pod.Spec.EvictionInterceptors
	}
	return nil
}
