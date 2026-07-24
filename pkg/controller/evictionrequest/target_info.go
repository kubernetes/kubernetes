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
	v1 "k8s.io/api/core/v1"
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
	lifecycleapply "k8s.io/client-go/applyconfigurations/lifecycle/v1alpha1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

var _ metav1.ObjectMetaAccessor = &targetInfo{}

type targetType int

const (
	noTarget targetType = iota
	podTarget
)

func (e targetType) String() string {
	switch e {
	case podTarget:
		return "pod"
	default:
		return "unknown"
	}
}

// targetInfo abstracts over the eviction target so that callers can ask
// semantic questions ("is the target valid?", "is it gone?") rather than
// reaching into pod-specific fields.
type targetInfo struct {
	target lifecyclev1alpha1.EvictionRequestTarget
	pod    *v1.Pod
}

// newTargetInfo creates a targetInfo from the resolved target.
// The caller is responsible for looking up the target object beforehand;
// a nil pod with a non-nil spec.Pod means the pod was not found.
func newTargetInfoForEviction(spec lifecyclev1alpha1.EvictionTarget, pod *v1.Pod) targetInfo {
	switch {
	case spec.Pod != nil:
		return targetInfo{target: lifecyclev1alpha1.EvictionRequestTarget{
			Pod: &lifecyclev1alpha1.EvictionRequestPodReference{
				Name: spec.Pod.Name,
				UID:  spec.Pod.UID,
			},
		}, pod: pod}
	}
	return targetInfo{pod: pod}
}

// newTargetInfo creates a targetInfo from the resolved target.
// The caller is responsible for looking up the target object beforehand;
// a nil pod with a non-nil spec.Pod means the pod was not found.
func newTargetInfoForEvictionRequest(spec lifecyclev1alpha1.EvictionRequestTarget, pod *v1.Pod) targetInfo {
	return targetInfo{target: spec, pod: pod}
}

// targetType returns the type of the eviction target (e.g. "pod").
func (t targetInfo) targetType() targetType {
	switch {
	case t.target.Pod != nil:
		return podTarget
	}
	return noTarget
}

// targetType returns the type of the eviction target (e.g. "pod").
func (t targetInfo) targetName() string {
	switch {
	case t.target.Pod != nil:
		return t.target.Pod.Name
	}
	return ""
}

// targetType returns the type of the eviction target (e.g. "pod").
func (t targetInfo) targetUID() apimachinerytypes.UID {
	switch {
	case t.target.Pod != nil:
		return t.target.Pod.UID
	}
	return ""
}

// targetFoundByName returns true if the target object has been found
func (t targetInfo) targetFoundByName() bool {
	switch {
	case t.target.Pod != nil:
		return t.pod != nil
	}
	return false
}

// GetObjectMeta returns the target's ObjectMeta, or nil if the target is unavailable.
func (t targetInfo) GetObjectMeta() metav1.Object {
	switch {
	case t.target.Pod != nil:
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
	case t.target.Pod != nil:
		if t.pod == nil {
			return true
		}
		return t.pod.UID != t.target.Pod.UID
	}
	return false
}

// hasSchedulingGroup reports whether the target belongs to a PodGroup.
func (t targetInfo) hasSchedulingGroup() bool {
	switch {
	case t.pod != nil:
		return t.pod.Spec.SchedulingGroup != nil
	}
	return false
}

// hasCompleted reports whether the target has reached a terminal lifecycle state.
func (t targetInfo) isTerminal() bool {
	switch {
	case t.pod != nil:
		return podutil.IsPodTerminal(t.pod)
	}
	return false
}

// evictionResponders returns the responders declared on the target, or nil
// if the target is unavailable.
func (t targetInfo) evictionResponders(includeDefault bool) []v1.EvictionResponder {
	switch {
	case t.pod != nil:
		responders := append([]v1.EvictionResponder(nil), t.pod.Spec.EvictionResponders...)
		if includeDefault {
			// Default imperative-eviction responder triggers imperative pod /eviction endpoint
			responders = append(responders, v1.EvictionResponder{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100))})
		}
		return responders
	}
	return nil
}

func (t targetInfo) toEvictionTargetApply() *lifecycleapply.EvictionTargetApplyConfiguration {
	switch {
	case t.target.Pod != nil:
		return lifecycleapply.EvictionTarget().WithPod(
			lifecycleapply.EvictionPodReference().
				WithName(t.target.Pod.Name).
				WithUID(t.target.Pod.UID),
		)
	}
	return nil
}
