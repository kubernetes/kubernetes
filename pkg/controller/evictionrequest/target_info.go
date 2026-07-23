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
	"time"

	v1 "k8s.io/api/core/v1"
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
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
	if e == podTarget {
		return "pod"
	}
	return "unknown"

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
	if spec.Pod != nil {
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
	if t.target.Pod != nil {
		return podTarget
	}
	return noTarget
}

// targetType returns the type of the eviction target (e.g. "pod").
func (t targetInfo) targetName() string {
	if t.target.Pod != nil {
		return t.target.Pod.Name
	}
	return ""
}

// targetType returns the type of the eviction target (e.g. "pod").
func (t targetInfo) targetUID() apimachinerytypes.UID {
	if t.target.Pod != nil {
		return t.target.Pod.UID
	}
	return ""
}

// targetFoundByName returns true if the target object has been found
func (t targetInfo) targetFoundByName() bool {
	if t.target.Pod != nil {
		return t.pod != nil
	}
	return false
}

// GetObjectMeta returns the target's ObjectMeta, or nil if the target is unavailable.
func (t targetInfo) GetObjectMeta() metav1.Object {
	if t.target.Pod != nil {
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
	if t.target.Pod != nil {
		if t.pod == nil {
			return true
		}
		return t.pod.UID != t.target.Pod.UID
	}
	return false
}

// hasSchedulingGroup reports whether the target belongs to a PodGroup.
func (t targetInfo) hasSchedulingGroup() bool {
	if t.pod != nil {
		return t.pod.Spec.SchedulingGroup != nil
	}
	return false
}

// hasCompleted reports whether the target has reached a terminal lifecycle state.
func (t targetInfo) isTerminal() bool {
	if t.pod != nil {
		return podutil.IsPodTerminal(t.pod)
	}
	return false
}

// terminalTime it reports the latest time this target has became terminal.
func (t targetInfo) terminalTime() *time.Time {
	var terminalTime *time.Time
	if t.pod != nil && t.isTerminal() {
		terminalTime = getPodFinishTime(t.pod)
	}
	objectMeta := t.GetObjectMeta()
	if objectMeta == nil || objectMeta.GetDeletionTimestamp() == nil {
		return terminalTime
	}
	if terminalTime == nil || objectMeta.GetDeletionTimestamp().Time.After(*terminalTime) {
		return new(objectMeta.GetDeletionTimestamp().Time)
	}
	return terminalTime
}

// evictionResponders returns the responders declared on the target, or nil
// if the target is unavailable.
func (t targetInfo) evictionResponders(includeDefault bool) []v1.EvictionResponder {
	if t.pod != nil {
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
	if t.target.Pod != nil {
		return lifecycleapply.EvictionTarget().WithPod(
			lifecycleapply.EvictionPodReference().
				WithName(t.target.Pod.Name).
				WithUID(t.target.Pod.UID),
		)
	}
	return nil
}

// getPodFinishTime returns the pod finish time using
// Similar tests in k8s.io/pkg/controller/job.getFinishedTime.
func getPodFinishTime(p *v1.Pod) *time.Time {
	finishTime := latestFinishTimeFromContainers(nil, p.Status.ContainerStatuses, nil)
	// We need to check InitContainerStatuses here also,
	// because with the sidecar (restartable init) containers,
	// sidecar containers will always finish later than regular containers.
	names := sets.New[string]()
	for _, c := range p.Spec.InitContainers {
		if c.RestartPolicy != nil && *c.RestartPolicy == v1.ContainerRestartPolicyAlways {
			names.Insert(c.Name)
		}
	}
	finishTime = latestFinishTimeFromContainers(finishTime, p.Status.InitContainerStatuses, func(status v1.ContainerStatus) bool {
		return names.Has(status.Name)
	})

	return finishTime
}

func latestFinishTimeFromContainers(prevFinishTime *time.Time, cs []v1.ContainerStatus, check func(status v1.ContainerStatus) bool) *time.Time {
	var finishTime = prevFinishTime
	for _, containerState := range cs {
		if check != nil && !check(containerState) {
			continue
		}
		if containerState.State.Terminated == nil ||
			containerState.State.Terminated.FinishedAt.Time.IsZero() {
			return nil
		}
		if finishTime == nil || finishTime.Before(containerState.State.Terminated.FinishedAt.Time) {
			finishTime = &containerState.State.Terminated.FinishedAt.Time
		}
	}
	return finishTime
}
