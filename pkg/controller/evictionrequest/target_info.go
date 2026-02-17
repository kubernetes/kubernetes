/*
Copyright 2026 The Kubernetes Authors.

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
	"time"

	coordinationv1alpha1 "k8s.io/api/coordination/v1alpha1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	corelisters "k8s.io/client-go/listers/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

// targetInfo abstracts over the eviction target so that callers can ask
// semantic questions ("is the target valid?", "is it gone?") rather than
// reaching into pod-specific fields. It wraps both the spec-level target
// reference and the looked-up runtime object.
//
// When Workload targets are added to EvictionTarget, each method gains a
// second branch instead of every caller changing.
type targetInfo struct {
	spec coordinationv1alpha1.EvictionTarget
	pod  *v1.Pod
	err  error
}

// newTargetInfo looks up the target for the given EvictionRequest and returns
// a targetInfo that wraps the result. The caller must check the returned error
// for transient lookup failures; not-found errors are captured inside
// targetInfo and surfaced via isGone() / isValidTarget().
func newTargetInfo(evictionRequest *coordinationv1alpha1.EvictionRequest, podLister corelisters.PodLister) (targetInfo, error) {
	t := targetInfo{spec: evictionRequest.Spec.Target}
	if t.spec.Pod != nil {
		pod, err := podLister.Pods(evictionRequest.Namespace).Get(t.spec.Pod.Name)
		if err != nil && !errors.IsNotFound(err) {
			return targetInfo{}, err
		}
		t.pod = pod
		t.err = err
	}
	return t, nil
}

// isGone reports whether the original target no longer exists. This is true
// when the target is not found or when the found object has a different UID
// (i.e., the original was deleted and a new object with the same name was created).
func (t targetInfo) isGone() bool {
	switch {
	case t.spec.Pod != nil:
		if t.pod == nil {
			return errors.IsNotFound(t.err)
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
		if t.pod == nil && errors.IsNotFound(t.err) {
			return false, "Target pod not found"
		}
		if t.pod != nil && string(t.pod.UID) != t.spec.Pod.UID {
			return false, fmt.Sprintf("Pod UID mismatch: expected %s, got %s", t.spec.Pod.UID, string(t.pod.UID))
		}
		if t.pod != nil && t.pod.Spec.WorkloadRef != nil {
			return false, fmt.Sprintf("Target pod %s is part of a Workload. Eviction of such pods is currently not supported.", t.pod.Name)
		}
	}
	return true, ""
}

// isTerminal reports whether the target has reached a terminal lifecycle state.
func (t targetInfo) isTerminal() bool {
	switch {
	case t.pod != nil:
		return podutil.IsPodTerminal(t.pod)
	}
	return false
}

// name returns the found target's name, or "" if unavailable.
func (t targetInfo) name() string {
	switch {
	case t.pod != nil:
		return t.pod.Name
	}
	return ""
}

// deletionTimestamp returns the target's deletion timestamp, or nil if the
// target is unavailable or has not been deleted.
func (t targetInfo) deletionTimestamp() time.Time {
	switch {
	case t.pod != nil:
		timestamp := t.pod.DeletionTimestamp
		if timestamp != nil {
			return timestamp.Time
		}
	}
	return time.Time{}
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
