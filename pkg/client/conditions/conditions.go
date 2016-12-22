/*
Copyright 2014 The Kubernetes Authors.

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

package conditions

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/watch"
)

// ErrPodCompleted is returned by PodRunning or PodContainerRunning to indicate that
// the pod has already reached completed state.
var ErrPodCompleted = fmt.Errorf("pod ran to completion")

// ErrContainerTerminated is returned by PodContainerRunning in the intermediate
// state where the pod indicates it's still running, but its container is already terminated
var ErrContainerTerminated = fmt.Errorf("container terminated")

// PodRunning returns true if the pod is running, false if the pod has not yet reached running state,
// returns ErrPodCompleted if the pod has run to completion, or an error in any other case.
func PodRunning(event watch.Event) (bool, error) {
	switch event.Type {
	case watch.Deleted:
		return false, errors.NewNotFound(schema.GroupResource{Resource: "pods"}, "")
	}
	switch t := event.Object.(type) {
	case *v1.Pod:
		switch t.Status.Phase {
		case v1.PodRunning:
			return true, nil
		case v1.PodFailed, v1.PodSucceeded:
			return false, ErrPodCompleted
		}
	}
	return false, nil
}

// PodCompleted returns true if the pod has run to completion, false if the pod has not yet
// reached running state, or an error in any other case.
func PodCompleted(event watch.Event) (bool, error) {
	switch event.Type {
	case watch.Deleted:
		return false, errors.NewNotFound(schema.GroupResource{Resource: "pods"}, "")
	}
	switch t := event.Object.(type) {
	case *v1.Pod:
		switch t.Status.Phase {
		case v1.PodFailed, v1.PodSucceeded:
			return true, nil
		}
	}
	return false, nil
}

// PodRunningAndReady returns true if the pod is running and ready, false if the pod has not
// yet reached those states, returns ErrPodCompleted if the pod has run to completion, or
// an error in any other case.
func PodRunningAndReady(event watch.Event) (bool, error) {
	switch event.Type {
	case watch.Deleted:
		return false, errors.NewNotFound(schema.GroupResource{Resource: "pods"}, "")
	}
	switch t := event.Object.(type) {
	case *v1.Pod:
		switch t.Status.Phase {
		case v1.PodFailed, v1.PodSucceeded:
			return false, ErrPodCompleted
		case v1.PodRunning:
			return v1.IsPodReady(t), nil
		}
	}
	return false, nil
}

// PodNotPending returns true if the pod has left the pending state, false if it has not,
// or an error in any other case (such as if the pod was deleted).
func PodNotPending(event watch.Event) (bool, error) {
	switch event.Type {
	case watch.Deleted:
		return false, errors.NewNotFound(schema.GroupResource{Resource: "pods"}, "")
	}
	switch t := event.Object.(type) {
	case *v1.Pod:
		switch t.Status.Phase {
		case v1.PodPending:
			return false, nil
		default:
			return true, nil
		}
	}
	return false, nil
}

// PodContainerRunning returns false until the named container has ContainerStatus running (at least once),
// and will return an error if the pod is deleted, runs to completion, or the container pod is not available.
func PodContainerRunning(containerName string) watch.ConditionFunc {
	return func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Deleted:
			return false, errors.NewNotFound(schema.GroupResource{Resource: "pods"}, "")
		}
		switch t := event.Object.(type) {
		case *v1.Pod:
			switch t.Status.Phase {
			case v1.PodRunning, v1.PodPending:
			case v1.PodFailed, v1.PodSucceeded:
				return false, ErrPodCompleted
			default:
				return false, nil
			}
			for _, s := range t.Status.ContainerStatuses {
				if s.Name != containerName {
					continue
				}
				if s.State.Terminated != nil {
					return false, ErrContainerTerminated
				}
				return s.State.Running != nil, nil
			}
			for _, s := range t.Status.InitContainerStatuses {
				if s.Name != containerName {
					continue
				}
				if s.State.Terminated != nil {
					return false, ErrContainerTerminated
				}
				return s.State.Running != nil, nil
			}
			return false, nil
		}
		return false, nil
	}
}

// ServiceAccountHasSecrets returns true if the service account has at least one secret,
// false if it does not, or an error.
func ServiceAccountHasSecrets(event watch.Event) (bool, error) {
	switch event.Type {
	case watch.Deleted:
		return false, errors.NewNotFound(schema.GroupResource{Resource: "serviceaccounts"}, "")
	}
	switch t := event.Object.(type) {
	case *v1.ServiceAccount:
		return len(t.Secrets) > 0, nil
	}
	return false, nil
}
