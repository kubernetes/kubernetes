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

package unversioned

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"
)

// ControllerHasDesiredReplicas returns a condition that will be true if and only if
// the desired replica count for a controller's ReplicaSelector equals the Replicas count.
func ControllerHasDesiredReplicas(c Interface, controller *api.ReplicationController) wait.ConditionFunc {

	// If we're given a controller where the status lags the spec, it either means that the controller is stale,
	// or that the rc manager hasn't noticed the update yet. Polling status.Replicas is not safe in the latter case.
	desiredGeneration := controller.Generation

	return func() (bool, error) {
		ctrl, err := c.ReplicationControllers(controller.Namespace).Get(controller.Name)
		if err != nil {
			return false, err
		}
		// There's a chance a concurrent update modifies the Spec.Replicas causing this check to pass,
		// or, after this check has passed, a modification causes the rc manager to create more pods.
		// This will not be an issue once we've implemented graceful delete for rcs, but till then
		// concurrent stop operations on the same rc might have unintended side effects.
		return ctrl.Status.ObservedGeneration >= desiredGeneration && ctrl.Status.Replicas == ctrl.Spec.Replicas, nil
	}
}

// ReplicaSetHasDesiredReplicas returns a condition that will be true if and only if
// the desired replica count for a ReplicaSet's ReplicaSelector equals the Replicas count.
func ReplicaSetHasDesiredReplicas(c ExtensionsInterface, replicaSet *extensions.ReplicaSet) wait.ConditionFunc {

	// If we're given a ReplicaSet where the status lags the spec, it either means that the
	// ReplicaSet is stale, or that the ReplicaSet manager hasn't noticed the update yet.
	// Polling status.Replicas is not safe in the latter case.
	desiredGeneration := replicaSet.Generation

	return func() (bool, error) {
		rs, err := c.ReplicaSets(replicaSet.Namespace).Get(replicaSet.Name)
		if err != nil {
			return false, err
		}
		// There's a chance a concurrent update modifies the Spec.Replicas causing this check to
		// pass, or, after this check has passed, a modification causes the ReplicaSet manager to
		// create more pods. This will not be an issue once we've implemented graceful delete for
		// ReplicaSets, but till then concurrent stop operations on the same ReplicaSet might have
		// unintended side effects.
		return rs.Status.ObservedGeneration >= desiredGeneration && rs.Status.Replicas == rs.Spec.Replicas, nil
	}
}

// JobHasDesiredParallelism returns a condition that will be true if the desired parallelism count
// for a job equals the current active counts or is less by an appropriate successful/unsuccessful count.
func JobHasDesiredParallelism(c BatchInterface, job *batch.Job) wait.ConditionFunc {

	return func() (bool, error) {
		job, err := c.Jobs(job.Namespace).Get(job.Name)
		if err != nil {
			return false, err
		}

		// desired parallelism can be either the exact number, in which case return immediately
		if job.Status.Active == *job.Spec.Parallelism {
			return true, nil
		}
		if job.Spec.Completions == nil {
			// A job without specified completions needs to wait for Active to reach Parallelism.
			return false, nil
		} else {
			// otherwise count successful
			progress := *job.Spec.Completions - job.Status.Active - job.Status.Succeeded
			return progress == 0, nil
		}
	}
}

// DeploymentHasDesiredReplicas returns a condition that will be true if and only if
// the desired replica count for a deployment equals its updated replicas count.
// (non-terminated pods that have the desired template spec).
func DeploymentHasDesiredReplicas(c ExtensionsInterface, deployment *extensions.Deployment) wait.ConditionFunc {
	// If we're given a deployment where the status lags the spec, it either
	// means that the deployment is stale, or that the deployment manager hasn't
	// noticed the update yet. Polling status.Replicas is not safe in the latter
	// case.
	desiredGeneration := deployment.Generation

	return func() (bool, error) {
		deployment, err := c.Deployments(deployment.Namespace).Get(deployment.Name)
		if err != nil {
			return false, err
		}
		return deployment.Status.ObservedGeneration >= desiredGeneration &&
			deployment.Status.UpdatedReplicas == deployment.Spec.Replicas, nil
	}
}

// ErrPodCompleted is returned by PodRunning or PodContainerRunning to indicate that
// the pod has already reached completed state.
var ErrPodCompleted = fmt.Errorf("pod ran to completion")

// PodRunning returns true if the pod is running, false if the pod has not yet reached running state,
// returns ErrPodCompleted if the pod has run to completion, or an error in any other case.
func PodRunning(event watch.Event) (bool, error) {
	switch event.Type {
	case watch.Deleted:
		return false, errors.NewNotFound(unversioned.GroupResource{Resource: "pods"}, "")
	}
	switch t := event.Object.(type) {
	case *api.Pod:
		switch t.Status.Phase {
		case api.PodRunning:
			return true, nil
		case api.PodFailed, api.PodSucceeded:
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
		return false, errors.NewNotFound(unversioned.GroupResource{Resource: "pods"}, "")
	}
	switch t := event.Object.(type) {
	case *api.Pod:
		switch t.Status.Phase {
		case api.PodFailed, api.PodSucceeded:
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
		return false, errors.NewNotFound(unversioned.GroupResource{Resource: "pods"}, "")
	}
	switch t := event.Object.(type) {
	case *api.Pod:
		switch t.Status.Phase {
		case api.PodFailed, api.PodSucceeded:
			return false, ErrPodCompleted
		case api.PodRunning:
			return api.IsPodReady(t), nil
		}
	}
	return false, nil
}

// PodNotPending returns true if the pod has left the pending state, false if it has not,
// or an error in any other case (such as if the pod was deleted).
func PodNotPending(event watch.Event) (bool, error) {
	switch event.Type {
	case watch.Deleted:
		return false, errors.NewNotFound(unversioned.GroupResource{Resource: "pods"}, "")
	}
	switch t := event.Object.(type) {
	case *api.Pod:
		switch t.Status.Phase {
		case api.PodPending:
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
			return false, errors.NewNotFound(unversioned.GroupResource{Resource: "pods"}, "")
		}
		switch t := event.Object.(type) {
		case *api.Pod:
			switch t.Status.Phase {
			case api.PodRunning, api.PodPending:
			case api.PodFailed, api.PodSucceeded:
				return false, ErrPodCompleted
			default:
				return false, nil
			}
			for _, s := range t.Status.ContainerStatuses {
				if s.Name != containerName {
					continue
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
		return false, errors.NewNotFound(unversioned.GroupResource{Resource: "serviceaccounts"}, "")
	}
	switch t := event.Object.(type) {
	case *api.ServiceAccount:
		return len(t.Secrets) > 0, nil
	}
	return false, nil
}
