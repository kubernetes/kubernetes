/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/util/wait"
)

// DefaultRetry is the recommended retry for a conflict where multiple clients
// are making changes to the same resource.
var DefaultRetry = wait.Backoff{
	Steps:    5,
	Duration: 10 * time.Millisecond,
	Factor:   1.0,
	Jitter:   0.1,
}

// DefaultBackoff is the recommended backoff for a conflict where a client
// may be attempting to make an unrelated modification to a resource under
// active management by one or more controllers.
var DefaultBackoff = wait.Backoff{
	Steps:    4,
	Duration: 10 * time.Millisecond,
	Factor:   5.0,
	Jitter:   0.1,
}

// RetryConflict executes the provided function repeatedly, retrying if the server returns a conflicting
// write. Callers should preserve previous executions if they wish to retry changes. It performs an
// exponential backoff.
//
//     var pod *api.Pod
//     err := RetryOnConflict(DefaultBackoff, func() (err error) {
//       pod, err = c.Pods("mynamespace").UpdateStatus(podStatus)
//       return
//     })
//     if err != nil {
//       // may be conflict if max retries were hit
//       return err
//     }
//     ...
//
// TODO: Make Backoff an interface?
func RetryOnConflict(backoff wait.Backoff, fn func() error) error {
	var lastConflictErr error
	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		err := fn()
		switch {
		case err == nil:
			return true, nil
		case errors.IsConflict(err):
			lastConflictErr = err
			return false, nil
		default:
			return false, err
		}
	})
	if err == wait.ErrWaitTimeout {
		err = lastConflictErr
	}
	return err
}

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
func JobHasDesiredParallelism(c ExtensionsInterface, job *extensions.Job) wait.ConditionFunc {

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
