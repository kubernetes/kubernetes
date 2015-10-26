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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/util/wait"
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

// JobHasDesiredParallelism returns a condition that will be true if the desired parallelism count
// for a job equals the current active counts or is less by an appropriate successful/unsuccessful count.
func JobHasDesiredParallelism(c Interface, job *extensions.Job) wait.ConditionFunc {

	return func() (bool, error) {
		job, err := c.Extensions().Jobs(job.Namespace).Get(job.Name)
		if err != nil {
			return false, err
		}

		// desired parallelism can be either the exact number, in which case return immediately
		if job.Status.Active == *job.Spec.Parallelism {
			return true, nil
		}
		// otherwise count successful
		progress := *job.Spec.Completions - job.Status.Active - job.Status.Succeeded
		return progress == 0, nil
	}
}
