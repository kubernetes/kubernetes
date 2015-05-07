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

package client

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
)

// ControllerHasDesiredReplicas returns a condition that will be true iff the desired replica count
// for a controller's ReplicaSelector equals the Replicas count.
func ControllerHasDesiredReplicas(c Interface, controller *api.ReplicationController) wait.ConditionFunc {
	return func() (bool, error) {
		ctrl, err := c.ReplicationControllers(controller.Namespace).Get(controller.Name)
		if err != nil {
			return false, err
		}
		return ctrl.Status.Replicas == ctrl.Spec.Replicas, nil
	}
}
