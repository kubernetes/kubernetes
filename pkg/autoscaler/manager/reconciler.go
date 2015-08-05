/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package manager

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler"
)

// Reconcile scaling actions - based on current state.
// Note: The implementation here prefers scale up actions.
func reconcileScalingAction(current, newer autoscaler.ScalingAction) autoscaler.ScalingAction {
	if newer.ScaleBy < 1 {
		// No scale by value or 0/negative implicitly means no
		// reconciliation is needed.
		return current
	}

	switch newer.Type {
	case api.AutoScaleActionTypeNone:
		// No reconciliation needed - current is king!

	case api.AutoScaleActionTypeScaleDown:
		// Only prefer a scale down if not in scale up mode.
		if current.Type != api.AutoScaleActionTypeScaleUp {
			// Check if currently not scaling.
			if api.AutoScaleActionTypeNone == current.Type {
				return newer
			}

			//  We are scaling down, check scaleby "factor".
			if current.ScaleBy > newer.ScaleBy {
				return newer
			}
		}

	case api.AutoScaleActionTypeScaleUp:
		// Scale up is the prefered action.
		if current.Type != api.AutoScaleActionTypeScaleUp {
			return newer
		}

		// Otherwise, we are scaling up - check if scale by is
		// greater than the current value.
		if current.ScaleBy < newer.ScaleBy {
			return newer
		}
	}

	return current
}

// Reconcile scaling actions. Coalesces multiple actions into a single one.
func ReconcileActions(actions []autoscaler.ScalingAction) autoscaler.ScalingAction {
	result := autoscaler.ScalingAction{
		Type:    api.AutoScaleActionTypeNone,
		ScaleBy: 0,
	}

	for _, action := range actions {
		result = reconcileScalingAction(result, action)
	}

	return result
}
