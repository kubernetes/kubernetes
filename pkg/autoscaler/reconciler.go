package autoscaler

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// Reconcile scaling actions - based on current state.
// Note: The implementation here prefers scale up actions.
func reconcileScalingAction(current, newer AutoScaleAction) AutoScaleAction {
	if newer.ScaleBy < 1 {
		// No scale by value or 0/negative implicitly means no
		// reconciliation is needed.
		return current
	}

	switch newer.ScaleType {
	case api.AutoScaleActionTypeNone:
		// No reconciliation needed - current is king!

	case api.AutoScaleActionTypeScaleDown:
		// Only prefer a scale down if not in scale up mode.
		if api.AutoScaleActionTypeScaleUp != current.ScaleType {
			// Check if currently not scaling.
			if api.AutoScaleActionTypeNone == current.ScaleType {
				return newer
			}

			//  We are scaling down, check scaleby "factor".
			if current.ScaleBy > newer.ScaleBy {
				return newer
			}
		}

	case api.AutoScaleActionTypeScaleUp:
		// Scale up is the prefered action.
		if api.AutoScaleActionTypeScaleUp != current.ScaleType {
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
func ReconcileActions(actions []AutoScaleAction) AutoScaleAction {
	result := AutoScaleAction{
		ScaleType: api.AutoScaleActionTypeNone,
		ScaleBy:   0,
	}

	for _, action := range actions {
		result = reconcileScalingAction(result, action)
	}

	return result
}
