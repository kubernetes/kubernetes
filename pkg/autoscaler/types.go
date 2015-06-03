package autoscaler

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// AutoScaleAction defines the auto scale action and how much to scale by.
// This what the autoscaler plugins use to indicate to the autoscaler
// manager of what actions are desired.
type AutoScaleAction struct {
	ScaleType api.AutoScaleActionType
	ScaleBy   int

	Trigger *api.AutoScaleThreshold
}
