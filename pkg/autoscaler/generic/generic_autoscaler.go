package autoscaler

import (
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
)

// GenericAutoScaler evaluates, reconciles and performs the scaling actions
// for threshold based intentions against one or more registered
// monitoring store plugins.
type GenericAutoScaler struct {
	name string
}

// Create a new generic auto scaler instance.
func NewGenericAutoScaler(name string) *AutoScalerPlugin {
	return &GenericAutoScaler{name}
}

// Return the name of this generic auto scaler instance.
func (gas *GenericAutoScaler) Name() string {
	return gas.name
}

// Assess the policies against all of the monitoring sources and return a
// set of of desired actions (and any errors).
// Note: A custom auto scaling environment could pick one or more or all of
//       the monitoring sources or alternatively use custom monitoring
//       source(s) ignoring the list of available sources.
func (gas *GenericAutoScaler) Assess(spec api.AutoScalerSpec, sources []autoscaler.MonitoringSource) ([]autoscaler.AutoScaleAction, error) {
	allActions := make([]autoscaler.AutoScaleAction, 0)
	allErrors := make([]error, 0)

	for _, t := range spec.Thresholds {
		desiredActions, err := gas.checkThreshold(t, sources)
		if err != nil {
			allErrors = append(allErrors, err)
			continue
		}

		for _, action := range desiredActions {
			allActions = append(allActions, action)
		}
	}

	return allActions, errors.NewAggregate(allErrors)
}

func (gas *GenericAutoScaler) checkThreshold(theshold AutoScaleThreshold, sources []autoscaler.MonitoringSource) ([]autoscaler.AutoScaleAction, error) {
	allActions := make([]autoscaler.AutoScaleAction, 0)
	allErrors := make([]error, 0)

	for _, monitor := range sources {
		scale, err := monitor.Evaluate(autoscaler.AggregateOperatorTypeAny, threshold.Intentions)
		if err != nil {
			allErrors = append(allErrors, err)
			continue
		}

		if scale {
			action := autoscaler.AutoScaleAction{
				actionType: t.ActionType,
				scaleBy:    t.Scaleby,
			}

			allActions = append(allActions, action)
		}
	}

	return allActions, errors.NewAggregate(allErrors)
}
