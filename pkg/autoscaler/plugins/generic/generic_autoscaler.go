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

package generic

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
)

const (
	// Default plugin name.
	DefaultName = "generic"
)

// GenericAutoScaler evaluates, reconciles and performs the scaling actions
// for threshold based intentions using one or more registered advisors.
type GenericAutoScaler struct {
	name string
}

// Create a new generic auto scaler instance.
func NewGenericAutoScaler(name string) autoscaler.AutoScalerPlugin {
	if 0 == len(name) {
		name = DefaultName
	}

	return &GenericAutoScaler{name}
}

// Return the name of this generic auto scaler instance.
func (gas *GenericAutoScaler) Name() string {
	return gas.name
}

// Assess the policies against all advisors specified in the spec and
// return a set of desired actions (and any errors).
// Note: A custom auto scaling environment could pick one or more or all of
//       the advisors or alternatively use custom advisor(s) ignoring the
//       list of available advisors.
func (gas *GenericAutoScaler) Assess(spec api.AutoScalerSpec, availableAdvisors []autoscaler.Advisor) ([]autoscaler.ScalingAction, error) {
	allActions := make([]autoscaler.ScalingAction, 0)
	allErrors := make([]error, 0)

	advisors := getFilteredAdvisors(availableAdvisors, spec.Advisors)

	// Ensure we have advisors to give us a judgement.
	if len(advisors) < 1 {
		return allActions, errors.NewAggregate(allErrors)
	}

	for _, t := range spec.Thresholds {
		desiredActions, err := gas.checkThreshold(t, advisors)
		if err != nil {
			allErrors = append(allErrors, err)
		}

		for _, action := range desiredActions {
			allActions = append(allActions, action)
		}
	}

	return allActions, errors.NewAggregate(allErrors)
}

// checkThreshold compares the threshold using all the advisors and
// returns a list of all the desired actions.
func (gas *GenericAutoScaler) checkThreshold(threshold api.AutoScaleThreshold, advisors []autoscaler.Advisor) ([]autoscaler.ScalingAction, error) {
	allActions := make([]autoscaler.ScalingAction, 0)
	allErrors := make([]error, 0)

	for _, bailiff := range advisors {
		scale, err := bailiff.Evaluate(autoscaler.AggregateOperatorTypeAny, threshold.Intentions)
		if err != nil {
			allErrors = append(allErrors, err)
		}

		if scale {
			action := autoscaler.ScalingAction{
				Type:    threshold.ActionType,
				ScaleBy: threshold.ScaleBy,
				Trigger: threshold,
			}

			allActions = append(allActions, action)
		}
	}

	return allActions, errors.NewAggregate(allErrors)
}

// Returns a filtered list of advisors based on the keys.
func getFilteredAdvisors(available []autoscaler.Advisor, keys []string) []autoscaler.Advisor {
	advisors := make([]autoscaler.Advisor, 0)

	// Make sure there's available advisors.
	if 0 == len(available) || 0 == len(keys) {
		// No advisors.
		return advisors
	}

	namesMap := make(map[string]string)
	for _, k := range keys {
		namesMap[k] = k
	}

	for _, ad := range available {
		if _, ok := namesMap[ad.Name()]; ok {
			advisors = append(advisors, ad)
		}
	}

	return advisors
}
