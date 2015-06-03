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

package autoscaler

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// Aggregate operator type used for evaluating autoscale thresholds.
type AggregateOperatorType string

const (
	// Aggregate operation on all/any of the thresholds.
	AggregateOperatorTypeAll AggregateOperatorType = "All"
	AggregateOperatorTypeAny AggregateOperatorType = "Any"

	// Aggregate operation on some (aka any) of the thresholds.
	AggregateOperatorTypeSome AggregateOperatorType = "Some"

	// Aggregate operation on none of the thresholds.
	AggregateOperatorTypeNone AggregateOperatorType = "None"
)

// AutoScaler plugin assess a given spec against one/more/all of the
// monitoring sources and indicates what autoscaler actions are desired.
type AutoScalerPlugin interface {
	Name() string
	Assess(spec api.AutoScalerSpec, sources []*MonitoringSource) ([]AutoScaleAction, error)
}

// MonitoringSource evaluates the intents based on the current state.
type MonitoringSource interface {
	Initialize(config map[string]string)
	Name() string
	Evaluate(op AggregateOperatorType, thresholds []api.AutoScaleIntentionThresholdConfig) (bool, error)
}
