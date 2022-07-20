/*
Copyright 2022 The Kubernetes Authors.

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

package request

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/apis/apiserver"
)

const (
	defaultEnableMutatingWorkEstimator = true
	defaultEventAdditionalDuration     = 5 * time.Millisecond
	defaultMaximumSeats                = 10
	defaultObjectsPerSeat              = 100.0
	defaultWatchesPerSeat              = 10.0

	// minimumSeats is the minimum number of seats a request must occupy.
	minimumSeats = 1
)

// DefaultWorkEstimatorConfiguration creates a new WorkEstimatorConfig with default values.
func DefaultWorkEstimatorConfiguration() *apiserver.WorkEstimatorConfiguration {
	return &apiserver.WorkEstimatorConfiguration{
		MaximumSeats:          defaultMaximumSeats,
		ListWorkEstimator:     defaultListWorkEstimatorConfig(),
		MutatingWorkEstimator: defaultMutatingWorkEstimatorConfig(),
	}
}

// defaultListWorkEstimatorConfig creates a new ListWorkEstimatorConfig with default values.
func defaultListWorkEstimatorConfig() *apiserver.ListWorkEstimatorConfiguration {
	return &apiserver.ListWorkEstimatorConfiguration{ObjectsPerSeat: defaultObjectsPerSeat}
}

// defaultMutatingWorkEstimatorConfig creates a new MutatingWorkEstimatorConfig with default values.
func defaultMutatingWorkEstimatorConfig() *apiserver.MutatingWorkEstimatorConfiguration {
	return &apiserver.MutatingWorkEstimatorConfiguration{
		EventAdditionalDuration: metav1.Duration{Duration: defaultEventAdditionalDuration},
		WatchesPerSeat:          defaultWatchesPerSeat,
	}
}
