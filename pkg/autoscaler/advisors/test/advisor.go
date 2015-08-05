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

package test

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

// Test scaling advisor.
type TestAdvisor struct {
	Tag    string
	Status bool
	Error  string
}

// Initializes the test scaling advisor.
func (tms *TestAdvisor) Initialize(Client client.Interface) error {
	return nil
}

// Returns the name (tag) of this test scaling advisor instance.
func (tms *TestAdvisor) Name() string {
	return tms.Tag
}

// Evaluates the thresholds and returns true if the desired conditions were
// (b)reached. $op governs evaluation of all/some/none of the thresholds.
func (tms *TestAdvisor) Evaluate(op autoscaler.AggregateOperatorType, thresholds []api.AutoScaleIntentionThresholdConfig) (bool, error) {
	if len(tms.Error) > 0 {
		return tms.Status, fmt.Errorf(tms.Error)
	}

	return tms.Status, nil
}
