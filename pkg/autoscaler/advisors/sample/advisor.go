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

package sample

import (
	"fmt"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/golang/glog"
)

const (
	// Name of the truthiness scaling advisor - truthiness - the W0RD!!
	TruthinessScalingAdvisorName = "truthiness"

	// Name of the falsiness scaling advisor - falsiness.
	FalsinessScalingAdvisorName = "falsiness"
)

// BooleanScalingAdvisor - returns true or false always. Implements the
// Truthiness and Falsiness scaling advisors.
type BooleanScalingAdvisor struct {
	Tag    string
	Status bool
	Client client.Interface
}

// Creates a new truthiness scaling advisor.
func NewTruthinessScalingAdvisor() autoscaler.Advisor {
	return &BooleanScalingAdvisor{
		Tag:    TruthinessScalingAdvisorName,
		Status: true,
	}
}

// Creates a new falsiness scaling advisor.
func NewFalsinessScalingAdvisor() autoscaler.Advisor {
	return &BooleanScalingAdvisor{
		Tag:    FalsinessScalingAdvisorName,
		Status: false,
	}
}

// Initializes the boolean scaling advisor.
func (bsa *BooleanScalingAdvisor) Initialize(Client client.Interface) error {
	bsa.Client = Client
	return nil
}

// Returns the name of the boolean scaling advisor.
func (bsa *BooleanScalingAdvisor) Name() string {
	return bsa.Tag
}

// Evaluates the thresholds and returns true if the desired conditions were
// (b)reached. $op governs evaluation of all/some/none of the thresholds.
func (bsa *BooleanScalingAdvisor) Evaluate(op autoscaler.AggregateOperatorType, thresholds []api.AutoScaleIntentionThresholdConfig) (bool, error) {
	checks := make([]string, 0)

	for _, t := range thresholds {
		// Example as a comparator ala V+ > V- ... really dependent
		// on the intention (MaxRPS > $value, FreeMemory < $value,
		// FailedChecks != $value (0), etc
		// All generalizations are false, including this one!!
		c := fmt.Sprintf("%v $op %v for %v seconds", t.Intent, t.Value, t.Duration)
		checks = append(checks, c)
	}

	// Instant karma's going to get you ... truthiness or falsiness!
	glog.Infof("evaluate(%v[%v]): %v", op, strings.Join(checks, ", "), bsa.Status)
	return bsa.Status, nil
}
