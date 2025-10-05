/*
Copyright 2024 The Kubernetes Authors.

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

package queueing

import (
	"fmt"
	"strings"
	"testing"
	"time"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/scheduler/queueing"
)

// TestCoreResourceEnqueue verify Pods failed by in-tree default plugins can be
// moved properly upon their registered events.
// Here, we run only the test cases where the EnableSchedulingQueueHint is enabled.
func TestCoreResourceEnqueue(t *testing.T) {
	for _, tt := range queueing.CoreResourceEnqueueTestCases {
		if tt.EnableSchedulingQueueHint != nil && !tt.EnableSchedulingQueueHint.Has(true) {
			continue
		}

		testName := strings.Join(append(tt.EnablePlugins, tt.Name), "/")

		t.Run(testName, func(t *testing.T) {
			start := time.Now()
			fmt.Printf("ENLOG: === START: %s at %s ===\n", testName, start.Format(time.RFC3339))

			// Enable the feature gate before running the test
			featuregatetesting.SetFeatureGateDuringTest(
				t,
				utilfeature.DefaultFeatureGate,
				features.SchedulerQueueingHints,
				true,
			)

			// Run the actual test logic
			queueing.RunTestCoreResourceEnqueue(t, tt)

			end := time.Now()
			duration := end.Sub(start)
			fmt.Printf("ENLOG: === END:   %s at %s (duration: %v) ===\n\n",
				testName,
				end.Format(time.RFC3339),
				duration,
			)
		})
	}
}
