/*
Copyright The Kubernetes Authors.

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

package job

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropDisabledFieldsScheduling(t *testing.T) {
	schedulingConfig := func() *batch.JobSchedulingConfiguration {
		return &batch.JobSchedulingConfiguration{Policy: &scheduling.PodGroupSchedulingPolicy{Basic: &scheduling.BasicSchedulingPolicy{}}}
	}

	cases := map[string]struct {
		enableWorkloadWithJob bool
		jobSpec               *batch.JobSchedulingConfiguration
		oldJobSpec            *batch.JobSchedulingConfiguration
		wantDropped           bool
		// hasOldSpec distinguishes create (nil old spec) from update.
		hasOldSpec bool
	}{
		"feature enabled, create keeps scheduling": {
			enableWorkloadWithJob: true,
			jobSpec:               schedulingConfig(),
		},
		"feature disabled, create drops scheduling": {
			enableWorkloadWithJob: false,
			jobSpec:               schedulingConfig(),
			wantDropped:           true,
		},
		"feature disabled, create without scheduling stays nil": {
			enableWorkloadWithJob: false,
			jobSpec:               nil,
		},
		"feature enabled, update keeps scheduling": {
			enableWorkloadWithJob: true,
			jobSpec:               schedulingConfig(),
			oldJobSpec:            schedulingConfig(),
			hasOldSpec:            true,
		},
		"feature disabled, update preserves scheduling already in use": {
			enableWorkloadWithJob: false,
			jobSpec:               schedulingConfig(),
			oldJobSpec:            schedulingConfig(),
			hasOldSpec:            true,
		},
		"feature disabled, update drops newly added scheduling": {
			enableWorkloadWithJob: false,
			jobSpec:               schedulingConfig(),
			oldJobSpec:            nil,
			hasOldSpec:            true,
			wantDropped:           true,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			// WorkloadWithJob depends on GenericWorkload, so toggle them together.
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: tc.enableWorkloadWithJob,
				features.WorkloadWithJob: tc.enableWorkloadWithJob,
			})

			newSpec := &batch.JobSpec{Scheduling: tc.jobSpec}
			var oldSpec *batch.JobSpec
			if tc.hasOldSpec {
				oldSpec = &batch.JobSpec{Scheduling: tc.oldJobSpec}
			}

			DropDisabledFields(newSpec, oldSpec)

			var want *batch.JobSchedulingConfiguration
			if !tc.wantDropped {
				want = tc.jobSpec
			}
			if diff := cmp.Diff(want, newSpec.Scheduling); diff != "" {
				t.Errorf("unexpected scheduling after DropDisabledFields (-want,+got):\n%s", diff)
			}
		})
	}
}
