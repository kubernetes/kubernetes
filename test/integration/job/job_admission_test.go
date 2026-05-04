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
	"strings"
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestJobAdmissionParallelismUpdate(t *testing.T) {
	enabledFeatureGates := featuregatetesting.FeatureOverrides{
		features.GenericWorkload: true,
		features.WorkloadWithJob: true,
	}

	cases := map[string]struct {
		featureGates    featuregatetesting.FeatureOverrides
		jobSpec         batchv1.JobSpec
		newParallelism  int32
		wantForbidden   bool
		wantErrContains string
		waitForPodGroup bool
		startController bool
	}{
		"parallelism update blocked for gang-scheduled job": {
			featureGates: enabledFeatureGates,
			jobSpec: batchv1.JobSpec{
				Parallelism:    ptr.To[int32](4),
				Completions:    ptr.To[int32](4),
				CompletionMode: ptr.To(batchv1.IndexedCompletion),
			},
			newParallelism:  2,
			wantForbidden:   true,
			wantErrContains: "cannot change parallelism",
			waitForPodGroup: true,
			startController: true,
		},
		"parallelism update allowed for non-indexed job": {
			featureGates: enabledFeatureGates,
			jobSpec: batchv1.JobSpec{
				Parallelism: ptr.To[int32](3),
			},
			newParallelism:  2,
			wantForbidden:   false,
			startController: true,
		},
		"parallelism update allowed for single-pod indexed job": {
			featureGates: enabledFeatureGates,
			jobSpec: batchv1.JobSpec{
				Parallelism:    ptr.To[int32](1),
				Completions:    ptr.To[int32](1),
				CompletionMode: ptr.To(batchv1.IndexedCompletion),
			},
			newParallelism:  2,
			wantForbidden:   false,
			startController: true,
		},
		"parallelism update allowed when feature gates are disabled": {
			featureGates: featuregatetesting.FeatureOverrides{
				features.GenericWorkload: false,
				features.WorkloadWithJob: false,
			},
			jobSpec: batchv1.JobSpec{
				Parallelism: ptr.To[int32](4),
			},
			newParallelism: 2,
			wantForbidden:  false,
		},
		"parallelism update allowed when only GenericWorkload is enabled": {
			featureGates: featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
				features.WorkloadWithJob: false,
			},
			jobSpec: batchv1.JobSpec{
				Parallelism: ptr.To[int32](4),
			},
			newParallelism: 2,
			wantForbidden:  false,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, feature.DefaultFeatureGate, tc.featureGates)

			closeFn, restConfig, clientSet, ns := setup(t, "job-admission")
			t.Cleanup(closeFn)

			ctx := t.Context()
			if tc.startController {
				var cancel func()
				ctx, cancel = startJobControllerAndWaitForCaches(t, restConfig)
				t.Cleanup(cancel)
			}

			jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{Spec: tc.jobSpec})
			if err != nil {
				t.Fatalf("Failed to create Job: %v", err)
			}

			if tc.waitForPodGroup {
				waitForPodGroup(ctx, t, clientSet, jobObj, false, wait.ForeverTestTimeout)
			}

			_, err = updateJob(ctx, clientSet.BatchV1().Jobs(ns.Name), jobObj.Name, func(j *batchv1.Job) {
				j.Spec.Parallelism = ptr.To(tc.newParallelism)
			})

			if tc.wantForbidden {
				if err == nil {
					t.Fatal("Expected Forbidden error when updating parallelism, got nil")
				}
				if !apierrors.IsForbidden(err) {
					t.Errorf("Expected Forbidden error, got: %v", err)
				}
				if tc.wantErrContains != "" && !strings.Contains(err.Error(), tc.wantErrContains) {
					t.Errorf("Expected error to contain %q, got: %v", tc.wantErrContains, err)
				}
			} else if err != nil {
				t.Errorf("Expected parallelism update to succeed, got: %v", err)
			}
		})
	}
}
