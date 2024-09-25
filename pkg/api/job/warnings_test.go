/*
Copyright 2023 The Kubernetes Authors.

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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

var (
	// validSelector is a valid label selector used to match a set of labels.
	// This is used to create a valid PodTemplateSpec for testing.
	validSelector = &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}

	// validPodTemplate defines a valid pod template specification used in test cases.
	// It contains required fields for creating a pod, such as labels, restart policy, DNS policy, and container details.
	validPodTemplate = core.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: core.PodSpec{
			RestartPolicy: core.RestartPolicyOnFailure,
			DNSPolicy:     core.DNSClusterFirst,
			Containers: []core.Container{
				{
					Name:                    "abc",
					Image:                   "image",
					ImagePullPolicy:         "IfNotPresent",
					TerminationMessagePolicy: core.TerminationMessageReadFile,
				},
			},
		},
	}
)

// TestWarningsForJobSpec is a table-driven test that validates the warning system for different JobSpec configurations.
// It checks the number of warnings generated based on the characteristics of the job, such as completion mode, parallelism, and pod template validity.
func TestWarningsForJobSpec(t *testing.T) {
	cases := map[string]struct {
		spec              *batch.JobSpec // JobSpec to be tested
		wantWarningsCount int            // Expected number of warnings
	}{
		// "valid NonIndexed" tests a valid non-indexed completion mode job.
		"valid NonIndexed": {
			spec: &batch.JobSpec{
				CompletionMode: ptr.To(batch.NonIndexedCompletion),
				Template:       validPodTemplate,
			},
		},
		// "NonIndexed with high completions and parallelism" tests a non-indexed job with extremely high completions and parallelism values.
		"NondIndexed with high completions and parallelism": {
			spec: &batch.JobSpec{
				CompletionMode: ptr.To(batch.NonIndexedCompletion),
				Template:       validPodTemplate,
				Parallelism:    pointer.Int32(1_000_000_000),  // Unusually high parallelism
				Completions:    pointer.Int32(1_000_000_000),  // Unusually high completions
			},
		},
		// "invalid PodTemplate" tests a job with an invalid pod template, such as an empty image pull secret.
		"invalid PodTemplate": {
			spec: &batch.JobSpec{
				CompletionMode: ptr.To(batch.NonIndexedCompletion),
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{ImagePullSecrets: []core.LocalObjectReference{{Name: ""}}},
				},
			},
			wantWarningsCount: 1,  // Expecting 1 warning due to invalid pod template
		},
		// "valid Indexed low completions low parallelism" tests an indexed job with low completions and parallelism.
		"valid Indexed low completions low parallelism": {
			spec: &batch.JobSpec{
				CompletionMode: ptr.To(batch.IndexedCompletion),
				Completions:    ptr.To[int32](10_000),
				Parallelism:    ptr.To[int32](10_000),
				Template:       validPodTemplate,
			},
		},
		// "valid Indexed high completions low parallelism" tests an indexed job with high completions and relatively low parallelism.
		"valid Indexed high completions low parallelism": {
			spec: &batch.JobSpec{
				CompletionMode: completionModePtr(batch.IndexedCompletion),
				Completions:    pointer.Int32(1_000_000_000),
				Parallelism:    pointer.Int32(10_000),
				Template:       validPodTemplate,
			},
		},
		// "valid Indexed medium completions medium parallelism" tests an indexed job with medium completions and medium parallelism.
		"valid Indexed medium completions medium parallelism": {
			spec: &batch.JobSpec{
				CompletionMode: ptr.To(batch.IndexedCompletion),
				Completions:    ptr.To[int32](100_000),
				Parallelism:    ptr.To[int32](100_000),
				Template:       validPodTemplate,
			},
		},
		// "invalid Indexed high completions high parallelism" tests an indexed job with completions and parallelism that exceed allowed thresholds.
		"invalid Indexed high completions high parallelism": {
			spec: &batch.JobSpec{
				CompletionMode: completionModePtr(batch.IndexedCompletion),
				Completions:    pointer.Int32(100_001), // Exceeds allowed completions
				Parallelism:    pointer.Int32(10_001),  // Exceeds allowed parallelism
				Template:       validPodTemplate,
			},
			wantWarningsCount: 1,  // Expecting 1 warning due to exceeding thresholds
		},
	}

	// Iterate over test cases and run each one.
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)  // Create a new test context
			warnings := WarningsForJobSpec(ctx, nil, tc.spec, nil)  // Call WarningsForJobSpec to get warnings for the job spec
			if len(warnings) != tc.wantWarningsCount {
				// If the number of warnings doesn't match the expected count, log an error
				t.Errorf("Got %d warnings, want %d.\nWarnings: %v", len(warnings), tc.wantWarningsCount, warnings)
			}
		})
	}
}

// completionModePtr is a helper function that returns a pointer to the given completion mode.
// This is used to easily create pointer-based values for completion modes.
func completionModePtr(m batch.CompletionMode) *batch.CompletionMode {
	return &m
}
