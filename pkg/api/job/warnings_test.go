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
	"k8s.io/utils/pointer"
)

var (
	validSelector = &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}

	validPodTemplate = core.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: core.PodSpec{
			RestartPolicy: core.RestartPolicyOnFailure,
			DNSPolicy:     core.DNSClusterFirst,
			Containers:    []core.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: core.TerminationMessageReadFile}},
		},
	}
)

func TestWarningsForJobSpec(t *testing.T) {
	cases := map[string]struct {
		spec              *batch.JobSpec
		wantWarningsCount int
	}{
		"valid NonIndexed": {
			spec: &batch.JobSpec{
				CompletionMode: completionModePtr(batch.NonIndexedCompletion),
				Template:       validPodTemplate,
			},
		},
		"NondIndexed with high completions and parallelism": {
			spec: &batch.JobSpec{
				CompletionMode: completionModePtr(batch.NonIndexedCompletion),
				Template:       validPodTemplate,
				Parallelism:    pointer.Int32(1_000_000_000),
				Completions:    pointer.Int32(1_000_000_000),
			},
		},
		"invalid PodTemplate": {
			spec: &batch.JobSpec{
				CompletionMode: completionModePtr(batch.NonIndexedCompletion),
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{ImagePullSecrets: []core.LocalObjectReference{{Name: ""}}},
				},
			},
			wantWarningsCount: 1,
		},
		"valid Indexed low completions low parallelism": {
			spec: &batch.JobSpec{
				CompletionMode: completionModePtr(batch.IndexedCompletion),
				Completions:    pointer.Int32(10_000),
				Parallelism:    pointer.Int32(10_000),
				Template:       validPodTemplate,
			},
		},
		"valid Indexed high completions low parallelism": {
			spec: &batch.JobSpec{
				CompletionMode: completionModePtr(batch.IndexedCompletion),
				Completions:    pointer.Int32(1000_000_000),
				Parallelism:    pointer.Int32(10_000),
				Template:       validPodTemplate,
			},
		},
		"valid Indexed medium completions medium parallelism": {
			spec: &batch.JobSpec{
				CompletionMode: completionModePtr(batch.IndexedCompletion),
				Completions:    pointer.Int32(100_000),
				Parallelism:    pointer.Int32(100_000),
				Template:       validPodTemplate,
			},
		},
		"invalid Indexed high completions high parallelism": {
			spec: &batch.JobSpec{
				CompletionMode: completionModePtr(batch.IndexedCompletion),
				Completions:    pointer.Int32(100_001),
				Parallelism:    pointer.Int32(10_001),
				Template:       validPodTemplate,
			},
			wantWarningsCount: 1,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			warnings := WarningsForJobSpec(ctx, nil, tc.spec, nil)
			if len(warnings) != tc.wantWarningsCount {
				t.Errorf("Got %d warnings, want %d.\nWarnings: %v", len(warnings), tc.wantWarningsCount, warnings)
			}
		})
	}
}

func completionModePtr(m batch.CompletionMode) *batch.CompletionMode {
	return &m
}
