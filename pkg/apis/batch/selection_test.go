/*
Copyright 2015 The Kubernetes Authors.

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

package batch

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

func TestJobStrategy_GetAttrs(t *testing.T) {
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplateSpec := core.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: core.PodSpec{
			RestartPolicy: core.RestartPolicyOnFailure,
			DNSPolicy:     core.DNSClusterFirst,
			Containers:    []core.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: core.TerminationMessageReadFile}},
		},
	}

	cases := map[string]struct {
		job          *Job
		wantErr      string
		nonJobObject *core.Pod
	}{
		"valid job with no labels": {
			job: &Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
				},
				Spec: JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},
		},
		"valid job with a label": {
			job: &Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Labels:          map[string]string{"a": "b"},
				},
				Spec: JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},
		},
		"pod instead": {
			job:          nil,
			nonJobObject: &core.Pod{},
			wantErr:      "given object is not a job",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if tc.job == nil {
				_, _, err := JobGetAttrs(tc.nonJobObject)
				if diff := cmp.Diff(tc.wantErr, err.Error()); diff != "" {
					t.Errorf("Unexpected errors (-want,+got):\n%s", diff)
				}
			} else {
				gotLabels, _, err := JobGetAttrs(tc.job)
				if err != nil {
					t.Errorf("Error %s supposed to be nil", err.Error())
				}
				if diff := cmp.Diff(labels.Set(tc.job.ObjectMeta.Labels), gotLabels); diff != "" {
					t.Errorf("Unexpected attrs (-want,+got):\n%s", diff)
				}
			}
		})
	}
}
