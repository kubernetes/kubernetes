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

package validation

import (
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func TestValidateJob(t *testing.T) {
	validSelector := &unversioned.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
	}
	successCases := []extensions.Job{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.JobSpec{
				Selector: validSelector,
				Template: validPodTemplateSpec,
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateJob(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	negative := -1
	negative64 := int64(-1)
	errorCases := map[string]extensions.Job{
		"spec.parallelism:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.JobSpec{
				Parallelism: &negative,
				Selector:    validSelector,
				Template:    validPodTemplateSpec,
			},
		},
		"spec.completions:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.JobSpec{
				Completions: &negative,
				Selector:    validSelector,
				Template:    validPodTemplateSpec,
			},
		},
		"spec.activeDeadlineSeconds:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.JobSpec{
				ActiveDeadlineSeconds: &negative64,
				Selector:              validSelector,
				Template:              validPodTemplateSpec,
			},
		},
		"spec.selector:Required value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.JobSpec{
				Template: validPodTemplateSpec,
			},
		},
		"spec.template.metadata.labels: Invalid value: {\"y\":\"z\"}: `selector` does not match template `labels`": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.JobSpec{
				Selector: validSelector,
				Template: api.PodTemplateSpec{
					ObjectMeta: api.ObjectMeta{
						Labels: map[string]string{"y": "z"},
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyOnFailure,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
				},
			},
		},
		"spec.template.spec.restartPolicy: Unsupported value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: extensions.JobSpec{
				Selector: validSelector,
				Template: api.PodTemplateSpec{
					ObjectMeta: api.ObjectMeta{
						Labels: validSelector.MatchLabels,
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
				},
			},
		},
	}

	for k, v := range errorCases {
		errs := ValidateJob(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %v, expected: %s", err, k)
			}
		}
	}
}
