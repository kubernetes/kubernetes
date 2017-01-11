/*
Copyright 2016 The Kubernetes Authors.

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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/batch"
)

func getValidManualSelector() *metav1.LabelSelector {
	return &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
}

func getValidPodTemplateSpecForManual(selector *metav1.LabelSelector) api.PodTemplateSpec {
	return api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Labels: selector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
	}
}

func getValidGeneratedSelector() *metav1.LabelSelector {
	return &metav1.LabelSelector{
		MatchLabels: map[string]string{"controller-uid": "1a2b3c", "job-name": "myjob"},
	}
}

func getValidPodTemplateSpecForGenerated(selector *metav1.LabelSelector) api.PodTemplateSpec {
	return api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Labels: selector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
	}
}

func TestValidateJob(t *testing.T) {
	validManualSelector := getValidManualSelector()
	validPodTemplateSpecForManual := getValidPodTemplateSpecForManual(validManualSelector)
	validGeneratedSelector := getValidGeneratedSelector()
	validPodTemplateSpecForGenerated := getValidPodTemplateSpecForGenerated(validGeneratedSelector)

	successCases := map[string]batch.Job{
		"manual selector": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Selector:       validManualSelector,
				ManualSelector: newBool(true),
				Template:       validPodTemplateSpecForManual,
			},
		},
		"generated selector": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGenerated,
			},
		},
	}
	for k, v := range successCases {
		if errs := ValidateJob(&v); len(errs) != 0 {
			t.Errorf("expected success for %s: %v", k, errs)
		}
	}
	negative := int32(-1)
	negative64 := int64(-1)
	errorCases := map[string]batch.Job{
		"spec.parallelism:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Parallelism: &negative,
				Selector:    validGeneratedSelector,
				Template:    validPodTemplateSpecForGenerated,
			},
		},
		"spec.completions:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Completions: &negative,
				Selector:    validGeneratedSelector,
				Template:    validPodTemplateSpecForGenerated,
			},
		},
		"spec.activeDeadlineSeconds:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				ActiveDeadlineSeconds: &negative64,
				Selector:              validGeneratedSelector,
				Template:              validPodTemplateSpecForGenerated,
			},
		},
		"spec.selector:Required value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Template: validPodTemplateSpecForGenerated,
			},
		},
		"spec.template.metadata.labels: Invalid value: {\"y\":\"z\"}: `selector` does not match template `labels`": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Selector:       validManualSelector,
				ManualSelector: newBool(true),
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
		"spec.template.metadata.labels: Invalid value: {\"controller-uid\":\"4d5e6f\"}: `selector` does not match template `labels`": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Selector:       validManualSelector,
				ManualSelector: newBool(true),
				Template: api.PodTemplateSpec{
					ObjectMeta: api.ObjectMeta{
						Labels: map[string]string{"controller-uid": "4d5e6f"},
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
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Selector:       validManualSelector,
				ManualSelector: newBool(true),
				Template: api.PodTemplateSpec{
					ObjectMeta: api.ObjectMeta{
						Labels: validManualSelector.MatchLabels,
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

func TestValidateJobUpdateStatus(t *testing.T) {
	type testcase struct {
		old    batch.Job
		update batch.Job
	}

	successCases := []testcase{
		{
			old: batch.Job{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Status: batch.JobStatus{
					Active:    1,
					Succeeded: 2,
					Failed:    3,
				},
			},
			update: batch.Job{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Status: batch.JobStatus{
					Active:    1,
					Succeeded: 1,
					Failed:    3,
				},
			},
		},
	}

	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateJobUpdateStatus(&successCase.update, &successCase.old); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]testcase{
		"[status.active: Invalid value: -1: must be greater than or equal to 0, status.succeeded: Invalid value: -2: must be greater than or equal to 0]": {
			old: batch.Job{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: batch.JobStatus{
					Active:    1,
					Succeeded: 2,
					Failed:    3,
				},
			},
			update: batch.Job{
				ObjectMeta: api.ObjectMeta{
					Name:            "abc",
					Namespace:       api.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: batch.JobStatus{
					Active:    -1,
					Succeeded: -2,
					Failed:    3,
				},
			},
		},
	}

	for testName, errorCase := range errorCases {
		errs := ValidateJobUpdateStatus(&errorCase.update, &errorCase.old)
		if len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
			continue
		}
		if errs.ToAggregate().Error() != testName {
			t.Errorf("expected '%s' got '%s'", errs.ToAggregate().Error(), testName)
		}
	}
}

func TestValidateCronJob(t *testing.T) {
	validManualSelector := getValidManualSelector()
	validPodTemplateSpec := getValidPodTemplateSpecForGenerated(getValidGeneratedSelector())
	validPodTemplateSpec.Labels = map[string]string{}

	successCases := map[string]batch.CronJob{
		"basic scheduled job": {
			ObjectMeta: api.ObjectMeta{
				Name:      "mycronjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"non-standard scheduled": {
			ObjectMeta: api.ObjectMeta{
				Name:      "mycronjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "@hourly",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
	}
	for k, v := range successCases {
		if errs := ValidateCronJob(&v); len(errs) != 0 {
			t.Errorf("expected success for %s: %v", k, errs)
		}
	}

	negative := int32(-1)
	negative64 := int64(-1)

	errorCases := map[string]batch.CronJob{
		"spec.schedule: Invalid value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "mycronjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "error",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.schedule: Required value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "mycronjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.startingDeadlineSeconds:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "mycronjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:                "* * * * ?",
				ConcurrencyPolicy:       batch.AllowConcurrent,
				StartingDeadlineSeconds: &negative64,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.concurrencyPolicy: Required value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "mycronjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule: "* * * * ?",
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.jobTemplate.spec.parallelism:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "mycronjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Parallelism: &negative,
						Template:    validPodTemplateSpec,
					},
				},
			},
		},
		"spec.jobTemplate.spec.completions:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "mycronjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{

					Spec: batch.JobSpec{
						Completions: &negative,
						Template:    validPodTemplateSpec,
					},
				},
			},
		},
		"spec.jobTemplate.spec.activeDeadlineSeconds:must be greater than or equal to 0": {
			ObjectMeta: api.ObjectMeta{
				Name:      "mycronjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						ActiveDeadlineSeconds: &negative64,
						Template:              validPodTemplateSpec,
					},
				},
			},
		},
		"spec.jobTemplate.spec.selector: Invalid value: {\"matchLabels\":{\"a\":\"b\"}}: `selector` will be auto-generated": {
			ObjectMeta: api.ObjectMeta{
				Name:      "mycronjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Selector: validManualSelector,
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.jobTemplate.spec.manualSelector: Unsupported value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "mycronjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						ManualSelector: newBool(true),
						Template:       validPodTemplateSpec,
					},
				},
			},
		},
		"spec.jobTemplate.spec.template.spec.restartPolicy: Unsupported value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "mycronjob",
				Namespace: api.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: api.PodTemplateSpec{
							Spec: api.PodSpec{
								RestartPolicy: api.RestartPolicyAlways,
								DNSPolicy:     api.DNSClusterFirst,
								Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
							},
						},
					},
				},
			},
		},
	}

	for k, v := range errorCases {
		errs := ValidateCronJob(&v)
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

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}
