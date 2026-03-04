/*
Copyright 2018 The Kubernetes Authors.

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

package create

import (
	"strings"
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestCreateJobValidation(t *testing.T) {
	tests := map[string]struct {
		image    string
		command  []string
		from     string
		expected string
	}{
		"empty flags": {
			expected: "--image or --from must be specified",
		},
		"both image and from specified": {
			image:    "my-image",
			from:     "cronjob/xyz",
			expected: "--image or --from must be specified",
		},
		"from and command specified": {
			from:     "cronjob/xyz",
			command:  []string{"test", "command"},
			expected: "cannot specify --from and command",
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreateJobOptions{
				Image:   tc.image,
				From:    tc.from,
				Command: tc.command,
			}

			err := o.Validate()
			if err != nil && !strings.Contains(err.Error(), tc.expected) {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestCreateJob(t *testing.T) {
	jobName := "test-job"
	tests := map[string]struct {
		image    string
		command  []string
		expected *batchv1.Job
	}{
		"just image": {
			image: "busybox",
			expected: &batchv1.Job{
				TypeMeta: metav1.TypeMeta{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job"},
				ObjectMeta: metav1.ObjectMeta{
					Name: jobName,
				},
				Spec: batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  jobName,
									Image: "busybox",
								},
							},
							RestartPolicy: corev1.RestartPolicyNever,
						},
					},
				},
			},
		},
		"image and command": {
			image:   "busybox",
			command: []string{"date"},
			expected: &batchv1.Job{
				TypeMeta: metav1.TypeMeta{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job"},
				ObjectMeta: metav1.ObjectMeta{
					Name: jobName,
				},
				Spec: batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:    jobName,
									Image:   "busybox",
									Command: []string{"date"},
								},
							},
							RestartPolicy: corev1.RestartPolicyNever,
						},
					},
				},
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreateJobOptions{
				Name:    jobName,
				Image:   tc.image,
				Command: tc.command,
			}
			job := o.createJob()
			if !apiequality.Semantic.DeepEqual(job, tc.expected) {
				t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, job)
			}
		})
	}
}

func TestCreateJobFromCronJob(t *testing.T) {
	jobName := "test-job"
	cronJob := &batchv1.CronJob{
		Spec: batchv1.CronJobSpec{
			JobTemplate: batchv1.JobTemplateSpec{
				Spec: batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{Image: "test-image"},
							},
							RestartPolicy: corev1.RestartPolicyNever,
						},
					},
				},
			},
		},
	}
	tests := map[string]struct {
		from     *batchv1.CronJob
		expected *batchv1.Job
	}{
		"from CronJob": {
			from: cronJob,
			expected: &batchv1.Job{
				TypeMeta: metav1.TypeMeta{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job"},
				ObjectMeta: metav1.ObjectMeta{
					Name:        jobName,
					Annotations: map[string]string{"cronjob.kubernetes.io/instantiate": "manual"},
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: batchv1.SchemeGroupVersion.String(),
							Kind:       "CronJob",
							Name:       cronJob.GetName(),
							UID:        cronJob.GetUID(),
							Controller: ptr.To(true),
						},
					},
				},
				Spec: batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{Image: "test-image"},
							},
							RestartPolicy: corev1.RestartPolicyNever,
						},
					},
				},
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreateJobOptions{
				Name: jobName,
			}
			job := o.createJobFromCronJob(tc.from)

			if !apiequality.Semantic.DeepEqual(job, tc.expected) {
				t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, job)
			}
		})
	}
}
