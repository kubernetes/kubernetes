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
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCreateCronJob(t *testing.T) {
	cronjobName := "test-job"
	tests := map[string]struct {
		image    string
		command  []string
		schedule string
		restart  string
		expected *batchv1beta1.CronJob
	}{
		"just image and OnFailure restart policy": {
			image:    "busybox",
			schedule: "0/5 * * * ?",
			restart:  "OnFailure",
			expected: &batchv1beta1.CronJob{
				TypeMeta: metav1.TypeMeta{APIVersion: batchv1beta1.SchemeGroupVersion.String(), Kind: "CronJob"},
				ObjectMeta: metav1.ObjectMeta{
					Name: cronjobName,
				},
				Spec: batchv1beta1.CronJobSpec{
					Schedule: "0/5 * * * ?",
					JobTemplate: batchv1beta1.JobTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Name: cronjobName,
						},
						Spec: batchv1.JobSpec{
							Template: corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{
										{
											Name:  cronjobName,
											Image: "busybox",
										},
									},
									RestartPolicy: corev1.RestartPolicyOnFailure,
								},
							},
						},
					},
				},
			},
		},
		"image, command , schedule and Never restart policy": {
			image:    "busybox",
			command:  []string{"date"},
			schedule: "0/5 * * * ?",
			restart:  "Never",
			expected: &batchv1beta1.CronJob{
				TypeMeta: metav1.TypeMeta{APIVersion: batchv1beta1.SchemeGroupVersion.String(), Kind: "CronJob"},
				ObjectMeta: metav1.ObjectMeta{
					Name: cronjobName,
				},
				Spec: batchv1beta1.CronJobSpec{
					Schedule: "0/5 * * * ?",
					JobTemplate: batchv1beta1.JobTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Name: cronjobName,
						},
						Spec: batchv1.JobSpec{
							Template: corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{
										{
											Name:    cronjobName,
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
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreateCronJobOptions{
				Name:     cronjobName,
				Image:    tc.image,
				Command:  tc.command,
				Schedule: tc.schedule,
				Restart:  tc.restart,
			}
			cronjob := o.createCronJob()
			if !apiequality.Semantic.DeepEqual(cronjob, tc.expected) {
				t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, cronjob)
			}
		})
	}
}
