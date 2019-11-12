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
	"reflect"
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
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
			actual := o.createCronJob()
			if !equality.Semantic.DeepEqual(tc.expected, actual) {
				t.Errorf("%s", diff.ObjectReflectDiff(tc.expected, actual))
			}
		})
	}
}

func TestValidateCreateCronJob(t *testing.T) {
	tests := map[string]struct {
		options   *CreateCronJobOptions
		expectErr bool
	}{
		"test-missing-name": {
			options: &CreateCronJobOptions{
				Image:    "busybox",
				Schedule: "0/5 * * * ?",
			},
			expectErr: true,
		},
		"test-missing-image": {
			options: &CreateCronJobOptions{
				Name:     "my-cronjob",
				Schedule: "0/5 * * * ?",
			},
			expectErr: true,
		},
		"test-missing-schedule": {
			options: &CreateCronJobOptions{
				Name:  "my-cronjob",
				Image: "busybox",
			},
			expectErr: true,
		},
		"test-valid-case": {
			options: &CreateCronJobOptions{
				Name:     "my-conjob",
				Image:    "busybox",
				Schedule: "0/5 * * * ?",
			},
			expectErr: false,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			err := test.options.Validate()
			if test.expectErr && err == nil {
				t.Errorf("expected error but validation passed")
			}
			if !test.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestCompleteCreateCronJob(t *testing.T) {
	// TODO: Test that CreateCronJobOptions.Command is properly set.
	//       When setting .Command, CompleteCreateCronJob relies heavily
	//       on Cobra, which makes testing difficult.

	defaultTestName := "my-cronjob"
	defaultTestDryRun := true
	defaultTestRestart := "Never"

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.Client = &fake.RESTClient{}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	tests := map[string]struct {
		params    []string
		dryRun    bool
		options   *CreateCronJobOptions
		expected  *CreateCronJobOptions
		expectErr bool
	}{
		"test-missing-name": {
			params:    []string{},
			options:   &CreateCronJobOptions{},
			expected:  &CreateCronJobOptions{},
			expectErr: true,
		},
		"test-missing-restart": {
			params: []string{defaultTestName},
			options: &CreateCronJobOptions{
				PrintFlags: genericclioptions.NewPrintFlags("created"),
			},
			expected: &CreateCronJobOptions{
				Name:    defaultTestName,
				Restart: "OnFailure",
			},
			expectErr: false,
		},
		"test-valid-complete-case": {
			params: []string{defaultTestName},
			dryRun: defaultTestDryRun,
			options: &CreateCronJobOptions{
				PrintFlags: genericclioptions.NewPrintFlags("created"),
				Restart:    defaultTestRestart,
			},
			expected: &CreateCronJobOptions{
				Name:    defaultTestName,
				DryRun:  defaultTestDryRun,
				Restart: defaultTestRestart,
			},
			expectErr: false,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			cmd := NewCmdCreateRole(tf, genericclioptions.NewTestIOStreamsDiscard())

			if test.dryRun {
				cmd.Flags().Set("dry-run", "true")
			}

			err := test.options.Complete(tf, cmd, test.params)

			if test.expectErr && err == nil {
				t.Errorf("expected error but none was returned")
			}
			if !test.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			fields := map[string]struct {
				expected, actual interface{}
			}{
				".Name": {
					expected: test.expected.Name,
					actual:   test.options.Name,
				},
				".DryRun": {
					expected: test.expected.DryRun,
					actual:   test.options.DryRun,
				},
				".Restart": {
					expected: test.expected.Restart,
					actual:   test.options.Restart,
				},
			}

			for name, value := range fields {
				if !reflect.DeepEqual(value.expected, value.actual) {
					t.Errorf("mismatched field %q:\n%s", name, diff.ObjectReflectDiff(value.expected, value.actual))
				}
			}
		})
	}
}

func TestRunCreateCronJob(t *testing.T) {
	defaultTestName := "my-cronjob"
	defaultTestNamespace := "my-namespace"
	defaultTestImage := "busybox"
	defaultTestCommand := []string{"date"}
	defaultTestSchedule := "0/5 * * * ?"
	defaultTestRestart := "OnFailure"

	defaultExpectedJobSpec := batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name:    defaultTestName,
						Image:   defaultTestImage,
						Command: defaultTestCommand,
					},
				},
				RestartPolicy: corev1.RestartPolicyOnFailure,
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.Client = &fake.RESTClient{}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	tests := map[string]struct {
		options  *CreateCronJobOptions
		expected *batchv1beta1.CronJob
	}{
		"test-valid-case": {
			options: &CreateCronJobOptions{
				Name:      defaultTestName,
				Namespace: defaultTestNamespace,
				DryRun:    true,
				Image:     defaultTestImage,
				Command:   defaultTestCommand,
				Schedule:  defaultTestSchedule,
				Restart:   defaultTestRestart,
			},
			expected: &batchv1beta1.CronJob{
				TypeMeta: metav1.TypeMeta{
					APIVersion: batchv1beta1.SchemeGroupVersion.String(),
					Kind:       "CronJob",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: defaultTestName,
				},
				Spec: batchv1beta1.CronJobSpec{
					Schedule: defaultTestSchedule,
					JobTemplate: batchv1beta1.JobTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Name: defaultTestName,
						},
						Spec: defaultExpectedJobSpec,
					},
				},
			},
		},
		"test-valid-case-enforce-namespace": {
			options: &CreateCronJobOptions{
				Name:             defaultTestName,
				Namespace:        defaultTestNamespace,
				EnforceNamespace: true,
				DryRun:           true,
				Image:            defaultTestImage,
				Command:          defaultTestCommand,
				Schedule:         defaultTestSchedule,
				Restart:          defaultTestRestart,
			},
			expected: &batchv1beta1.CronJob{
				TypeMeta: metav1.TypeMeta{
					APIVersion: batchv1beta1.SchemeGroupVersion.String(),
					Kind:       "CronJob",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      defaultTestName,
					Namespace: defaultTestNamespace,
				},
				Spec: batchv1beta1.CronJobSpec{
					Schedule: defaultTestSchedule,
					JobTemplate: batchv1beta1.JobTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Name:      defaultTestName,
							Namespace: defaultTestNamespace,
						},
						Spec: defaultExpectedJobSpec,
					},
				},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			var err error
			clientset, err := tf.KubernetesClientSet()
			if err != nil {
				t.Fatal(err)
			}
			test.options.Client = clientset.BatchV1beta1()

			var actual *batchv1beta1.CronJob
			test.options.PrintObj = func(obj runtime.Object) error {
				actual = obj.(*batchv1beta1.CronJob)
				return nil
			}

			err = test.options.Run()
			if err != nil {
				t.Fatal(err)
			}

			if !equality.Semantic.DeepEqual(test.expected, actual) {
				t.Errorf("%s", diff.ObjectReflectDiff(test.expected, actual))
			}
		})
	}
}
