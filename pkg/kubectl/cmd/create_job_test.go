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

package cmd

import (
	"bytes"
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	fake "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestCreateJobFromCronJob(t *testing.T) {
	var submittedJob *batchv1.Job
	testNamespaceName := "test"
	testCronJobName := "test-cronjob"
	testJobName := "test-job"
	testImageName := "fake"

	expectedLabels := make(map[string]string)
	expectedAnnotations := make(map[string]string)
	expectedLabels["test-label"] = "test-value"

	expectJob := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:   testNamespaceName,
			Labels:      expectedLabels,
			Annotations: expectedAnnotations,
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Image: testImageName},
					},
				},
			},
		},
	}

	cronJob := &batchv1beta1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testCronJobName,
			Namespace: testNamespaceName,
		},
		Spec: batchv1beta1.CronJobSpec{
			Schedule: "* * * * *",
			JobTemplate: batchv1beta1.JobTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: testNamespaceName,
					Labels:    expectedLabels,
				},
				Spec: expectJob.Spec,
			},
		},
	}

	clientset := fake.Clientset{}
	clientset.PrependReactor("create", "jobs", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
		ca := action.(clienttesting.CreateAction)
		submittedJob = ca.GetObject().(*batchv1.Job)
		return true, expectJob, nil
	})
	f := cmdtesting.NewTestFactory()
	buf := bytes.NewBuffer([]byte{})
	cmdOptions := &CreateJobOptions{
		Name:      testJobName,
		Namespace: testNamespaceName,
		Client:    clientset.BatchV1(),
		Out:       buf,
		Cmd:       NewCmdCreateJob(f, buf),
	}

	err := cmdOptions.createJob(cronJob)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if submittedJob.ObjectMeta.Name != testJobName {
		t.Errorf("expected '%s', got '%s'", testJobName, submittedJob.ObjectMeta.Name)
	}

	if l := len(submittedJob.Annotations); l != 1 {
		t.Errorf("expected length of annotations array to be 1, got %d", l)
	}
	if v, ok := submittedJob.Annotations["cronjob.kubernetes.io/instantiate"]; !ok || v != "manual" {
		t.Errorf("expected annotation cronjob.kubernetes.io/instantiate=manual to exist, got '%s'", v)
	}

	if l := len(submittedJob.Labels); l != 1 {
		t.Errorf("expected length of labels array to be 1, got %d", l)
	}
	if v, ok := submittedJob.Labels["test-label"]; !ok || v != "test-value" {
		t.Errorf("expected label test-label=test-value to to exist, got '%s'", v)
	}

	if l := len(submittedJob.Spec.Template.Spec.Containers); l != 1 {
		t.Errorf("expected length of container array to be 1, got %d", l)
	}
	if submittedJob.Spec.Template.Spec.Containers[0].Image != testImageName {
		t.Errorf("expected '%s', got '%s'", testImageName, submittedJob.Spec.Template.Spec.Containers[0].Image)
	}
}
