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
	"fmt"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fake "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

var submittedJob *batchv1.Job

func TestCreateJobFromCronJob(t *testing.T) {
	testNamespaceName := "test"
	testCronJobName := "test-cronjob"
	testImageName := "fake"

	expectedLabels := make(map[string]string)
	expectedAnnotations := make(map[string]string)
	expectedLabels["test-label"] = "test-value"
	expectedAnnotations["cronjob.kubernetes.io/instantiate"] = "manual"

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

	cronJobToCreate := &batchv1beta1.CronJob{
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
	clientset.PrependReactor("get", "cronjobs", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
		return true, cronJobToCreate, nil
	})
	clientset.PrependReactor("create", "jobs", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
		ca := action.(clienttesting.CreateAction)
		submittedJob = ca.GetObject().(*batchv1.Job)
		return true, expectJob, nil
	})

	f, _, _, _ := cmdtesting.NewAPIFactory()
	buf := bytes.NewBuffer([]byte{})

	mapper, _ := f.Object()
	cmdOptions := &CreateJobOptions{
		Out:           buf,
		OutputFormat:  "na",
		Namespace:     testNamespaceName,
		FromCronJob:   testCronJobName,
		V1Client:      clientset.BatchV1(),
		V1Beta1Client: clientset.BatchV1beta1(),
		Mapper:        mapper,
		PrintObject: func(obj runtime.Object) error {
			return nil
		},
	}
	cmdOptions.RunCreateJob()

	composedJobName := fmt.Sprintf("%s-manual-", testCronJobName)
	if !strings.HasPrefix(submittedJob.ObjectMeta.Name, composedJobName) {
		t.Errorf("expected '%s', got '%s'", composedJobName, submittedJob.ObjectMeta.Name)
	}

	annotationsArrayLength := len(submittedJob.Annotations)
	if annotationsArrayLength != 1 {
		t.Errorf("expected length of annotations array to be 1, got %d", annotationsArrayLength)
	}

	if v, ok := submittedJob.Annotations["cronjob.kubernetes.io/instantiate"]; ok {
		if v != "manual" {
			t.Errorf("expected annotation cronjob.kubernetes.io/instantiate to be 'manual', got '%s'", v)
		}
	} else {
		t.Errorf("expected annotation cronjob.kubernetes.io/instantiate to exist")
	}

	labelsArrayLength := len(submittedJob.Labels)
	if labelsArrayLength != 1 {
		t.Errorf("expected length of labels array to be 1, got %d", labelsArrayLength)
	}

	if v, ok := submittedJob.Labels["test-label"]; ok {
		if v != "test-value" {
			t.Errorf("expected label test-label to be 'test-value', got '%s'", v)
		}
	} else {
		t.Errorf("expected label test-label to exist")
	}

	containerArrayLength := len(submittedJob.Spec.Template.Spec.Containers)
	if containerArrayLength != 1 {
		t.Errorf("expected length of container array to be 1, got %d", containerArrayLength)
	}

	if submittedJob.Spec.Template.Spec.Containers[0].Image != testImageName {
		t.Errorf("expected '%s', got '%s'", testImageName, submittedJob.Spec.Template.Spec.Containers[0].Image)
	}
}
