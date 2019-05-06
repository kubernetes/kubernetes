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

package cronjob

import (
	"fmt"
	"net/http/httptest"
	"testing"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	clientbatchv1beta1 "k8s.io/client-go/kubernetes/typed/batch/v1beta1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/controller/cronjob"
	"k8s.io/kubernetes/pkg/controller/job"
	"k8s.io/kubernetes/test/integration/framework"
)

func setup(t *testing.T) (*httptest.Server, framework.CloseFunc, *cronjob.Controller, *job.JobController, informers.SharedInformerFactory, clientset.Interface, restclient.Config) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, server, closeFn := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: server.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informerSet := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "cronjob-informers")), resyncPeriod)
	cjc, err := cronjob.NewController(clientSet)
	if err != nil {
		t.Fatalf("Error creating CronJob controller: %v", err)
	}
	jc := job.NewJobController(informerSet.Core().V1().Pods(), informerSet.Batch().V1().Jobs(), clientSet)

	return server, closeFn, cjc, jc, informerSet, clientSet, config
}

func newCronJob(name, namespace, schedule string) *batchv1beta1.CronJob {
	zero64 := int64(0)
	zero32 := int32(0)
	return &batchv1beta1.CronJob{
		TypeMeta: metav1.TypeMeta{
			Kind:       "CronJob",
			APIVersion: "batch/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: batchv1beta1.CronJobSpec{
			Schedule:                   schedule,
			SuccessfulJobsHistoryLimit: &zero32,
			JobTemplate: batchv1beta1.JobTemplateSpec{
				Spec: batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers:                    []corev1.Container{{Name: "foo", Image: "bar"}},
							TerminationGracePeriodSeconds: &zero64,
							RestartPolicy:                 "Never",
						},
					},
				},
			},
		},
	}
}

func cleanupCronJobs(t *testing.T, cjClient clientbatchv1beta1.CronJobInterface, name string) {
	deletePropagation := metav1.DeletePropagationForeground
	err := cjClient.Delete(name, &metav1.DeleteOptions{PropagationPolicy: &deletePropagation})
	if err != nil {
		t.Errorf("Failed to delete CronJob: %v", err)
	}
}

func validateJobAndPod(t *testing.T, clientSet clientset.Interface, namespace string) {
	if err := wait.PollImmediate(1*time.Second, 120*time.Second, func() (bool, error) {
		jobs, err := clientSet.BatchV1().Jobs(namespace).List(metav1.ListOptions{})
		if err != nil {
			t.Fatalf("Failed to list jobs: %v", err)
		}

		if len(jobs.Items) == 0 {
			return false, nil
		}

		for _, j := range jobs.Items {
			ownerReferences := j.ObjectMeta.OwnerReferences
			if refCount := len(ownerReferences); refCount != 1 {
				return false, fmt.Errorf("job %s has %d OwnerReferences, expected only 1", j.Name, refCount)
			}

			reference := ownerReferences[0]
			if reference.Kind != "CronJob" {
				return false, fmt.Errorf("job %s has OwnerReference with Kind %s, expected CronJob", j.Name, reference.Kind)
			}
		}

		pods, err := clientSet.CoreV1().Pods(namespace).List(metav1.ListOptions{})
		if err != nil {
			t.Fatalf("Failed to list pods: %v", err)
		}

		if len(pods.Items) != 1 {
			return false, nil
		}

		for _, pod := range pods.Items {
			ownerReferences := pod.ObjectMeta.OwnerReferences
			if refCount := len(ownerReferences); refCount != 1 {
				return false, fmt.Errorf("pod %s has %d OwnerReferences, expected only 1", pod.Name, refCount)
			}

			reference := ownerReferences[0]
			if reference.Kind != "Job" {
				return false, fmt.Errorf("pod %s has OwnerReference with Kind %s, expected Job", pod.Name, reference.Kind)
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestCronJobLaunchesPodAndCleansUp(t *testing.T) {
	server, closeFn, cjc, jc, informerSet, clientSet, _ := setup(t)
	defer closeFn()

	cronJobName := "foo"
	namespaceName := "simple-cronjob-test"

	ns := framework.CreateTestingNamespace(namespaceName, server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	cjClient := clientSet.BatchV1beta1().CronJobs(ns.Name)

	stopCh := make(chan struct{})
	defer close(stopCh)

	informerSet.Start(stopCh)
	go cjc.Run(stopCh)
	go jc.Run(1, stopCh)

	_, err := cjClient.Create(newCronJob(cronJobName, ns.Name, "* * * * ?"))
	if err != nil {
		t.Fatalf("Failed to create CronJob: %v", err)
	}
	defer cleanupCronJobs(t, cjClient, cronJobName)

	validateJobAndPod(t, clientSet, namespaceName)
}
