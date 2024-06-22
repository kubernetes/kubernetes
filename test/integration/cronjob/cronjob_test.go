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
	"context"
	"fmt"
	"testing"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	clientbatchv1 "k8s.io/client-go/kubernetes/typed/batch/v1"
	restclient "k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/cronjob"
	"k8s.io/kubernetes/pkg/controller/job"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func setup(ctx context.Context, t *testing.T) (kubeapiservertesting.TearDownFunc, *cronjob.ControllerV2, *job.Controller, informers.SharedInformerFactory, clientset.Interface) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())

	config := restclient.CopyConfig(server.ClientConfig)
	clientSet, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informerSet := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "cronjob-informers")), resyncPeriod)
	cjc, err := cronjob.NewControllerV2(ctx, informerSet.Batch().V1().Jobs(), informerSet.Batch().V1().CronJobs(), clientSet)
	if err != nil {
		t.Fatalf("Error creating CronJob controller: %v", err)
	}
	jc, err := job.NewController(ctx, informerSet.Core().V1().Pods(), informerSet.Batch().V1().Jobs(), clientSet)
	if err != nil {
		t.Fatalf("Error creating Job controller: %v", err)
	}

	return server.TearDownFn, cjc, jc, informerSet, clientSet
}

func newCronJob(name, namespace, schedule string) *batchv1.CronJob {
	zero64 := int64(0)
	zero32 := int32(0)
	return &batchv1.CronJob{
		TypeMeta: metav1.TypeMeta{
			Kind:       "CronJob",
			APIVersion: "batch/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: batchv1.CronJobSpec{
			Schedule:                   schedule,
			SuccessfulJobsHistoryLimit: &zero32,
			JobTemplate: batchv1.JobTemplateSpec{
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

func cleanupCronJobs(t *testing.T, cjClient clientbatchv1.CronJobInterface, name string) {
	deletePropagation := metav1.DeletePropagationForeground
	err := cjClient.Delete(context.TODO(), name, metav1.DeleteOptions{PropagationPolicy: &deletePropagation})
	if err != nil {
		t.Errorf("Failed to delete CronJob: %v", err)
	}
}

func validateJobAndPod(t *testing.T, clientSet clientset.Interface, namespace string) {
	if err := wait.PollImmediate(1*time.Second, 120*time.Second, func() (bool, error) {
		jobs, err := clientSet.BatchV1().Jobs(namespace).List(context.TODO(), metav1.ListOptions{})
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

		pods, err := clientSet.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{})
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
	tCtx := ktesting.Init(t)

	closeFn, cjc, jc, informerSet, clientSet := setup(tCtx, t)
	defer closeFn()

	// When shutting down, cancel must be called before closeFn.
	defer tCtx.Cancel("test has completed")

	cronJobName := "foo"
	namespaceName := "simple-cronjob-test"

	ns := framework.CreateNamespaceOrDie(clientSet, namespaceName, t)
	defer framework.DeleteNamespaceOrDie(clientSet, ns, t)

	cjClient := clientSet.BatchV1().CronJobs(ns.Name)

	informerSet.Start(tCtx.Done())
	go cjc.Run(tCtx, 1)
	go jc.Run(tCtx, 1)

	_, err := cjClient.Create(tCtx, newCronJob(cronJobName, ns.Name, "* * * * ?"), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create CronJob: %v", err)
	}
	defer cleanupCronJobs(t, cjClient, cronJobName)

	validateJobAndPod(t, clientSet, namespaceName)
}
