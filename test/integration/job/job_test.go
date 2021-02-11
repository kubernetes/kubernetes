/*
Copyright 2021 The Kubernetes Authors.

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

package job

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/controller/job"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/pointer"
)

// TestNonParallelJob tests that a Job that only executes one Pod. The test
// recreates the Job controller at some points to make sure a new controller
// is able to pickup.
func TestNonParallelJob(t *testing.T) {
	closeFn, restConfig, clientSet, ns := setup(t, "simple")
	defer closeFn()
	ctx, cancel := startJobController(restConfig, clientSet)
	defer func() {
		cancel()
	}()

	jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{})
	if err != nil {
		t.Fatalf("Failed to create Job: %v", err)
	}
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Active: 1,
	})

	// Restarting controller.
	cancel()
	ctx, cancel = startJobController(restConfig, clientSet)

	// Failed Pod is replaced.
	if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodFailed, 1); err != nil {
		t.Fatalf("Failed setting phase %s on Job Pod: %v", v1.PodFailed, err)
	}
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Active: 1,
		Failed: 1,
	})

	// Restarting controller.
	cancel()
	ctx, cancel = startJobController(restConfig, clientSet)

	// No more Pods are created after the Pod succeeds.
	if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodSucceeded, 1); err != nil {
		t.Fatalf("Failed setting phase %s on Job Pod: %v", v1.PodSucceeded, err)
	}
	validateJobSucceeded(ctx, t, clientSet, jobObj)
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Failed:    1,
		Succeeded: 1,
	})
}

func TestParallelJob(t *testing.T) {
	closeFn, restConfig, clientSet, ns := setup(t, "parallel")
	defer closeFn()
	ctx, cancel := startJobController(restConfig, clientSet)
	defer cancel()

	jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{
		Spec: batchv1.JobSpec{
			Parallelism: pointer.Int32Ptr(5),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create Job: %v", err)
	}
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Active: 5,
	})
	// Failed Pods are replaced.
	if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodFailed, 2); err != nil {
		t.Fatalf("Failed setting phase %s on Job Pods: %v", v1.PodFailed, err)
	}
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Active: 5,
		Failed: 2,
	})
	// Once one Pod succeeds, no more Pods are created, even if some fail.
	if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodSucceeded, 1); err != nil {
		t.Fatalf("Failed setting phase %s on Job Pod: %v", v1.PodSucceeded, err)
	}
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Failed:    2,
		Succeeded: 1,
		Active:    4,
	})
	if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodFailed, 2); err != nil {
		t.Fatalf("Failed setting phase %s on Job Pods: %v", v1.PodFailed, err)
	}
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Failed:    4,
		Succeeded: 1,
		Active:    2,
	})
	// No more Pods are created after remaining Pods succeed.
	if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodSucceeded, 2); err != nil {
		t.Fatalf("Failed setting phase %s on Job Pods: %v", v1.PodSucceeded, err)
	}
	validateJobSucceeded(ctx, t, clientSet, jobObj)
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Failed:    4,
		Succeeded: 3,
	})
}

func TestParallelJobWithCompletions(t *testing.T) {
	closeFn, restConfig, clientSet, ns := setup(t, "completions")
	defer closeFn()
	ctx, cancel := startJobController(restConfig, clientSet)
	defer cancel()

	jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{
		Spec: batchv1.JobSpec{
			Parallelism: pointer.Int32Ptr(4),
			Completions: pointer.Int32Ptr(6),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create Job: %v", err)
	}
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Active: 4,
	})
	// Failed Pods are replaced.
	if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodFailed, 2); err != nil {
		t.Fatalf("Failed setting phase %s on Job Pods: %v", v1.PodFailed, err)
	}
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Active: 4,
		Failed: 2,
	})
	// Pods are created until the number of succeeded Pods equals completions.
	if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodSucceeded, 3); err != nil {
		t.Fatalf("Failed setting phase %s on Job Pod: %v", v1.PodSucceeded, err)
	}
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Failed:    2,
		Succeeded: 3,
		Active:    3,
	})
	// No more Pods are created after the Job completes.
	if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodSucceeded, 3); err != nil {
		t.Fatalf("Failed setting phase %s on Job Pods: %v", v1.PodSucceeded, err)
	}
	validateJobSucceeded(ctx, t, clientSet, jobObj)
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Failed:    2,
		Succeeded: 6,
	})
}

type podsByStatus struct {
	Active    int
	Failed    int
	Succeeded int
}

func validateJobPodsStatus(ctx context.Context, t *testing.T, clientSet clientset.Interface, jobObj *batchv1.Job, desired podsByStatus) {
	t.Helper()
	var actualCounts podsByStatus
	if err := wait.Poll(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		updatedJob, err := clientSet.BatchV1().Jobs(jobObj.Namespace).Get(ctx, jobObj.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get updated Job: %v", err)
		}
		actualCounts = podsByStatus{
			Active:    int(updatedJob.Status.Active),
			Succeeded: int(updatedJob.Status.Succeeded),
			Failed:    int(updatedJob.Status.Failed),
		}
		return cmp.Equal(actualCounts, desired), nil
	}); err != nil {
		diff := cmp.Diff(desired, actualCounts)
		t.Errorf("Waiting for Job Pods: %v\nPods (-want,+got):\n%s", err, diff)
	}
	// Verify active Pods. No need for another wait.Poll.
	pods, err := clientSet.CoreV1().Pods(jobObj.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list Job Pods: %v", err)
	}
	active := 0
	for _, pod := range pods.Items {
		if isPodOwnedByJob(&pod, jobObj) {
			if pod.Status.Phase == v1.PodPending || pod.Status.Phase == v1.PodRunning {
				active++
			}
		}
	}
	if active != desired.Active {
		t.Errorf("Found %d active Pods, want %d", active, desired.Active)
	}
}

func validateJobSucceeded(ctx context.Context, t *testing.T, clientSet clientset.Interface, jobObj *batchv1.Job) {
	t.Helper()
	if err := wait.Poll(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		j, err := clientSet.BatchV1().Jobs(jobObj.Namespace).Get(ctx, jobObj.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to obtain updated Job: %v", err)
		}
		for _, cond := range j.Status.Conditions {
			if cond.Type == batchv1.JobComplete && cond.Status == v1.ConditionTrue {
				return true, nil
			}
		}
		return false, nil
	}); err != nil {
		t.Errorf("Waiting for Job to succeed: %v", err)
	}
}

func setJobPodsPhase(ctx context.Context, clientSet clientset.Interface, jobObj *batchv1.Job, phase v1.PodPhase, cnt int) error {
	pods, err := clientSet.CoreV1().Pods(jobObj.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("listing Job Pods: %w", err)
	}
	for _, pod := range pods.Items {
		if cnt == 0 {
			break
		}
		if p := pod.Status.Phase; isPodOwnedByJob(&pod, jobObj) && p != v1.PodFailed && p != v1.PodSucceeded {
			pod.Status.Phase = phase
			_, err := clientSet.CoreV1().Pods(pod.Namespace).UpdateStatus(ctx, &pod, metav1.UpdateOptions{})
			if err != nil {
				return fmt.Errorf("updating Pod status: %w", err)
			}
			cnt--
		}
	}
	if cnt != 0 {
		return fmt.Errorf("couldn't set phase on %d Job Pods", cnt)
	}
	return nil
}

func isPodOwnedByJob(p *v1.Pod, j *batchv1.Job) bool {
	for _, owner := range p.ObjectMeta.OwnerReferences {
		if owner.Kind == "Job" && owner.UID == j.UID {
			return true
		}
	}
	return false
}

func createJobWithDefaults(ctx context.Context, clientSet clientset.Interface, ns string, jobObj *batchv1.Job) (*batchv1.Job, error) {
	if jobObj.Name == "" {
		jobObj.Name = "test-job"
	}
	if len(jobObj.Spec.Template.Spec.Containers) == 0 {
		jobObj.Spec.Template.Spec.Containers = []v1.Container{
			{Name: "foo", Image: "bar"},
		}
	}
	if jobObj.Spec.Template.Spec.RestartPolicy == "" {
		jobObj.Spec.Template.Spec.RestartPolicy = v1.RestartPolicyNever
	}
	return clientSet.BatchV1().Jobs(ns).Create(ctx, jobObj, metav1.CreateOptions{})
}

func setup(t *testing.T, nsBaseName string) (framework.CloseFunc, *restclient.Config, clientset.Interface, *v1.Namespace) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, server, masterCloseFn := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: server.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}
	ns := framework.CreateTestingNamespace(nsBaseName, server, t)
	closeFn := func() {
		framework.DeleteTestingNamespace(ns, server, t)
		masterCloseFn()
	}
	return closeFn, &config, clientSet, ns
}

func startJobController(restConfig *restclient.Config, clientSet clientset.Interface) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(context.Background())
	resyncPeriod := 12 * time.Hour
	informerSet := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(restConfig, "cronjob-informers")), resyncPeriod)
	jc := job.NewController(informerSet.Core().V1().Pods(), informerSet.Batch().V1().Jobs(), clientSet)
	informerSet.Start(ctx.Done())
	go jc.Run(1, ctx.Done())
	return ctx, cancel
}
