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

package apps

import (
	"context"
	"fmt"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/util/slice"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ejob "k8s.io/kubernetes/test/e2e/framework/job"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	dummyFinalizer = "k8s.io/dummy-finalizer"

	// JobTimeout is how long to wait for a job to finish.
	JobTimeout = 15 * time.Minute
)

var _ = SIGDescribe("TTLAfterFinished", func() {
	f := framework.NewDefaultFramework("ttlafterfinished")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.It("job should be deleted once it finishes after TTL seconds", func(ctx context.Context) {
		testFinishedJob(ctx, f)
	})
})

func cleanupJob(ctx context.Context, f *framework.Framework, job *batchv1.Job) {
	ns := f.Namespace.Name
	c := f.ClientSet

	framework.Logf("Remove the Job's dummy finalizer; the Job should be deleted cascadingly")
	removeFinalizerFunc := func(j *batchv1.Job) {
		j.ObjectMeta.Finalizers = slice.Remove(j.ObjectMeta.Finalizers, dummyFinalizer)
	}
	_, err := updateJobWithRetries(ctx, c, ns, job.Name, removeFinalizerFunc)
	framework.ExpectNoError(err)
	e2ejob.WaitForJobGone(ctx, c, ns, job.Name, wait.ForeverTestTimeout)

	err = e2ejob.WaitForAllJobPodsGone(ctx, c, ns, job.Name)
	framework.ExpectNoError(err)
}

func testFinishedJob(ctx context.Context, f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	parallelism := int32(1)
	completions := int32(1)
	backoffLimit := int32(2)
	ttl := int32(10)

	job := e2ejob.NewTestJob("randomlySucceedOrFail", "rand-non-local", v1.RestartPolicyNever, parallelism, completions, nil, backoffLimit)
	job.Spec.TTLSecondsAfterFinished = &ttl
	job.ObjectMeta.Finalizers = []string{dummyFinalizer}
	ginkgo.DeferCleanup(cleanupJob, f, job)

	framework.Logf("Create a Job %s/%s with TTL", ns, job.Name)
	job, err := e2ejob.CreateJob(ctx, c, ns, job)
	framework.ExpectNoError(err)

	framework.Logf("Wait for the Job to finish")
	err = e2ejob.WaitForJobFinish(ctx, c, ns, job.Name)
	framework.ExpectNoError(err)

	framework.Logf("Wait for TTL after finished controller to delete the Job")
	err = waitForJobDeleting(ctx, c, ns, job.Name)
	framework.ExpectNoError(err)

	framework.Logf("Check Job's deletionTimestamp and compare with the time when the Job finished")
	job, err = e2ejob.GetJob(ctx, c, ns, job.Name)
	framework.ExpectNoError(err)
	jobFinishTime := finishTime(job)
	finishTimeUTC := jobFinishTime.UTC()
	if jobFinishTime.IsZero() {
		framework.Fail("Expected job finish time not to be zero.")
	}

	deleteAtUTC := job.ObjectMeta.DeletionTimestamp.UTC()
	gomega.Expect(deleteAtUTC).NotTo(gomega.BeNil())

	expireAtUTC := finishTimeUTC.Add(time.Duration(ttl) * time.Second)
	if deleteAtUTC.Before(expireAtUTC) {
		framework.Fail("Expected job's deletion time to be after expiration time.")
	}
}

// finishTime returns finish time of the specified job.
func finishTime(finishedJob *batchv1.Job) metav1.Time {
	var finishTime metav1.Time
	for _, c := range finishedJob.Status.Conditions {
		if (c.Type == batchv1.JobComplete || c.Type == batchv1.JobFailed) && c.Status == v1.ConditionTrue {
			return c.LastTransitionTime
		}
	}
	return finishTime
}

// updateJobWithRetries updates job with retries.
func updateJobWithRetries(ctx context.Context, c clientset.Interface, namespace, name string, applyUpdate func(*batchv1.Job)) (job *batchv1.Job, err error) {
	jobs := c.BatchV1().Jobs(namespace)
	var updateErr error
	pollErr := wait.PollUntilContextTimeout(ctx, framework.Poll, JobTimeout, true, func(ctx context.Context) (bool, error) {
		if job, err = jobs.Get(ctx, name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(job)
		if job, err = jobs.Update(ctx, job, metav1.UpdateOptions{}); err == nil {
			framework.Logf("Updating job %s", name)
			return true, nil
		}
		updateErr = err
		return false, nil
	})
	if wait.Interrupted(pollErr) {
		pollErr = fmt.Errorf("couldn't apply the provided updated to job %q: %v", name, updateErr)
	}
	return job, pollErr
}

// waitForJobDeleting uses c to wait for the Job jobName in namespace ns to have
// a non-nil deletionTimestamp (i.e. being deleted).
func waitForJobDeleting(ctx context.Context, c clientset.Interface, ns, jobName string) error {
	return wait.PollUntilContextTimeout(ctx, framework.Poll, JobTimeout, true, func(ctx context.Context) (bool, error) {
		curr, err := c.BatchV1().Jobs(ns).Get(ctx, jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return curr.ObjectMeta.DeletionTimestamp != nil, nil
	})
}
