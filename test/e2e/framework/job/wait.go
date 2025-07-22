/*
Copyright 2019 The Kubernetes Authors.

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
	"time"

	"github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils/format"
	"k8s.io/utils/ptr"
)

// JobState is used to verify if Job matches a particular condition.
// If it matches, an empty string is returned.
// Otherwise, the string explains why the condition is not matched.
// This should be a short string. A dump of the job object will
// get added by the caller.
type JobState func(job *batchv1.Job) string

// WaitForJobPodsRunning wait for all pods for the Job named JobName in namespace ns to become Running.  Only use
// when pods will run for a long time, or it will be racy.
func WaitForJobPodsRunning(ctx context.Context, c clientset.Interface, ns, jobName string, expectedCount int32) error {
	return waitForJobPodsInPhase(ctx, c, ns, jobName, expectedCount, v1.PodRunning, JobTimeout)
}

// WaitForJobPodsRunningWithTimeout wait for all pods for the Job named JobName in namespace ns to become Running.  Only use
// when pods will run for a long time, or it will be racy. same as WaitForJobPodsRunning but with an additional timeout parameter
func WaitForJobPodsRunningWithTimeout(ctx context.Context, c clientset.Interface, ns, jobName string, expectedCount int32, timeout time.Duration) error {
	return waitForJobPodsInPhase(ctx, c, ns, jobName, expectedCount, v1.PodRunning, timeout)
}

// WaitForJobPodsSucceeded wait for all pods for the Job named JobName in namespace ns to become Succeeded.
func WaitForJobPodsSucceeded(ctx context.Context, c clientset.Interface, ns, jobName string, expectedCount int32) error {
	return waitForJobPodsInPhase(ctx, c, ns, jobName, expectedCount, v1.PodSucceeded, JobTimeout)
}

// waitForJobPodsInPhase wait for all pods for the Job named JobName in namespace ns to be in a given phase.
func waitForJobPodsInPhase(ctx context.Context, c clientset.Interface, ns, jobName string, expectedCount int32, phase v1.PodPhase, timeout time.Duration) error {
	get := func(ctx context.Context) (*v1.PodList, error) {
		return GetJobPods(ctx, c, ns, jobName)
	}
	match := func(pods *v1.PodList) (func() string, error) {
		count := int32(0)
		for _, p := range pods.Items {
			if p.Status.Phase == phase {
				count++
			}
		}
		if count == expectedCount {
			return nil, nil
		}
		return func() string {
			return fmt.Sprintf("job %q expected %d pods in %q phase, but got %d:\n%s",
				klog.KRef(ns, jobName), expectedCount, phase, count, format.Object(pods, 1))
		}, nil
	}
	return framework.Gomega().
		Eventually(ctx, framework.HandleRetry(get)).
		WithPolling(framework.Poll).
		WithTimeout(timeout).
		Should(framework.MakeMatcher(match))
}

// WaitForJobComplete uses c to wait for completions to complete for the Job jobName in namespace ns.
// This function checks if the number of succeeded Job Pods reached expected completions and
// the Job has a "Complete" condition with the expected reason.
func WaitForJobComplete(ctx context.Context, c clientset.Interface, ns, jobName string, reason string, completions int32) error {
	// This function is called by HandleRetry, which will retry
	// on transient API errors or stop polling in the case of other errors.
	get := func(ctx context.Context) (*batchv1.Job, error) {
		job, err := c.BatchV1().Jobs(ns).Get(ctx, jobName, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		if isJobFailed(job) {
			return nil, gomega.StopTrying("job failed while waiting for its completion").Attach("job", job)
		}
		return job, nil
	}
	match := func(job *batchv1.Job) (func() string, error) {
		if job.Status.Succeeded == completions {
			return nil, nil
		}
		return func() string {
			return fmt.Sprintf("expected job %q to have %v successful pods. got %v", klog.KObj(job), completions, job.Status.Succeeded)
		}, nil
	}
	err := framework.Gomega().
		Eventually(ctx, framework.HandleRetry(get)).
		WithTimeout(JobTimeout).
		WithPolling(framework.Poll).
		Should(framework.MakeMatcher(match))
	if err != nil {
		return err
	}
	return WaitForJobCondition(ctx, c, ns, jobName, batchv1.JobComplete, &reason)
}

// WaitForJobReady waits for particular value of the Job .status.ready field
func WaitForJobReady(ctx context.Context, c clientset.Interface, ns, jobName string, ready *int32) error {
	return WaitForJobState(ctx, c, ns, jobName, JobTimeout, func(job *batchv1.Job) string {
		if ptr.Equal(ready, job.Status.Ready) {
			return ""
		}
		return "job does not match intended ready status"
	})
}

// WaitForJobSuspend uses c to wait for suspend condition for the Job jobName in namespace ns.
func WaitForJobSuspend(ctx context.Context, c clientset.Interface, ns, jobName string) error {
	return WaitForJobState(ctx, c, ns, jobName, JobTimeout, func(job *batchv1.Job) string {
		for _, c := range job.Status.Conditions {
			if c.Type == batchv1.JobSuspended && c.Status == v1.ConditionTrue {
				return ""
			}
		}
		return "job should be suspended"
	})
}

// WaitForJobFailed uses c to wait for the Job jobName in namespace ns to fail
func WaitForJobFailed(ctx context.Context, c clientset.Interface, ns, jobName string) error {
	// This function is called by HandleRetry, which will retry
	// on transient API errors or stop polling in the case of other errors.
	get := func(ctx context.Context) (*batchv1.Job, error) {
		job, err := c.BatchV1().Jobs(ns).Get(ctx, jobName, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		if isJobCompleted(job) {
			return nil, gomega.StopTrying("job completed while waiting for its failure").Attach("job", job)
		}
		return job, nil
	}
	match := func(job *batchv1.Job) (func() string, error) {
		if isJobFailed(job) {
			return nil, nil
		}
		return func() string {
			return fmt.Sprintf("expected job %q to fail", klog.KObj(job))
		}, nil
	}
	return framework.Gomega().
		Eventually(ctx, framework.HandleRetry(get)).
		WithTimeout(JobTimeout).
		WithPolling(framework.Poll).
		Should(framework.MakeMatcher(match))
}

// WaitForJobCondition waits for the specified Job to have the expected condition with the specific reason.
// When the nil reason is passed, the "reason" string in the condition is
// not checked.
func WaitForJobCondition(ctx context.Context, c clientset.Interface, ns, jobName string, cType batchv1.JobConditionType, reason *string) error {
	match := func(job *batchv1.Job) (func() string, error) {
		for _, c := range job.Status.Conditions {
			if c.Type == cType && c.Status == v1.ConditionTrue {
				if reason == nil || *reason == c.Reason {
					return nil, nil
				}
			}
		}
		return func() string {
			return fmt.Sprintf("expected job %q to reach the expected condition %q with reason %q", klog.KObj(job), cType, ptr.Deref(reason, "<nil>"))
		}, nil
	}
	return framework.Gomega().
		Eventually(ctx, framework.GetObject(c.BatchV1().Jobs(ns).Get, jobName, metav1.GetOptions{})).
		WithTimeout(JobTimeout).
		WithPolling(framework.Poll).
		Should(framework.MakeMatcher(match))
}

// WaitForJobFinish uses c to wait for the Job jobName in namespace ns to finish (either Failed or Complete).
func WaitForJobFinish(ctx context.Context, c clientset.Interface, ns, jobName string) error {
	return WaitForJobFinishWithTimeout(ctx, c, ns, jobName, JobTimeout)
}

// WaitForJobFinishWithTimeout uses c to wait for the Job jobName in namespace ns to finish (either Failed or Complete).
func WaitForJobFinishWithTimeout(ctx context.Context, c clientset.Interface, ns, jobName string, timeout time.Duration) error {
	return framework.Gomega().
		Eventually(ctx, framework.GetObject(c.BatchV1().Jobs(ns).Get, jobName, metav1.GetOptions{})).
		WithPolling(framework.Poll).
		WithTimeout(timeout).
		Should(framework.MakeMatcher(func(job *batchv1.Job) (func() string, error) {
			if isJobFinished(job) {
				return nil, nil
			}
			return func() string {
				return fmt.Sprintf("expected job %q to be finished\n%s", klog.KObj(job), format.Object(job, 1))
			}, nil
		}))
}

func isJobFinished(j *batchv1.Job) bool {
	return isJobCompleted(j) || isJobFailed(j)
}

func isJobFailed(j *batchv1.Job) bool {
	return isConditionTrue(j, batchv1.JobFailed)
}

func isJobCompleted(j *batchv1.Job) bool {
	return isConditionTrue(j, batchv1.JobComplete)
}

func isConditionTrue(j *batchv1.Job, condition batchv1.JobConditionType) bool {
	for _, c := range j.Status.Conditions {
		if c.Type == condition && c.Status == v1.ConditionTrue {
			return true
		}
	}

	return false
}

// WaitForJobGone uses c to wait for up to timeout for the Job named jobName in namespace ns to be removed.
func WaitForJobGone(ctx context.Context, c clientset.Interface, ns, jobName string, timeout time.Duration) error {
	return framework.Gomega().
		Eventually(ctx, func(ctx context.Context) error {
			_, err := framework.GetObject(c.BatchV1().Jobs(ns).Get, jobName, metav1.GetOptions{})(ctx)
			return err
		}).
		WithPolling(framework.Poll).
		WithTimeout(timeout).
		Should(gomega.MatchError(apierrors.IsNotFound, fmt.Sprintf("that expected job %q to be gone", klog.KRef(ns, jobName))))
}

// WaitForAllJobPodsGone waits for all pods for the Job named jobName in namespace ns
// to be deleted.
func WaitForAllJobPodsGone(ctx context.Context, c clientset.Interface, ns, jobName string) error {
	get := func(ctx context.Context) (*v1.PodList, error) {
		return GetJobPods(ctx, c, ns, jobName)
	}
	return framework.Gomega().
		Eventually(ctx, framework.HandleRetry(get)).
		WithPolling(framework.Poll).
		WithTimeout(JobTimeout).
		Should(gomega.HaveField("Items", gomega.BeEmpty()))
}

// WaitForJobState waits for a job to be matched to the given state function.
func WaitForJobState(ctx context.Context, c clientset.Interface, ns, jobName string, timeout time.Duration, state JobState) error {
	return framework.Gomega().
		Eventually(ctx, framework.GetObject(c.BatchV1().Jobs(ns).Get, jobName, metav1.GetOptions{})).
		WithTimeout(timeout).
		Should(framework.MakeMatcher(func(job *batchv1.Job) (func() string, error) {
			matches := state(job)
			if matches == "" {
				return nil, nil
			}
			return func() string {
				return fmt.Sprintf("%v\n%s", matches, format.Object(job, 1))
			}, nil
		}))
}
