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

	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
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
	return waitForJobPodsInPhase(ctx, c, ns, jobName, expectedCount, v1.PodRunning)
}

// WaitForJobPodsSucceeded wait for all pods for the Job named JobName in namespace ns to become Succeeded.
func WaitForJobPodsSucceeded(ctx context.Context, c clientset.Interface, ns, jobName string, expectedCount int32) error {
	return waitForJobPodsInPhase(ctx, c, ns, jobName, expectedCount, v1.PodSucceeded)
}

// waitForJobPodsInPhase wait for all pods for the Job named JobName in namespace ns to be in a given phase.
func waitForJobPodsInPhase(ctx context.Context, c clientset.Interface, ns, jobName string, expectedCount int32, phase v1.PodPhase) error {
	return wait.PollWithContext(ctx, framework.Poll, JobTimeout, func(ctx context.Context) (bool, error) {
		pods, err := GetJobPods(ctx, c, ns, jobName)
		if err != nil {
			return false, err
		}
		count := int32(0)
		for _, p := range pods.Items {
			if p.Status.Phase == phase {
				count++
			}
		}
		return count == expectedCount, nil
	})
}

// WaitForJobComplete uses c to wait for completions to complete for the Job jobName in namespace ns.
func WaitForJobComplete(ctx context.Context, c clientset.Interface, ns, jobName string, completions int32) error {
	return wait.PollWithContext(ctx, framework.Poll, JobTimeout, func(ctx context.Context) (bool, error) {
		curr, err := c.BatchV1().Jobs(ns).Get(ctx, jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return curr.Status.Succeeded == completions, nil
	})
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
func WaitForJobFailed(c clientset.Interface, ns, jobName string) error {
	return wait.PollImmediate(framework.Poll, JobTimeout, func() (bool, error) {
		curr, err := c.BatchV1().Jobs(ns).Get(context.TODO(), jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		return isJobFailed(curr), nil
	})
}

func isJobFailed(j *batchv1.Job) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == batchv1.JobFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

// WaitForJobFinish uses c to wait for the Job jobName in namespace ns to finish (either Failed or Complete).
func WaitForJobFinish(ctx context.Context, c clientset.Interface, ns, jobName string) error {
	return wait.PollUntilContextTimeout(ctx, framework.Poll, JobTimeout, true, func(ctx context.Context) (bool, error) {
		curr, err := c.BatchV1().Jobs(ns).Get(ctx, jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		return isJobFinished(curr), nil
	})
}

func isJobFinished(j *batchv1.Job) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == batchv1.JobComplete || c.Type == batchv1.JobFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}

	return false
}

// WaitForJobGone uses c to wait for up to timeout for the Job named jobName in namespace ns to be removed.
func WaitForJobGone(ctx context.Context, c clientset.Interface, ns, jobName string, timeout time.Duration) error {
	return wait.PollWithContext(ctx, framework.Poll, timeout, func(ctx context.Context) (bool, error) {
		_, err := c.BatchV1().Jobs(ns).Get(ctx, jobName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
}

// WaitForAllJobPodsGone waits for all pods for the Job named jobName in namespace ns
// to be deleted.
func WaitForAllJobPodsGone(ctx context.Context, c clientset.Interface, ns, jobName string) error {
	return wait.PollUntilContextTimeout(ctx, framework.Poll, JobTimeout, true, func(ctx context.Context) (bool, error) {
		pods, err := GetJobPods(ctx, c, ns, jobName)
		if err != nil {
			return false, err
		}
		return len(pods.Items) == 0, nil
	})
}

// WaitForJobState waits for a job to be matched to the given condition.
// The condition callback may use gomega.StopTrying to abort early.
func WaitForJobState(ctx context.Context, c clientset.Interface, ns, jobName string, timeout time.Duration, state JobState) error {
	return framework.Gomega().
		Eventually(ctx, framework.RetryNotFound(framework.GetObject(c.BatchV1().Jobs(ns).Get, jobName, metav1.GetOptions{}))).
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
