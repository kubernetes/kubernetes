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
	"fmt"
	"strings"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	jobutil "k8s.io/kubernetes/pkg/controller/job"
	"k8s.io/kubernetes/test/e2e/framework"
)

// WaitForAllJobPodsRunning wait for all pods for the Job named JobName in namespace ns to become Running.  Only use
// when pods will run for a long time, or it will be racy.
func WaitForAllJobPodsRunning(c clientset.Interface, ns, jobName string, parallelism int32) error {
	return wait.Poll(framework.Poll, JobTimeout, func() (bool, error) {
		pods, err := GetJobPods(c, ns, jobName)
		if err != nil {
			return false, err
		}
		count := int32(0)
		for _, p := range pods.Items {
			if p.Status.Phase == v1.PodRunning {
				count++
			}
		}
		return count == parallelism, nil
	})
}

// WaitForJobComplete uses c to wait for completions to complete for the Job jobName in namespace ns.
func WaitForJobComplete(c clientset.Interface, ns, jobName string, completions int32) error {
	return wait.Poll(framework.Poll, JobTimeout, func() (bool, error) {
		curr, err := c.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return curr.Status.Succeeded == completions, nil
	})
}

// WaitForJobFinish uses c to wait for the Job jobName in namespace ns to finish (either Failed or Complete).
func WaitForJobFinish(c clientset.Interface, ns, jobName string) error {
	return wait.PollImmediate(framework.Poll, JobTimeout, func() (bool, error) {
		curr, err := c.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return jobutil.IsJobFinished(curr), nil
	})
}

// WaitForJobFailure uses c to wait for up to timeout for the Job named jobName in namespace ns to fail.
func WaitForJobFailure(c clientset.Interface, ns, jobName string, timeout time.Duration, reason string) error {
	return wait.Poll(framework.Poll, timeout, func() (bool, error) {
		curr, err := c.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, c := range curr.Status.Conditions {
			if c.Type == batchv1.JobFailed && c.Status == v1.ConditionTrue {
				if reason == "" || reason == c.Reason {
					return true, nil
				}
			}
		}
		return false, nil
	})
}

// WaitForJobGone uses c to wait for up to timeout for the Job named jobName in namespace ns to be removed.
func WaitForJobGone(c clientset.Interface, ns, jobName string, timeout time.Duration) error {
	return wait.Poll(framework.Poll, timeout, func() (bool, error) {
		_, err := c.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
}

// EnsureAllJobPodsRunning uses c to check in the Job named jobName in ns
// is running, returning an error if the expected parallelism is not
// satisfied.
func EnsureAllJobPodsRunning(c clientset.Interface, ns, jobName string, parallelism int32) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{JobSelectorKey: jobName}))
	options := metav1.ListOptions{LabelSelector: label.String()}
	pods, err := c.CoreV1().Pods(ns).List(options)
	if err != nil {
		return err
	}
	podsSummary := make([]string, 0, parallelism)
	count := int32(0)
	for _, p := range pods.Items {
		if p.Status.Phase == v1.PodRunning {
			count++
		}
		podsSummary = append(podsSummary, fmt.Sprintf("%s (%s: %s)", p.ObjectMeta.Name, p.Status.Phase, p.Status.Message))
	}
	if count != parallelism {
		return fmt.Errorf("job has %d of %d expected running pods: %s", count, parallelism, strings.Join(podsSummary, ", "))
	}
	return nil
}

// WaitForAllJobPodsGone waits for all pods for the Job named jobName in namespace ns
// to be deleted.
func WaitForAllJobPodsGone(c clientset.Interface, ns, jobName string) error {
	return wait.PollImmediate(framework.Poll, JobTimeout, func() (bool, error) {
		pods, err := GetJobPods(c, ns, jobName)
		if err != nil {
			return false, err
		}
		return len(pods.Items) == 0, nil
	})
}

// WaitForJobDeleting uses c to wait for the Job jobName in namespace ns to have
// a non-nil deletionTimestamp (i.e. being deleted).
func WaitForJobDeleting(c clientset.Interface, ns, jobName string) error {
	return wait.PollImmediate(framework.Poll, JobTimeout, func() (bool, error) {
		curr, err := c.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return curr.ObjectMeta.DeletionTimestamp != nil, nil
	})
}
