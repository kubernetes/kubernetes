/*
Copyright 2017 The Kubernetes Authors.

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
	"strings"

	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ejob "k8s.io/kubernetes/test/e2e/framework/job"
	"k8s.io/kubernetes/test/e2e/upgrades"

	"github.com/onsi/ginkgo/v2"
)

// JobUpgradeTest is a test harness for batch Jobs.
type JobUpgradeTest struct {
	job       *batchv1.Job
	namespace string
}

// Name returns the tracking name of the test.
func (JobUpgradeTest) Name() string { return "[sig-apps] job-upgrade" }

// Setup starts a Job with a parallelism of 2 and 2 completions running.
func (t *JobUpgradeTest) Setup(ctx context.Context, f *framework.Framework) {
	t.namespace = f.Namespace.Name

	ginkgo.By("Creating a job")
	t.job = e2ejob.NewTestJob("neverTerminate", "foo", v1.RestartPolicyOnFailure, 2, 2, nil, 6)
	job, err := e2ejob.CreateJob(ctx, f.ClientSet, t.namespace, t.job)
	t.job = job
	framework.ExpectNoError(err)

	ginkgo.By("Ensuring active pods == parallelism")
	err = e2ejob.WaitForJobPodsRunning(ctx, f.ClientSet, t.namespace, job.Name, 2)
	framework.ExpectNoError(err)
}

// Test verifies that the Jobs Pods are running after the an upgrade
func (t *JobUpgradeTest) Test(ctx context.Context, f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	<-done
	ginkgo.By("Ensuring job is running")
	err := ensureJobRunning(ctx, f.ClientSet, t.namespace, t.job.Name)
	framework.ExpectNoError(err)
	ginkgo.By("Ensuring active pods == parallelism")
	err = ensureAllJobPodsRunning(ctx, f.ClientSet, t.namespace, t.job.Name, 2)
	framework.ExpectNoError(err)
}

// Teardown cleans up any remaining resources.
func (t *JobUpgradeTest) Teardown(ctx context.Context, f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

// ensureAllJobPodsRunning uses c to check if the Job named jobName in ns
// is running, returning an error if the expected parallelism is not
// satisfied.
func ensureAllJobPodsRunning(ctx context.Context, c clientset.Interface, ns, jobName string, parallelism int32) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{e2ejob.JobSelectorKey: jobName}))
	options := metav1.ListOptions{LabelSelector: label.String()}
	pods, err := c.CoreV1().Pods(ns).List(ctx, options)
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

// ensureJobRunning uses c to check if the Job named jobName in ns is running,
// (not completed, nor failed, nor suspended) returning an error if it can't
// read the job or when it's not running
func ensureJobRunning(ctx context.Context, c clientset.Interface, ns, jobName string) error {
	job, err := e2ejob.GetJob(ctx, c, ns, jobName)
	if err != nil {
		return err
	}
	for _, c := range job.Status.Conditions {
		if (c.Type == batchv1.JobComplete || c.Type == batchv1.JobFailed || c.Type == batchv1.JobSuspended) && c.Status == v1.ConditionTrue {
			return fmt.Errorf("job is not running %#v", job)
		}
	}
	return nil
}
