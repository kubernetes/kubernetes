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

package upgrades

import (
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

	"github.com/onsi/ginkgo"
)

// JobUpgradeTest is a test harness for batch Jobs.
type JobUpgradeTest struct {
	job       *batchv1.Job
	namespace string
}

// Name returns the tracking name of the test.
func (JobUpgradeTest) Name() string { return "[sig-apps] job-upgrade" }

// Setup starts a Job with a parallelism of 2 and 2 completions running.
func (t *JobUpgradeTest) Setup(f *framework.Framework) {
	t.namespace = f.Namespace.Name

	ginkgo.By("Creating a job")
	t.job = e2ejob.NewTestJob("notTerminate", "foo", v1.RestartPolicyOnFailure, 2, 2, nil, 6)
	job, err := e2ejob.CreateJob(f.ClientSet, t.namespace, t.job)
	t.job = job
	framework.ExpectNoError(err)

	ginkgo.By("Ensuring active pods == parallelism")
	err = e2ejob.WaitForAllJobPodsRunning(f.ClientSet, t.namespace, job.Name, 2)
	framework.ExpectNoError(err)
}

// Test verifies that the Jobs Pods are running after the an upgrade
func (t *JobUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	<-done
	ginkgo.By("Ensuring active pods == parallelism")
	err := ensureAllJobPodsRunning(f.ClientSet, t.namespace, t.job.Name, 2)
	framework.ExpectNoError(err)
}

// Teardown cleans up any remaining resources.
func (t *JobUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

// ensureAllJobPodsRunning uses c to check in the Job named jobName in ns
// is running, returning an error if the expected parallelism is not
// satisfied.
func ensureAllJobPodsRunning(c clientset.Interface, ns, jobName string, parallelism int32) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{e2ejob.JobSelectorKey: jobName}))
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
