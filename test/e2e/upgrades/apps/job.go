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
	batch "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/upgrades"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// JobUpgradeTest is a test harness for batch Jobs.
type JobUpgradeTest struct {
	job       *batch.Job
	namespace string
}

func (JobUpgradeTest) Name() string { return "[sig-apps] job-upgrade" }

// Setup starts a Job with a parallelism of 2 and 2 completions running.
func (t *JobUpgradeTest) Setup(f *framework.Framework) {
	t.namespace = f.Namespace.Name

	By("Creating a job")
	t.job = framework.NewTestJob("notTerminate", "foo", v1.RestartPolicyOnFailure, 2, 2, nil, 6)
	job, err := framework.CreateJob(f.ClientSet, t.namespace, t.job)
	t.job = job
	Expect(err).NotTo(HaveOccurred())

	By("Ensuring active pods == parallelism")
	err = framework.WaitForAllJobPodsRunning(f.ClientSet, t.namespace, job.Name, 2)
	Expect(err).NotTo(HaveOccurred())
}

// Test verifies that the Jobs Pods are running after the an upgrade
func (t *JobUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	<-done
	By("Ensuring active pods == parallelism")
	running, err := framework.CheckForAllJobPodsRunning(f.ClientSet, t.namespace, t.job.Name, 2)
	Expect(err).NotTo(HaveOccurred())
	Expect(running).To(BeTrue())
}

// Teardown cleans up any remaining resources.
func (t *JobUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}
