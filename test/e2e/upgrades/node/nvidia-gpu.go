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

package node

import (
	"k8s.io/kubernetes/test/e2e/framework"
	e2ejob "k8s.io/kubernetes/test/e2e/framework/job"
	"k8s.io/kubernetes/test/e2e/scheduling"
	"k8s.io/kubernetes/test/e2e/upgrades"

	"github.com/onsi/ginkgo/v2"
)

const (
	completions = int32(1)
)

// NvidiaGPUUpgradeTest tests that gpu resource is available before and after
// a cluster upgrade.
type NvidiaGPUUpgradeTest struct {
}

// Name returns the tracking name of the test.
func (NvidiaGPUUpgradeTest) Name() string { return "nvidia-gpu-upgrade [sig-node] [sig-scheduling]" }

// Setup creates a job requesting gpu.
func (t *NvidiaGPUUpgradeTest) Setup(f *framework.Framework) {
	scheduling.SetupNVIDIAGPUNode(f, false)
	ginkgo.By("Creating a job requesting gpu")
	scheduling.StartJob(f, completions)
}

// Test waits for the upgrade to complete, and then verifies that the
// cuda pod started by the gpu job can successfully finish.
func (t *NvidiaGPUUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	<-done
	ginkgo.By("Verifying gpu job success")
	scheduling.VerifyJobNCompletions(f, completions)
	if upgrade == upgrades.MasterUpgrade || upgrade == upgrades.ClusterUpgrade {
		// MasterUpgrade should be totally hitless.
		job, err := e2ejob.GetJob(f.ClientSet, f.Namespace.Name, "cuda-add")
		framework.ExpectNoError(err)
		framework.ExpectEqual(job.Status.Failed, 0, "Job pods failed during master upgrade: %v", job.Status.Failed)
	}
}

// Teardown cleans up any remaining resources.
func (t *NvidiaGPUUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}
