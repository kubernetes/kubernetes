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

package upgrades

import (
	"regexp"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/scheduling"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// NvidiaGPUUpgradeTest tests that gpu resource is available before and after
// a cluster upgrade.
type NvidiaGPUUpgradeTest struct {
}

func (NvidiaGPUUpgradeTest) Name() string { return "nvidia-gpu-upgrade [sig-node] [sig-scheduling]" }

// Setup creates a job requesting gpu.
func (t *NvidiaGPUUpgradeTest) Setup(f *framework.Framework) {
	scheduling.SetupNVIDIAGPUNode(f, false)
	By("Creating a job requesting gpu")
	t.startJob(f)
}

// Test waits for the upgrade to complete, and then verifies that the
// cuda pod started by the gpu job can successfully finish.
func (t *NvidiaGPUUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	<-done
	By("Verifying gpu job success")
	t.verifyJobPodSuccess(f)
	if upgrade == MasterUpgrade {
		// MasterUpgrade should be totally hitless.
		job, err := framework.GetJob(f.ClientSet, f.Namespace.Name, "cuda-add")
		Expect(err).NotTo(HaveOccurred())
		Expect(job.Status.Failed).To(BeZero(), "Job pods failed during master upgrade: %v", job.Status.Failed)
	}
}

// Teardown cleans up any remaining resources.
func (t *NvidiaGPUUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

// startJob creates a job that requests gpu and runs a simple cuda container.
func (t *NvidiaGPUUpgradeTest) startJob(f *framework.Framework) {
	var activeSeconds int64 = 3600
	// Specifies 100 completions to make sure the job life spans across the upgrade.
	testJob := framework.NewTestJob("succeed", "cuda-add", v1.RestartPolicyAlways, 1, 100, &activeSeconds, 6)
	testJob.Spec.Template.Spec = v1.PodSpec{
		RestartPolicy: v1.RestartPolicyOnFailure,
		Containers: []v1.Container{
			{
				Name:    "vector-addition",
				Image:   imageutils.GetE2EImage(imageutils.CudaVectorAdd),
				Command: []string{"/bin/sh", "-c", "./vectorAdd && sleep 60"},
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						framework.NVIDIAGPUResourceName: *resource.NewQuantity(1, resource.DecimalSI),
					},
				},
			},
		},
	}
	ns := f.Namespace.Name
	_, err := framework.CreateJob(f.ClientSet, ns, testJob)
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Created job %v", testJob)
	By("Waiting for gpu job pod start")
	err = framework.WaitForAllJobPodsRunning(f.ClientSet, ns, testJob.Name, 1)
	Expect(err).NotTo(HaveOccurred())
	By("Done with gpu job pod start")
}

// verifyJobPodSuccess verifies that the started cuda pod successfully passes.
func (t *NvidiaGPUUpgradeTest) verifyJobPodSuccess(f *framework.Framework) {
	// Wait for client pod to complete.
	ns := f.Namespace.Name
	err := framework.WaitForAllJobPodsRunning(f.ClientSet, f.Namespace.Name, "cuda-add", 1)
	Expect(err).NotTo(HaveOccurred())
	pods, err := framework.GetJobPods(f.ClientSet, f.Namespace.Name, "cuda-add")
	Expect(err).NotTo(HaveOccurred())
	createdPod := pods.Items[0].Name
	framework.Logf("Created pod %v", createdPod)
	f.PodClient().WaitForSuccess(createdPod, 5*time.Minute)
	logs, err := framework.GetPodLogs(f.ClientSet, ns, createdPod, "vector-addition")
	framework.ExpectNoError(err, "Should be able to get pod logs")
	framework.Logf("Got pod logs: %v", logs)
	regex := regexp.MustCompile("PASSED")
	Expect(regex.MatchString(logs)).To(BeTrue())
}
