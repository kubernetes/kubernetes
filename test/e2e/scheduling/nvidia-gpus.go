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

package scheduling

import (
	"os"
	"regexp"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/gpu"
	jobutil "k8s.io/kubernetes/test/e2e/framework/job"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	testPodNamePrefix = "nvidia-gpu-"
	// Nvidia driver installation can take upwards of 5 minutes.
	driverInstallTimeout = 10 * time.Minute
)

var (
	gpuResourceName v1.ResourceName
	dsYamlURL       string
)

func makeCudaAdditionDevicePluginTestPod() *v1.Pod {
	podName := testPodNamePrefix + string(uuid.NewUUID())
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  "vector-addition-cuda8",
					Image: imageutils.GetE2EImage(imageutils.CudaVectorAdd),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							gpuResourceName: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
				},
				{
					Name:  "vector-addition-cuda10",
					Image: imageutils.GetE2EImage(imageutils.CudaVectorAdd2),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							gpuResourceName: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
				},
			},
		},
	}
	return testPod
}

func logOSImages(f *framework.Framework) {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	for _, node := range nodeList.Items {
		e2elog.Logf("Nodename: %v, OS Image: %v", node.Name, node.Status.NodeInfo.OSImage)
	}
}

func areGPUsAvailableOnAllSchedulableNodes(f *framework.Framework) bool {
	e2elog.Logf("Getting list of Nodes from API server")
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	for _, node := range nodeList.Items {
		if node.Spec.Unschedulable {
			continue
		}
		e2elog.Logf("gpuResourceName %s", gpuResourceName)
		if val, ok := node.Status.Capacity[gpuResourceName]; !ok || val.Value() == 0 {
			e2elog.Logf("Nvidia GPUs not available on Node: %q", node.Name)
			return false
		}
	}
	e2elog.Logf("Nvidia GPUs exist on all schedulable nodes")
	return true
}

func getGPUsAvailable(f *framework.Framework) int64 {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	var gpusAvailable int64
	for _, node := range nodeList.Items {
		if val, ok := node.Status.Allocatable[gpuResourceName]; ok {
			gpusAvailable += (&val).Value()
		}
	}
	return gpusAvailable
}

// SetupNVIDIAGPUNode install Nvidia Drivers and wait for Nvidia GPUs to be available on nodes
func SetupNVIDIAGPUNode(f *framework.Framework, setupResourceGatherer bool) *framework.ContainerResourceGatherer {
	logOSImages(f)

	dsYamlURLFromEnv := os.Getenv("NVIDIA_DRIVER_INSTALLER_DAEMONSET")
	if dsYamlURLFromEnv != "" {
		dsYamlURL = dsYamlURLFromEnv
	} else {
		dsYamlURL = "https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/daemonset.yaml"
	}
	gpuResourceName = gpu.NVIDIAGPUResourceName

	e2elog.Logf("Using %v", dsYamlURL)
	// Creates the DaemonSet that installs Nvidia Drivers.
	ds, err := framework.DsFromManifest(dsYamlURL)
	framework.ExpectNoError(err)
	ds.Namespace = f.Namespace.Name
	_, err = f.ClientSet.AppsV1().DaemonSets(f.Namespace.Name).Create(ds)
	framework.ExpectNoError(err, "failed to create nvidia-driver-installer daemonset")
	e2elog.Logf("Successfully created daemonset to install Nvidia drivers.")

	pods, err := e2epod.WaitForControlledPods(f.ClientSet, ds.Namespace, ds.Name, extensionsinternal.Kind("DaemonSet"))
	framework.ExpectNoError(err, "failed to get pods controlled by the nvidia-driver-installer daemonset")

	devicepluginPods, err := e2epod.WaitForControlledPods(f.ClientSet, "kube-system", "nvidia-gpu-device-plugin", extensionsinternal.Kind("DaemonSet"))
	if err == nil {
		e2elog.Logf("Adding deviceplugin addon pod.")
		pods.Items = append(pods.Items, devicepluginPods.Items...)
	}

	var rsgather *framework.ContainerResourceGatherer
	if setupResourceGatherer {
		e2elog.Logf("Starting ResourceUsageGather for the created DaemonSet pods.")
		rsgather, err = framework.NewResourceUsageGatherer(f.ClientSet, framework.ResourceGathererOptions{InKubemark: false, Nodes: framework.AllNodes, ResourceDataGatheringPeriod: 2 * time.Second, ProbeDuration: 2 * time.Second, PrintVerboseLogs: true}, pods)
		framework.ExpectNoError(err, "creating ResourceUsageGather for the daemonset pods")
		go rsgather.StartGatheringData()
	}

	// Wait for Nvidia GPUs to be available on nodes
	e2elog.Logf("Waiting for drivers to be installed and GPUs to be available in Node Capacity...")
	gomega.Eventually(func() bool {
		return areGPUsAvailableOnAllSchedulableNodes(f)
	}, driverInstallTimeout, time.Second).Should(gomega.BeTrue())

	return rsgather
}

func getGPUsPerPod() int64 {
	var gpusPerPod int64
	gpuPod := makeCudaAdditionDevicePluginTestPod()
	for _, container := range gpuPod.Spec.Containers {
		if val, ok := container.Resources.Limits[gpuResourceName]; ok {
			gpusPerPod += (&val).Value()
		}
	}
	return gpusPerPod
}

func testNvidiaGPUs(f *framework.Framework) {
	rsgather := SetupNVIDIAGPUNode(f, true)
	gpuPodNum := getGPUsAvailable(f) / getGPUsPerPod()
	e2elog.Logf("Creating %d pods and have the pods run a CUDA app", gpuPodNum)
	podList := []*v1.Pod{}
	for i := int64(0); i < gpuPodNum; i++ {
		podList = append(podList, f.PodClient().Create(makeCudaAdditionDevicePluginTestPod()))
	}
	e2elog.Logf("Wait for all test pods to succeed")
	// Wait for all pods to succeed
	for _, pod := range podList {
		f.PodClient().WaitForSuccess(pod.Name, 5*time.Minute)
		logContainers(f, pod)
	}

	e2elog.Logf("Stopping ResourceUsageGather")
	constraints := make(map[string]framework.ResourceConstraint)
	// For now, just gets summary. Can pass valid constraints in the future.
	summary, err := rsgather.StopAndSummarize([]int{50, 90, 100}, constraints)
	f.TestSummaries = append(f.TestSummaries, summary)
	framework.ExpectNoError(err, "getting resource usage summary")
}

func logContainers(f *framework.Framework, pod *v1.Pod) {
	for _, container := range pod.Spec.Containers {
		logs, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, container.Name)
		framework.ExpectNoError(err, "Should be able to get container logs for container: %s", container.Name)
		e2elog.Logf("Got container logs for %s:\n%v", container.Name, logs)
	}
}

var _ = SIGDescribe("[Feature:GPUDevicePlugin]", func() {
	f := framework.NewDefaultFramework("device-plugin-gpus")
	ginkgo.It("run Nvidia GPU Device Plugin tests", func() {
		testNvidiaGPUs(f)
	})
})

func testNvidiaGPUsJob(f *framework.Framework) {
	_ = SetupNVIDIAGPUNode(f, false)
	// Job set to have 5 completions with parallelism of 1 to ensure that it lasts long enough to experience the node recreation
	completions := int32(5)
	ginkgo.By("Starting GPU job")
	StartJob(f, completions)

	job, err := jobutil.GetJob(f.ClientSet, f.Namespace.Name, "cuda-add")
	framework.ExpectNoError(err)

	// make sure job is running by waiting for its first pod to start running
	err = jobutil.WaitForAllJobPodsRunning(f.ClientSet, f.Namespace.Name, job.Name, 1)
	framework.ExpectNoError(err)

	numNodes, err := e2enode.TotalRegistered(f.ClientSet)
	framework.ExpectNoError(err)
	nodes, err := e2enode.CheckReady(f.ClientSet, numNodes, framework.NodeReadyInitialTimeout)
	framework.ExpectNoError(err)

	ginkgo.By("Recreating nodes")
	err = gce.RecreateNodes(f.ClientSet, nodes)
	framework.ExpectNoError(err)
	ginkgo.By("Done recreating nodes")

	ginkgo.By("Waiting for gpu job to finish")
	err = jobutil.WaitForJobFinish(f.ClientSet, f.Namespace.Name, job.Name)
	framework.ExpectNoError(err)
	ginkgo.By("Done with gpu job")

	gomega.Expect(job.Status.Failed).To(gomega.BeZero(), "Job pods failed during node recreation: %v", job.Status.Failed)

	VerifyJobNCompletions(f, completions)
}

// StartJob starts a simple CUDA job that requests gpu and the specified number of completions
func StartJob(f *framework.Framework, completions int32) {
	var activeSeconds int64 = 3600
	testJob := jobutil.NewTestJob("succeed", "cuda-add", v1.RestartPolicyAlways, 1, completions, &activeSeconds, 6)
	testJob.Spec.Template.Spec = v1.PodSpec{
		RestartPolicy: v1.RestartPolicyOnFailure,
		Containers: []v1.Container{
			{
				Name:    "vector-addition",
				Image:   imageutils.GetE2EImage(imageutils.CudaVectorAdd),
				Command: []string{"/bin/sh", "-c", "./vectorAdd && sleep 60"},
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						gpuResourceName: *resource.NewQuantity(1, resource.DecimalSI),
					},
				},
			},
		},
	}
	ns := f.Namespace.Name
	_, err := jobutil.CreateJob(f.ClientSet, ns, testJob)
	framework.ExpectNoError(err)
	e2elog.Logf("Created job %v", testJob)
}

// VerifyJobNCompletions verifies that the job has completions number of successful pods
func VerifyJobNCompletions(f *framework.Framework, completions int32) {
	ns := f.Namespace.Name
	pods, err := jobutil.GetJobPods(f.ClientSet, f.Namespace.Name, "cuda-add")
	framework.ExpectNoError(err)
	createdPods := pods.Items
	createdPodNames := podNames(createdPods)
	e2elog.Logf("Got the following pods for job cuda-add: %v", createdPodNames)

	successes := int32(0)
	for _, podName := range createdPodNames {
		f.PodClient().WaitForFinish(podName, 5*time.Minute)
		logs, err := e2epod.GetPodLogs(f.ClientSet, ns, podName, "vector-addition")
		framework.ExpectNoError(err, "Should be able to get logs for pod %v", podName)
		regex := regexp.MustCompile("PASSED")
		if regex.MatchString(logs) {
			successes++
		}
	}
	if successes != completions {
		e2elog.Failf("Only got %v completions. Expected %v completions.", successes, completions)
	}
}

func podNames(pods []v1.Pod) []string {
	originalPodNames := make([]string, len(pods))
	for i, p := range pods {
		originalPodNames[i] = p.ObjectMeta.Name
	}
	return originalPodNames
}

var _ = SIGDescribe("GPUDevicePluginAcrossRecreate [Feature:Recreate]", func() {
	ginkgo.BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke")
	})
	f := framework.NewDefaultFramework("device-plugin-gpus-recreate")
	ginkgo.It("run Nvidia GPU Device Plugin tests with a recreation", func() {
		testNvidiaGPUsJob(f)
	})
})
