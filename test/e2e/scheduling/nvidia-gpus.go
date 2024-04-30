//go:build !providerless
// +build !providerless

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
	"context"
	"os"
	"regexp"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edebug "k8s.io/kubernetes/test/e2e/framework/debug"
	e2egpu "k8s.io/kubernetes/test/e2e/framework/gpu"
	e2ejob "k8s.io/kubernetes/test/e2e/framework/job"
	e2emanifest "k8s.io/kubernetes/test/e2e/framework/manifest"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	e2eresource "k8s.io/kubernetes/test/e2e/framework/resource"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	testPodNamePrefix = "nvidia-gpu-"
	// Nvidia driver installation can take upwards of 5 minutes.
	driverInstallTimeout = 10 * time.Minute
)

var (
	gpuResourceName v1.ResourceName
)

func makeCudaAdditionDevicePluginTestPod() *v1.Pod {
	podName := testPodNamePrefix + string(uuid.NewUUID())
	testContainers := []v1.Container{
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
	}
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	testPod.Spec.Containers = testContainers
	if os.Getenv("TEST_MAX_GPU_COUNT") == "1" {
		testPod.Spec.Containers = []v1.Container{testContainers[0]}
	}
	framework.Logf("testPod.Spec.Containers {%#v}", testPod.Spec.Containers)
	return testPod
}

func logOSImages(ctx context.Context, f *framework.Framework) {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	for _, node := range nodeList.Items {
		framework.Logf("Nodename: %v, OS Image: %v", node.Name, node.Status.NodeInfo.OSImage)
	}
}

func isControlPlaneNode(node v1.Node) bool {
	_, isControlPlane := node.Labels["node-role.kubernetes.io/control-plane"]
	if isControlPlane {
		framework.Logf("Node: %q is a control-plane node (label)", node.Name)
		return true
	}

	for _, taint := range node.Spec.Taints {
		if taint.Key == "node-role.kubernetes.io/control-plane" {
			framework.Logf("Node: %q is a control-plane node (taint)", node.Name)
			return true
		}
	}
	framework.Logf("Node: %q is NOT a control-plane node", node.Name)
	return false
}

func areGPUsAvailableOnAllSchedulableNodes(ctx context.Context, f *framework.Framework) bool {
	framework.Logf("Getting list of Nodes from API server")
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	for _, node := range nodeList.Items {
		if node.Spec.Unschedulable || isControlPlaneNode(node) {
			continue
		}

		if val, ok := node.Status.Capacity[gpuResourceName]; !ok || val.Value() == 0 {
			framework.Logf("Nvidia GPUs not available on Node: %q", node.Name)
			return false
		}
	}
	framework.Logf("Nvidia GPUs exist on all schedulable worker nodes")
	return true
}

func getGPUsAvailable(ctx context.Context, f *framework.Framework) int64 {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
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
func SetupNVIDIAGPUNode(ctx context.Context, f *framework.Framework, setupResourceGatherer bool) *e2edebug.ContainerResourceGatherer {
	logOSImages(ctx, f)

	var err error
	var ds *appsv1.DaemonSet
	dsYamlURLFromEnv := os.Getenv("NVIDIA_DRIVER_INSTALLER_DAEMONSET")
	if dsYamlURLFromEnv != "" {
		// Using DaemonSet from remote URL
		framework.Logf("Using remote nvidia-driver-installer daemonset manifest from %v", dsYamlURLFromEnv)
		ds, err = e2emanifest.DaemonSetFromURL(ctx, dsYamlURLFromEnv)
		framework.ExpectNoError(err, "failed get remote")
	} else {
		// Using default local DaemonSet
		framework.Logf("Using default local nvidia-driver-installer daemonset manifest.")
		data, err := e2etestfiles.Read("test/e2e/testing-manifests/scheduling/nvidia-driver-installer.yaml")
		framework.ExpectNoError(err, "failed to read local manifest for nvidia-driver-installer daemonset")
		ds, err = e2emanifest.DaemonSetFromData(data)
		framework.ExpectNoError(err, "failed to parse local manifest for nvidia-driver-installer daemonset")
	}
	gpuResourceName = e2egpu.NVIDIAGPUResourceName
	ds.Namespace = f.Namespace.Name

	_, err = f.ClientSet.AppsV1().DaemonSets(ds.Namespace).Create(ctx, ds, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create nvidia-driver-installer daemonset")
	framework.Logf("Successfully created daemonset to install Nvidia drivers.")

	pods, err := e2eresource.WaitForControlledPods(ctx, f.ClientSet, ds.Namespace, ds.Name, extensionsinternal.Kind("DaemonSet"))
	framework.ExpectNoError(err, "failed to get pods controlled by the nvidia-driver-installer daemonset")

	devicepluginPods, err := e2eresource.WaitForControlledPods(ctx, f.ClientSet, "kube-system", "nvidia-gpu-device-plugin", extensionsinternal.Kind("DaemonSet"))
	if err == nil {
		framework.Logf("Adding deviceplugin addon pod.")
		pods.Items = append(pods.Items, devicepluginPods.Items...)
	}

	var rsgather *e2edebug.ContainerResourceGatherer
	if setupResourceGatherer {
		framework.Logf("Starting ResourceUsageGather for the created DaemonSet pods.")
		rsgather, err = e2edebug.NewResourceUsageGatherer(ctx, f.ClientSet, e2edebug.ResourceGathererOptions{InKubemark: false, Nodes: e2edebug.AllNodes, ResourceDataGatheringPeriod: 2 * time.Second, ProbeDuration: 2 * time.Second, PrintVerboseLogs: true}, pods)
		framework.ExpectNoError(err, "creating ResourceUsageGather for the daemonset pods")
		go rsgather.StartGatheringData(ctx)
	}

	// Wait for Nvidia GPUs to be available on nodes
	framework.Logf("Waiting for drivers to be installed and GPUs to be available in Node Capacity...")
	gomega.Eventually(ctx, func(ctx context.Context) bool {
		return areGPUsAvailableOnAllSchedulableNodes(ctx, f)
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

func testNvidiaGPUs(ctx context.Context, f *framework.Framework) {
	rsgather := SetupNVIDIAGPUNode(ctx, f, true)
	gpuPodNum := getGPUsAvailable(ctx, f) / getGPUsPerPod()
	framework.Logf("Creating %d pods and have the pods run a CUDA app", gpuPodNum)
	podList := []*v1.Pod{}
	for i := int64(0); i < gpuPodNum; i++ {
		podList = append(podList, e2epod.NewPodClient(f).Create(ctx, makeCudaAdditionDevicePluginTestPod()))
	}
	framework.Logf("Wait for all test pods to succeed")
	// Wait for all pods to succeed
	for _, pod := range podList {
		e2epod.NewPodClient(f).WaitForSuccess(ctx, pod.Name, 5*time.Minute)
		logContainers(ctx, f, pod)
	}

	framework.Logf("Stopping ResourceUsageGather")
	constraints := make(map[string]e2edebug.ResourceConstraint)
	// For now, just gets summary. Can pass valid constraints in the future.
	summary, err := rsgather.StopAndSummarize([]int{50, 90, 100}, constraints)
	f.TestSummaries = append(f.TestSummaries, summary)
	framework.ExpectNoError(err, "getting resource usage summary")
}

func logContainers(ctx context.Context, f *framework.Framework, pod *v1.Pod) {
	for _, container := range pod.Spec.Containers {
		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, container.Name)
		framework.ExpectNoError(err, "Should be able to get container logs for container: %s", container.Name)
		framework.Logf("Got container logs for %s:\n%v", container.Name, logs)
	}
}

var _ = SIGDescribe(feature.GPUDevicePlugin, func() {
	f := framework.NewDefaultFramework("device-plugin-gpus")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.It("run Nvidia GPU Device Plugin tests", func(ctx context.Context) {
		testNvidiaGPUs(ctx, f)
	})
})

func testNvidiaGPUsJob(ctx context.Context, f *framework.Framework) {
	_ = SetupNVIDIAGPUNode(ctx, f, false)
	// Job set to have 5 completions with parallelism of 1 to ensure that it lasts long enough to experience the node recreation
	completions := int32(5)
	ginkgo.By("Starting GPU job")
	StartJob(ctx, f, completions)

	job, err := e2ejob.GetJob(ctx, f.ClientSet, f.Namespace.Name, "cuda-add")
	framework.ExpectNoError(err)

	// make sure job is running by waiting for its first pod to start running
	err = e2ejob.WaitForJobPodsRunning(ctx, f.ClientSet, f.Namespace.Name, job.Name, 1)
	framework.ExpectNoError(err)

	numNodes, err := e2enode.TotalRegistered(ctx, f.ClientSet)
	framework.ExpectNoError(err)
	nodes, err := e2enode.CheckReady(ctx, f.ClientSet, numNodes, framework.NodeReadyInitialTimeout)
	framework.ExpectNoError(err)

	ginkgo.By("Recreating nodes")
	err = gce.RecreateNodes(f.ClientSet, nodes)
	framework.ExpectNoError(err)
	ginkgo.By("Done recreating nodes")

	ginkgo.By("Waiting for gpu job to finish")
	err = e2ejob.WaitForJobFinish(ctx, f.ClientSet, f.Namespace.Name, job.Name)
	framework.ExpectNoError(err)
	ginkgo.By("Done with gpu job")

	gomega.Expect(job.Status.Failed).To(gomega.BeZero(), "Job pods failed during node recreation: %v", job.Status.Failed)

	VerifyJobNCompletions(ctx, f, completions)
}

// StartJob starts a simple CUDA job that requests gpu and the specified number of completions
func StartJob(ctx context.Context, f *framework.Framework, completions int32) {
	var activeSeconds int64 = 3600
	testJob := e2ejob.NewTestJob("succeed", "cuda-add", v1.RestartPolicyAlways, 1, completions, &activeSeconds, 6)
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
	_, err := e2ejob.CreateJob(ctx, f.ClientSet, ns, testJob)
	framework.ExpectNoError(err)
	framework.Logf("Created job %v", testJob)
}

// VerifyJobNCompletions verifies that the job has completions number of successful pods
func VerifyJobNCompletions(ctx context.Context, f *framework.Framework, completions int32) {
	ns := f.Namespace.Name
	pods, err := e2ejob.GetJobPods(ctx, f.ClientSet, f.Namespace.Name, "cuda-add")
	framework.ExpectNoError(err)
	createdPods := pods.Items
	createdPodNames := podNames(createdPods)
	framework.Logf("Got the following pods for job cuda-add: %v", createdPodNames)

	successes := int32(0)
	regex := regexp.MustCompile("PASSED")
	for _, podName := range createdPodNames {
		e2epod.NewPodClient(f).WaitForFinish(ctx, podName, 5*time.Minute)
		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, ns, podName, "vector-addition")
		framework.ExpectNoError(err, "Should be able to get logs for pod %v", podName)
		if regex.MatchString(logs) {
			successes++
		}
	}
	if successes != completions {
		framework.Failf("Only got %v completions. Expected %v completions.", successes, completions)
	}
}

func podNames(pods []v1.Pod) []string {
	originalPodNames := make([]string, len(pods))
	for i, p := range pods {
		originalPodNames[i] = p.ObjectMeta.Name
	}
	return originalPodNames
}

var _ = SIGDescribe("GPUDevicePluginAcrossRecreate", feature.Recreate, func() {
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
	})
	f := framework.NewDefaultFramework("device-plugin-gpus-recreate")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.It("run Nvidia GPU Device Plugin tests with a recreation", func(ctx context.Context) {
		testNvidiaGPUsJob(ctx, f)
	})
})
