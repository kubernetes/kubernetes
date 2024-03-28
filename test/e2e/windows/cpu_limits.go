/*
Copyright 2020 The Kubernetes Authors.

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

package windows

import (
	"context"
	"fmt"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"time"

	"github.com/onsi/ginkgo/v2"
)

const (
	pollingDuration   = time.Second * 10
	timeout           = time.Minute * 2
	cpuUsageThreshold = .5 * 1.05
)

var _ = sigDescribe(feature.Windows, "Cpu Resources", framework.WithSerial(), skipUnlessWindows(func() {
	f := framework.NewDefaultFramework("cpu-resources-test-windows")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// The Windows 'BusyBox' image is PowerShell plus a collection of scripts and utilities to mimic common busybox commands
	powershellImage := imageutils.GetConfig(imageutils.BusyBox)

	ginkgo.Context("Container limits", func() {
		ginkgo.It("should not be exceeded the limit threshold", func(ctx context.Context) {
			ginkgo.By("Creating one pod with limit set to '0.5'")
			pods := newCPUBurnPods(1, powershellImage, "0.5", "1Gi")

			ginkgo.By("Creating one pod with limit set to '500m'")
			pods = append(pods, newCPUBurnPods(1, powershellImage, "500m", "1Gi")...)
			e2epod.NewPodClient(f).CreateBatch(ctx, pods)

			ginkgo.By("Ensuring pods are running")
			err := e2epod.WaitForPodsRunning(f.ClientSet, f.Namespace.Name, 2, framework.PodStartTimeout)
			framework.ExpectNoError(err, "Error waiting for pods entering on running state.")

			// eventually expect and reinsure both pods have the CPU usage
			// greater than zero and lower than test threshold, checked within 3 minutes.
			gomega.Eventually(ctx, checkCPUUsageThreshold, timeout, pollingDuration).WithArguments(ctx, f, pods).ShouldNot(gomega.HaveOccurred())
		})
	})
}))

func checkCPUUsageThreshold(ctx context.Context, f *framework.Framework, pods []*v1.Pod) error {
	var (
		err      error
		cpuUsage = float64(0)
	)

	for _, pod := range pods {
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.GetName(), metav1.GetOptions{})
		framework.ExpectNoError(err, "Error retrieving pod.")

		ginkgo.By("Gathering node summary stats")
		nodeStats, err := e2ekubelet.GetStatsSummary(ctx, f.ClientSet, pod.Spec.NodeName)
		framework.ExpectNoError(err, "Error retrieving node stats summary.")
		for _, nodePod := range nodeStats.Pods {
			if nodePod.PodRef.Name != pod.Name || nodePod.PodRef.Namespace != pod.Namespace {
				continue
			}
			cpuUsage = float64(*nodePod.CPU.UsageNanoCores) * 1e-9
		}

		framework.Logf("Pod %s usage: %v", pod.Name, cpuUsage)
		if cpuUsage <= 0 {
			return fmt.Errorf("pod %s/%s reported usage is %v, but it should be greater than 0", pod.Namespace, pod.Name, cpuUsage)
		}
		if cpuUsage >= cpuUsageThreshold {
			return fmt.Errorf("pod %s/%s reported usage is %v, but it should not exceed limit by > 5%%", pod.Namespace, pod.Name, cpuUsage)
		}
	}
	return nil
}

// newCPUBurnPods creates a list of pods (specification) with a workload that will consume all available CPU resources up to container limit
func newCPUBurnPods(numPods int, image imageutils.Config, cpuLimit string, memoryLimit string) []*v1.Pod {
	var pods []*v1.Pod

	memLimitQuantity, err := resource.ParseQuantity(memoryLimit)
	framework.ExpectNoError(err)

	cpuLimitQuantity, err := resource.ParseQuantity(cpuLimit)
	framework.ExpectNoError(err)

	for i := 0; i < numPods; i++ {
		podName := "cpulimittest-" + string(uuid.NewUUID())
		pod := v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
				Labels: map[string]string{
					"name":    podName,
					"testapp": "cpuburn",
				},
			},
			Spec: v1.PodSpec{
				// Restart policy is always (default).
				Containers: []v1.Container{
					{
						Image: image.GetE2EImage(),
						Name:  podName,
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceMemory: memLimitQuantity,
								v1.ResourceCPU:    cpuLimitQuantity,
							},
						},
						Command: []string{
							"powershell.exe",
							"-Command",
							"foreach ($loopnumber in 1..8) { Start-Job -ScriptBlock { $result = 1; foreach($mm in 1..2147483647){$res1=1;foreach($num in 1..2147483647){$res1=$mm*$num*1340371};$res1} } } ; get-job | wait-job",
						},
					},
				},
				NodeSelector: map[string]string{
					"kubernetes.io/os": "windows",
				},
			},
		}

		pods = append(pods, &pod)
	}

	return pods
}
