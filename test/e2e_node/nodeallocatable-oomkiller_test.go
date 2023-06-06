/*
Copyright 2022 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("ndixita Node Allocatable OOMKiller [LinuxOnly] [NodeConformance]", func() {
	f := framework.NewDefaultFramework("nodeallocatable-oomkiller-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	containerName := "oomkill-target-container-cgroup"
	runOOMKillerTest(f, testCase{podSpec: nil, oomTargetContainerName: containerName})
})

func getOOMNodeAllocatableContainer(name string, nodeAllocatable int64) *v1.Container {
	return &v1.Container{
		Name:  name,
		Image: busyboxImage,
		Command: []string{
			"sh",
			"-c",
			"sleep 5 && dd if=/dev/zero of=/dev/null iflag=fullblock count=10 bs=10G",
		},
		// Resources: v1.ResourceRequirements{
		// 	Requests: v1.ResourceList{
		// 		v1.ResourceMemory: resource.MustParse("15Mi"),
		// 	},
		// 	Limits: v1.ResourceList{
		// 		v1.ResourceMemory: resource.MustParse("15Mi"),
		// 	},
		// },
	}
}
func getOOMNodeAllocatablePodSpec(f *framework.Framework, podName string, containerName string) *v1.Pod {
	ctx := context.Background()
	nodes, _ := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
	// framework.ExpectNoError(err, "failed to get running nodes")
	n := nodes.Items[0]
	nodeAllocatable := n.Status.Allocatable.Memory().Value()
	fmt.Printf("Node allocatable is: %d\n", nodeAllocatable)
	fmt.Println(n.Name)
	fmt.Printf("Node capacity is: %d\n", n.Status.Capacity.Memory().Value())
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				*getOOMNodeAllocatableContainer(containerName, nodeAllocatable),
			},
		},
	}

	e2epod.SetNodeAffinity(&testPod.Spec, n.Name)
	return testPod
}

func fileValue(filePath string) error {
	out, err := os.ReadFile(filePath)
	fmt.Printf("reading from: %s\n", filePath)
	fmt.Println(out)
	fmt.Println(err)
	if err != nil {
		return fmt.Errorf("failed to read file %q", filePath)
	}

	actual, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return fmt.Errorf("failed to parse output %v", err)
	}

	fmt.Printf("Cgroup settings: %v\n", actual)
	return nil
}

func runOOMKillerTest(f *framework.Framework, testCase testCase) {
	ginkgo.Context("", func() {
		ginkgo.BeforeEach(func() {
			var oldCfg *kubeletconfig.KubeletConfiguration
			var err error
			oldCfg, err = getCurrentKubeletConfig(context.TODO())
			framework.ExpectNoError(err)

			newCfg := oldCfg.DeepCopy()
			fmt.Println("Printing Kubelet Config")
			fmt.Printf("KubeReserved: %v\n", newCfg.KubeReserved)
			if newCfg.KubeReserved == nil {
				fmt.Printf("Is kube reserved nil: %v", newCfg.KubeReserved == nil)
				newCfg.KubeReserved = map[string]string{}
			}

			if newCfg.SystemReserved == nil {
				fmt.Printf("Is system reserved nil: %v", newCfg.SystemReserved == nil)
				newCfg.SystemReserved = map[string]string{}
			}

			fmt.Printf("oldCfg: %v\n", oldCfg)
			fmt.Printf("newCfg: %v\n", newCfg)
			fmt.Printf("KubeReserved: %v\n", newCfg.KubeReserved)
			newCfg.KubeReserved["memory"] = ""
			fmt.Printf("KubeReserved: %v\n", newCfg.KubeReserved)
			newCfg.KubeReserved["memory"] = "2000Mi"
			newCfg.SystemReserved["memory"] = "100Mi"
			updateKubeletConfig(context.TODO(), f, newCfg, true)
			updatedCfg, _ := getCurrentKubeletConfig(context.TODO())
			// fmt.Println("Holla")
			// time.Sleep(time.Hour)
			fmt.Printf("Updated config: %v\n", updatedCfg)
			fmt.Println(updatedCfg.KubeReserved)
			fmt.Println(updatedCfg.SystemReserved)

			subsystems, err := cm.GetCgroupSubsystems()
			if err != nil {
				fmt.Print("oh no")
			}

			cgroupName := "kubepods"
			if newCfg.CgroupDriver == "systemd" {
				cgroupName = "kubepods.slice"
			}

			fmt.Print("Node level cgroup \n")
			fileValue(filepath.Join(subsystems.MountPoints["memory"], cgroupName, "memory.limit_in_bytes"))
			fileValue(filepath.Join(subsystems.MountPoints["memory"], cgroupName, "memory.max"))

			containerName := "oomkill-target-container-cgroup"
			testCase.podSpec = getOOMNodeAllocatablePodSpec(f, "oomkill-target-pod-cgroup", containerName)
			ginkgo.By("setting up the pod to be used in the test")
			pod1 := e2epod.NewPodClient(f).Create(context.TODO(), testCase.podSpec)

			fmt.Print("Pod level cgroup \n")

			cgroupName = toCgroupFsName([]string{"kubepods", "besteffort-pod", string(pod1.UID)})
			fileValue(filepath.Join(subsystems.MountPoints["memory"], cgroupName, "memory.limit_in_bytes"))
			fileValue(filepath.Join(subsystems.MountPoints["memory"], cgroupName, "memory.max"))
			fmt.Println("printing overall")
			cgroupName = toCgroupFsName([]string{"kubepods", "besteffort"})
			fileValue(filepath.Join(subsystems.MountPoints["memory"], cgroupName, "memory.max"))

		})

		ginkgo.It("The containers terminated by OOM killer should have the reason set to OOMKilled", func() {

			ginkgo.By("Waiting for the pod to be failed")
			e2epod.WaitForPodTerminatedInNamespace(context.TODO(), f.ClientSet, testCase.podSpec.Name, "", f.Namespace.Name)

			ginkgo.By("Fetching the latest pod status")
			pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), testCase.podSpec.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to get the recent pod object for name: %q", pod.Name)

			ginkgo.By("Verifying the OOM target container has the expected reason")
			verifyReasonForOOMKilledContainer(pod, testCase.oomTargetContainerName)
			// time.Sleep(20 * time.Minute)
		})

		ginkgo.AfterEach(func() {
			ginkgo.By(fmt.Sprintf("deleting pod: %s", testCase.podSpec.Name))
			e2epod.NewPodClient(f).DeleteSync(context.TODO(), testCase.podSpec.Name, metav1.DeleteOptions{}, framework.PodDeleteTimeout)
		})
	})
}

// func runOOMKillerTest(f *framework.Framework, testCase testCase) {
// 	cgroupName := "kubepods"
// 	var newCfg *kubeletconfig.KubeletConfiguration
// 	ginkgo.Context("", func() {
// 		ginkgo.BeforeEach(func() {
// 			var oldCfg *kubeletconfig.KubeletConfiguration
// 			var err error
// 			oldCfg, err = getCurrentKubeletConfig(context.TODO())
// 			framework.ExpectNoError(err)

// 			newCfg = oldCfg.DeepCopy()
// 			fmt.Println("Printing Kubelet Config")
// 			fmt.Printf("KubeReserved: %v\n", newCfg.KubeReserved)
// 			if newCfg.KubeReserved == nil {
// 				fmt.Printf("Is kube reserved nil: %v", newCfg.KubeReserved == nil)
// 				newCfg.KubeReserved = map[string]string{}
// 			}

// 			if newCfg.SystemReserved == nil {
// 				fmt.Printf("Is system reserved nil: %v", newCfg.SystemReserved == nil)
// 				newCfg.SystemReserved = map[string]string{}
// 			}

// 			fmt.Printf("oldCfg: %v\n", oldCfg)
// 			fmt.Printf("newCfg: %v\n", newCfg)
// 			fmt.Printf("KubeReserved: %v\n", newCfg.KubeReserved)
// 			fmt.Printf("SystemReserved: %v\n", newCfg.SystemReserved)

// 			newCfg.KubeReserved["memory"] = "500Mi"
// 			newCfg.SystemReserved["memory"] = "100Mi"

// 			updateKubeletConfig(context.TODO(), f, newCfg, true)
// 			updatedCfg, _ := getCurrentKubeletConfig(context.TODO())
// 			fmt.Printf("Updated config: %v\n", updatedCfg)
// 			fmt.Println(updatedCfg.KubeReserved)
// 			fmt.Println(updatedCfg.SystemReserved)

// 		})

// 		ginkgo.It("The containers terminated by OOM killer should have the reason set to OOMKilled", func() {
// 			subsystems, err := cm.GetCgroupSubsystems()
// 			if err != nil {
// 				fmt.Print("unable to get cgroup subsytems")
// 			}

// 			if newCfg.CgroupDriver == "systemd" {
// 				cgroupName = "kubepods.slice"
// 			}

// 			fmt.Print("Node level cgroup \n")
// 			fileValue(filepath.Join(subsystems.MountPoints["memory"], cgroupName, "memory.limit_in_bytes"))
// 			fileValue(filepath.Join(subsystems.MountPoints["memory"], cgroupName, "memory.max"))

// 			testCase.podSpec = getOOMNodeAllocatablePodSpec(f, "oomkill-target-pod", testCase.oomTargetContainerName)
// 			ginkgo.By("setting up the pod to be used in the test")
// 			pod := e2epod.NewPodClient(f).Create(context.TODO(), testCase.podSpec)

// 			fmt.Print("Pod level cgroup \n")

// 			cgroupName = toCgroupFsName([]string{"kubepods", "besteffort-pod", string(pod.UID)})
// 			fileValue(filepath.Join(subsystems.MountPoints["memory"], cgroupName, "memory.limit_in_bytes"))
// 			fileValue(filepath.Join(subsystems.MountPoints["memory"], cgroupName, "memory.max"))
// 			fmt.Println("printing overall")
// 			cgroupName = toCgroupFsName([]string{"kubepods", "besteffort"})
// 			fileValue(filepath.Join(subsystems.MountPoints["memory"], cgroupName, "memory.max"))

// 			ginkgo.By("Verifying the OOM target container has the expected reason")
// 			fmt.Println(pod.Spec.Containers)
// 			fmt.Println("lalalalla")
// 			fmt.Println(testCase)
// 			verifyReasonForOOMKilledContainer(pod, testCase.oomTargetContainerName)
// 		})

// 		ginkgo.AfterEach(func() {
// 			ginkgo.By(fmt.Sprintf("deleting pod: %s", testCase.podSpec.Name))
// 			e2epod.NewPodClient(f).DeleteSync(context.TODO(), testCase.podSpec.Name, metav1.DeleteOptions{}, framework.PodDeleteTimeout)
// 		})
// 	})
// }

// func verifyReasonForOOMKilledContainer(pod *v1.Pod, oomTargetContainerName string) {
// 	container := e2epod.FindContainerStatusInPod(pod, oomTargetContainerName)
// 	if container == nil {
// 		framework.Failf("OOM target pod %q, container %q does not have the expected state terminated", pod.Name, container.Name)
// 	}
// 	if container.State.Terminated == nil {
// 		framework.Failf("OOM target pod %q, container %q is not in the terminated state", pod.Name, container.Name)
// 	}
// 	framework.ExpectEqual(container.State.Terminated.ExitCode, int32(137),
// 		fmt.Sprintf("pod: %q, container: %q has unexpected exitCode: %q", pod.Name, container.Name, container.State.Terminated.ExitCode))
// 	framework.ExpectEqual(container.State.Terminated.Reason, "OOMKilled",
// 		fmt.Sprintf("pod: %q, container: %q has unexpected reason: %q", pod.Name, container.Name, container.State.Terminated.Reason))
// }
