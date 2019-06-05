// +build linux

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

package e2e_node

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/kubelet/cm"

	units "github.com/docker/go-units"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"

	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

// makePodToVerifyHugePagesCgroup returns a pod that verifies specified cgroup with hugetlb
func makePodToVerifyHugePagesCgroup(baseName string, hugePagesLimit resource.Quantity, hugePageSize resource.Quantity) *v1.Pod {
	// convert the cgroup name to its literal form
	cgroupFsName := ""
	cgroupName := cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup, baseName)
	if framework.TestContext.KubeletConfig.CgroupDriver == "systemd" {
		cgroupFsName = cgroupName.ToSystemd()
	} else {
		cgroupFsName = cgroupName.ToCgroupfs()
	}
	hugePageSizeString := units.CustomSize("%g%s", float64(hugePageSize.Value()), 1024.0, libcontainercgroups.HugePageSizeUnitList)

	// this command takes the expected value and compares it against the actual value for the pod cgroup hugetlb.<hugepagesize>.limit_in_bytes
	command := fmt.Sprintf("expected=%v; actual=$(cat /tmp/hugetlb/%v/hugetlb.%s.limit_in_bytes); if [ \"$expected\" -ne \"$actual\" ]; then exit 1; fi; ", hugePagesLimit.Value(), cgroupFsName, hugePageSizeString)
	e2elog.Logf("Pod to run command: %v", command)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Image:   busyboxImage,
					Name:    "container" + string(uuid.NewUUID()),
					Command: []string{"sh", "-c", command},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "sysfscgroup",
							MountPath: "/tmp",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "sysfscgroup",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{Path: "/sys/fs/cgroup"},
					},
				},
			},
		},
	}
	return pod
}

// makeHugePagePod returns a pod that requests the the given amount of huge page memory, and execute the given command
func makeHugePagePod(baseName string, command string, totalHugePageMemory resource.Quantity, hugePageSize resource.Quantity) *v1.Pod {
	e2elog.Logf("Pod to run command: %v", command)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Image:   imageutils.GetE2EImage(imageutils.HugePageTester),
					Name:    "container" + string(uuid.NewUUID()),
					Command: []string{"sh", "-c", command},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName("cpu"):                                resource.MustParse("1"),
							v1.ResourceName("memory"):                             resource.MustParse("100Mi"),
							v1.ResourceName("hugepages-" + hugePageSize.String()): totalHugePageMemory,
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "hugetlb",
							MountPath: "/hugetlb",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "hugetlb",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{Medium: "HugePages"},
					},
				},
			},
		},
	}
	return pod
}

// enableHugePagesInKubelet enables hugepages feature for kubelet
func enableHugePagesInKubelet(f *framework.Framework) *kubeletconfig.KubeletConfiguration {
	oldCfg, err := getCurrentKubeletConfig()
	framework.ExpectNoError(err)
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = make(map[string]bool)
		newCfg.FeatureGates["HugePages"] = true
	}

	// Update the Kubelet configuration.
	framework.ExpectNoError(setKubeletConfiguration(f, newCfg))

	// Wait for the Kubelet to be ready.
	gomega.Eventually(func() bool {
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		return len(nodeList.Items) == 1
	}, time.Minute, time.Second).Should(gomega.BeTrue())

	return oldCfg
}

// configureHugePages attempts to allocate _pageCount_ hugepages of the default hugepage size for testing purposes
func configureHugePages(pageCount int64) error {
	err := exec.Command("/bin/sh", "-c", fmt.Sprintf("echo %d > /proc/sys/vm/nr_hugepages", pageCount)).Run()
	if err != nil {
		return err
	}
	outData, err := exec.Command("/bin/sh", "-c", "cat /proc/meminfo | grep 'HugePages_Total' | awk '{print $2}'").Output()
	if err != nil {
		return err
	}
	numHugePages, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	if err != nil {
		return err
	}
	e2elog.Logf("HugePages_Total is set to %v", numHugePages)
	if int64(numHugePages) == pageCount {
		return nil
	}
	return fmt.Errorf("expected hugepages %v, but found %v", pageCount, numHugePages)
}

// releaseHugePages releases all pre-allocated hugepages
func releaseHugePages() error {
	return exec.Command("/bin/sh", "-c", "echo 0 > /proc/sys/vm/nr_hugepages").Run()
}

// getDefaultHugePageSize returns the default huge page size, and a boolean if huge pages are supported
func getDefaultHugePageSize() (resource.Quantity, bool) {
	outData, err := exec.Command("/bin/sh", "-c", "cat /proc/meminfo | grep 'Hugepagesize:' | awk '{print $2}'").Output()
	framework.ExpectNoError(err)
	pageSize, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	framework.ExpectNoError(err)
	if pageSize == 0 {
		return resource.Quantity{}, false
	}
	return *resource.NewQuantity(int64(pageSize*1024), resource.BinarySI), true
}

func getTestValues() (hugePageSize resource.Quantity, totalMemory resource.Quantity, pageCount int64) {
	hugePageSize, _ = getDefaultHugePageSize()
	// If huge page size is  equal to bigger than 1GB, only use two pages
	if hugePageSize.Value() >= (1 << 30) {
		pageCount = 2
	} else {
		pageCount = 5
	}
	totalMemory = *resource.NewQuantity(hugePageSize.Value()*pageCount, resource.BinarySI)
	return
}

// pollResourceAsString polls for a specified resource and capacity from node
func pollResourceAsString(f *framework.Framework, resourceName string) string {
	node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	amount := amountOfResourceAsString(node, resourceName)
	e2elog.Logf("amount of %v: %v", resourceName, amount)
	return amount
}

// amountOfResourceAsString returns the amount of resourceName advertised by a node
func amountOfResourceAsString(node *v1.Node, resourceName string) string {
	val, ok := node.Status.Capacity[v1.ResourceName(resourceName)]
	if !ok {
		return ""
	}
	return val.String()
}

func runHugePagesTests(f *framework.Framework) {
	fileName := "/hugetlb/file"
	ginkgo.It("should assign hugepages as expected based on the Pod spec", func() {
		hugePageSize, totalHugePageMemory, _ := getTestValues()
		ginkgo.By("running a pod that requests hugepages and allocates the memory")
		command := fmt.Sprintf(`./hugetlb-tester %d %d %s`, totalHugePageMemory.Value(), hugePageSize.Value(), fileName)

		verifyPod := makeHugePagePod("hugepage-pod", command, totalHugePageMemory, hugePageSize)
		f.PodClient().Create(verifyPod)
		err := e2epod.WaitForPodSuccessInNamespace(f.ClientSet, verifyPod.Name, f.Namespace.Name)
		ginkgo.By("checking that pod execution succeeded")
		framework.ExpectNoError(err)

	})
	ginkgo.It("should not be possible to allocate more hugepage memory than the Pod spec", func() {
		hugePageSize, totalHugePageMemory, _ := getTestValues()
		ginkgo.By("running a pod that requests hugepages and allocates twice the amount of the requested memory")
		command := fmt.Sprintf(`./hugetlb-tester %d %d %s`, totalHugePageMemory.Value()*2, hugePageSize.Value(), fileName)

		verifyPod := makeHugePagePod("hugepage-pod", command, totalHugePageMemory, hugePageSize)
		f.PodClient().Create(verifyPod)
		err := e2epod.WaitForPodSuccessInNamespace(f.ClientSet, verifyPod.Name, f.Namespace.Name)
		ginkgo.By("checking that pod execution failed")
		framework.ExpectError(err)
	})
	ginkgo.It("should not be possible to allocate hugepage memory with a huge page size not requested in the Pod spec", func() {
		hugePageSize, totalHugePageMemory, _ := getTestValues()
		ginkgo.By("running a pod that requests hugepages and allocates using a page size euqal to twice the requested size")
		command := fmt.Sprintf(`./hugetlb-tester %d %d %s`, totalHugePageMemory.Value(), hugePageSize.Value()*2, fileName)

		verifyPod := makeHugePagePod("hugepage-pod", command, totalHugePageMemory, hugePageSize)
		f.PodClient().Create(verifyPod)
		err := e2epod.WaitForPodSuccessInNamespace(f.ClientSet, verifyPod.Name, f.Namespace.Name)
		ginkgo.By("checking that pod execution failed")
		framework.ExpectError(err)

	})
	ginkgo.It("should assign hugepages in cgroup as expected based on the Pod spec", func() {
		hugePageSize, totalHugePageMemory, _ := getTestValues()
		ginkgo.By("running a G pod that requests hugepages")
		pod := f.PodClient().Create(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Image: imageutils.GetPauseImageName(),
						Name:  "container" + string(uuid.NewUUID()),
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceName("cpu"):                                resource.MustParse("10m"),
								v1.ResourceName("memory"):                             resource.MustParse("100Mi"),
								v1.ResourceName("hugepages-" + hugePageSize.String()): totalHugePageMemory,
							},
						},
					},
				},
			},
		})
		podUID := string(pod.UID)
		ginkgo.By("checking if the expected hugetlb settings were applied")
		verifyPod := makePodToVerifyHugePagesCgroup("pod"+podUID, totalHugePageMemory, hugePageSize)
		f.PodClient().Create(verifyPod)
		err := e2epod.WaitForPodSuccessInNamespace(f.ClientSet, verifyPod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
	})

}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("HugePages [Serial] [Feature:HugePages][NodeFeature:HugePages]", func() {
	f := framework.NewDefaultFramework("hugepages-test")

	ginkgo.Context("With config updated with hugepages feature enabled", func() {
		ginkgo.BeforeEach(func() {
			ginkgo.By("verifying hugepages are supported")

			hugePageSize, supported := getDefaultHugePageSize()
			if !supported {
				framework.Skipf("skipping test because hugepages are not supported")
				return
			}
			_, testTotalHugePageMemory, testPageCount := getTestValues()

			// pre-allocate twice the amount of huge page memory as the test will use to ensure that limits are enforced
			reserveTotalHugePageMemory, reservePageCount := *resource.NewQuantity(testTotalHugePageMemory.Value()*2, resource.BinarySI), testPageCount*2
			ginkgo.By("configuring the host to reserve a number of pre-allocated hugepages")
			gomega.Eventually(func() error {
				err := configureHugePages(reservePageCount)
				if err != nil {
					return err
				}
				ginkgo.By("restarting kubelet to pick up pre-allocated hugepages")
				restartKubelet()
				return nil
			}, 30*time.Second, framework.Poll).Should(gomega.BeNil())
			ginkgo.By("by waiting for hugepages resource to become available on the local node")
			gomega.Eventually(func() string {
				return pollResourceAsString(f, "hugepages-"+hugePageSize.String())
			}, 30*time.Second, framework.Poll).Should(gomega.Equal(reserveTotalHugePageMemory.String()))
		})

		runHugePagesTests(f)

		ginkgo.AfterEach(func() {
			ginkgo.By("Releasing hugepages")
			gomega.Eventually(func() error {
				err := releaseHugePages()
				if err != nil {
					return err
				}
				return nil
			}, 30*time.Second, framework.Poll).Should(gomega.BeNil())
			ginkgo.By("by waiting for hugepages resource to not appear available on the local node")
			restartKubelet()
			hugePageSize, _ := getDefaultHugePageSize()
			gomega.Eventually(func() string {
				return pollResourceAsString(f, "hugepages-"+hugePageSize.String())
			}, 30*time.Second, framework.Poll).Should(gomega.Equal("0"))
		})
	})
})
