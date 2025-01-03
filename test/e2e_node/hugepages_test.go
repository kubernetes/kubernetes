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

package e2enode

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	hugepagesSize2M          = 2048
	hugepagesSize1G          = 1048576
	hugepagesDirPrefix       = "/sys/kernel/mm/hugepages/hugepages"
	hugepagesCapacityFile    = "nr_hugepages"
	hugepagesResourceName2Mi = "hugepages-2Mi"
	hugepagesResourceName1Gi = "hugepages-1Gi"
	hugepagesCgroup2MB       = "hugetlb.2MB"
	hugepagesCgroup1GB       = "hugetlb.1GB"
	mediumHugepages          = "HugePages"
	mediumHugepages2Mi       = "HugePages-2Mi"
	mediumHugepages1Gi       = "HugePages-1Gi"
)

var (
	resourceToSize = map[string]int{
		hugepagesResourceName2Mi: hugepagesSize2M,
		hugepagesResourceName1Gi: hugepagesSize1G,
	}
	resourceToCgroup = map[string]string{
		hugepagesResourceName2Mi: hugepagesCgroup2MB,
		hugepagesResourceName1Gi: hugepagesCgroup1GB,
	}
)

// makePodToVerifyHugePages returns a pod that verifies specified cgroup with hugetlb
func makePodToVerifyHugePages(baseName string, hugePagesLimit resource.Quantity, hugepagesCgroup string) *v1.Pod {
	// convert the cgroup name to its literal form
	cgroupName := cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup, baseName)
	cgroupFsName := ""
	if kubeletCfg.CgroupDriver == "systemd" {
		cgroupFsName = cgroupName.ToSystemd()
	} else {
		cgroupFsName = cgroupName.ToCgroupfs()
	}

	hugetlbLimitFile := ""
	// this command takes the expected value and compares it against the actual value for the pod cgroup hugetlb.2MB.<LIMIT>
	if IsCgroup2UnifiedMode() {
		hugetlbLimitFile = fmt.Sprintf("/tmp/%s/%s.max", cgroupFsName, hugepagesCgroup)
	} else {
		hugetlbLimitFile = fmt.Sprintf("/tmp/hugetlb/%s/%s.limit_in_bytes", cgroupFsName, hugepagesCgroup)
	}

	command := fmt.Sprintf("expected=%v; actual=$(cat %v); if [ \"$expected\" -ne \"$actual\" ]; then exit 1; fi; ", hugePagesLimit.Value(), hugetlbLimitFile)
	framework.Logf("Pod to run command: %v", command)
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

// configureHugePages attempts to allocate hugepages of the specified size
func configureHugePages(hugepagesSize int, hugepagesCount int, numaNodeID *int) error {
	// Compact memory to make bigger contiguous blocks of memory available
	// before allocating huge pages.
	// https://www.kernel.org/doc/Documentation/sysctl/vm.txt
	if _, err := os.Stat("/proc/sys/vm/compact_memory"); err == nil {
		if err := exec.Command("/bin/sh", "-c", "echo 1 > /proc/sys/vm/compact_memory").Run(); err != nil {
			return err
		}
	}

	// e.g. hugepages/hugepages-2048kB/nr_hugepages
	hugepagesSuffix := fmt.Sprintf("hugepages/hugepages-%dkB/%s", hugepagesSize, hugepagesCapacityFile)

	// e.g. /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
	hugepagesFile := fmt.Sprintf("/sys/kernel/mm/%s", hugepagesSuffix)
	if numaNodeID != nil {
		// e.g. /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
		hugepagesFile = fmt.Sprintf("/sys/devices/system/node/node%d/%s", *numaNodeID, hugepagesSuffix)
	}

	// Reserve number of hugepages
	// e.g. /bin/sh -c "echo 5 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages"
	command := fmt.Sprintf("echo %d > %s", hugepagesCount, hugepagesFile)
	if err := exec.Command("/bin/sh", "-c", command).Run(); err != nil {
		return err
	}

	// verify that the number of hugepages was updated
	// e.g. /bin/sh -c "cat /sys/kernel/mm/hugepages/hugepages-2048kB/vm.nr_hugepages"
	command = fmt.Sprintf("cat %s", hugepagesFile)
	outData, err := exec.Command("/bin/sh", "-c", command).Output()
	if err != nil {
		return err
	}

	numHugePages, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	if err != nil {
		return err
	}

	framework.Logf("Hugepages total is set to %v", numHugePages)
	if numHugePages == hugepagesCount {
		return nil
	}

	return fmt.Errorf("expected hugepages %v, but found %v", hugepagesCount, numHugePages)
}

// isHugePageAvailable returns true if hugepages of the specified size is available on the host
func isHugePageAvailable(hugepagesSize int) bool {
	path := fmt.Sprintf("%s-%dkB/%s", hugepagesDirPrefix, hugepagesSize, hugepagesCapacityFile)
	if _, err := os.Stat(path); err != nil {
		return false
	}
	return true
}

func getHugepagesTestPod(f *framework.Framework, limits v1.ResourceList, mounts []v1.VolumeMount, volumes []v1.Volume) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "hugepages-",
			Namespace:    f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "container" + string(uuid.NewUUID()),
					Image: busyboxImage,
					Resources: v1.ResourceRequirements{
						Limits: limits,
					},
					Command:      []string{"sleep", "3600"},
					VolumeMounts: mounts,
				},
			},
			Volumes: volumes,
		},
	}
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("HugePages", framework.WithSerial(), feature.HugePages, func() {
	f := framework.NewDefaultFramework("hugepages-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should remove resources for huge page sizes no longer supported", func(ctx context.Context) {
		ginkgo.By("mimicking support for 9Mi of 3Mi huge page memory by patching the node status")
		patch := []byte(`[{"op": "add", "path": "/status/capacity/hugepages-3Mi", "value": "9Mi"}, {"op": "add", "path": "/status/allocatable/hugepages-3Mi", "value": "9Mi"}]`)
		result := f.ClientSet.CoreV1().RESTClient().Patch(types.JSONPatchType).Resource("nodes").Name(framework.TestContext.NodeName).SubResource("status").Body(patch).Do(ctx)
		framework.ExpectNoError(result.Error(), "while patching")

		node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, framework.TestContext.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err, "while getting node status")

		ginkgo.By("Verifying that the node now supports huge pages with size 3Mi")
		value, ok := node.Status.Capacity["hugepages-3Mi"]
		if !ok {
			framework.Failf("capacity should contain resource hugepages-3Mi: %v", node.Status.Capacity)
		}
		gomega.Expect(value.String()).To(gomega.Equal("9Mi"), "huge pages with size 3Mi should be supported")

		ginkgo.By("restarting the node and verifying that huge pages with size 3Mi are not supported")
		restartKubelet(ctx, true)

		ginkgo.By("verifying that the hugepages-3Mi resource no longer is present")
		gomega.Eventually(ctx, func() bool {
			node, err = f.ClientSet.CoreV1().Nodes().Get(ctx, framework.TestContext.NodeName, metav1.GetOptions{})
			framework.ExpectNoError(err, "while getting node status")
			_, isPresent := node.Status.Capacity["hugepages-3Mi"]
			return isPresent
		}, 30*time.Second, framework.Poll).Should(gomega.BeFalseBecause("hugepages resource should not be present"))
	})

	ginkgo.It("should add resources for new huge page sizes on kubelet restart", func(ctx context.Context) {
		ginkgo.By("Stopping kubelet")
		restartKubelet := mustStopKubelet(ctx, f)
		ginkgo.By(`Patching away support for hugepage resource "hugepages-2Mi"`)
		patch := []byte(`[{"op": "remove", "path": "/status/capacity/hugepages-2Mi"}, {"op": "remove", "path": "/status/allocatable/hugepages-2Mi"}]`)
		result := f.ClientSet.CoreV1().RESTClient().Patch(types.JSONPatchType).Resource("nodes").Name(framework.TestContext.NodeName).SubResource("status").Body(patch).Do(ctx)
		framework.ExpectNoError(result.Error(), "while patching")

		ginkgo.By("Restarting kubelet again")
		restartKubelet(ctx)

		ginkgo.By("verifying that the hugepages-2Mi resource is present")
		gomega.Eventually(ctx, func() bool {
			node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, framework.TestContext.NodeName, metav1.GetOptions{})
			framework.ExpectNoError(err, "while getting node status")
			_, isPresent := node.Status.Capacity["hugepages-2Mi"]
			return isPresent
		}, 30*time.Second, framework.Poll).Should(gomega.BeTrueBecause("hugepages resource should be present"))
	})

	ginkgo.When("start the pod", func() {
		var (
			testpod   *v1.Pod
			limits    v1.ResourceList
			mounts    []v1.VolumeMount
			volumes   []v1.Volume
			hugepages map[string]int
		)

		setHugepages := func(ctx context.Context) {
			for hugepagesResource, count := range hugepages {
				size := resourceToSize[hugepagesResource]
				ginkgo.By(fmt.Sprintf("Verifying hugepages %d are supported", size))
				if !isHugePageAvailable(size) {
					e2eskipper.Skipf("skipping test because hugepages of size %d not supported", size)
					return
				}

				ginkgo.By(fmt.Sprintf("Configuring the host to reserve %d of pre-allocated hugepages of size %d", count, size))
				gomega.Eventually(ctx, func() error {
					if err := configureHugePages(size, count, nil); err != nil {
						return err
					}
					return nil
				}, 30*time.Second, framework.Poll).Should(gomega.BeNil())
			}
		}

		waitForHugepages := func(ctx context.Context) {
			ginkgo.By("Waiting for hugepages resource to become available on the local node")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, framework.TestContext.NodeName, metav1.GetOptions{})
				if err != nil {
					return err
				}

				for hugepagesResource, count := range hugepages {
					capacity, ok := node.Status.Capacity[v1.ResourceName(hugepagesResource)]
					if !ok {
						return fmt.Errorf("the node does not have the resource %s", hugepagesResource)
					}

					size, succeed := capacity.AsInt64()
					if !succeed {
						return fmt.Errorf("failed to convert quantity to int64")
					}

					expectedSize := count * resourceToSize[hugepagesResource] * 1024
					if size != int64(expectedSize) {
						return fmt.Errorf("the actual size %d is different from the expected one %d", size, expectedSize)
					}
				}
				return nil
			}, time.Minute, framework.Poll).Should(gomega.BeNil())
		}

		releaseHugepages := func(ctx context.Context) {
			ginkgo.By("Releasing hugepages")
			gomega.Eventually(ctx, func() error {
				for hugepagesResource := range hugepages {
					command := fmt.Sprintf("echo 0 > %s-%dkB/%s", hugepagesDirPrefix, resourceToSize[hugepagesResource], hugepagesCapacityFile)
					if err := exec.Command("/bin/sh", "-c", command).Run(); err != nil {
						return err
					}
				}
				return nil
			}, 30*time.Second, framework.Poll).Should(gomega.BeNil())
		}

		runHugePagesTests := func() {
			ginkgo.It("should set correct hugetlb mount and limit under the container cgroup", func(ctx context.Context) {
				ginkgo.By("getting mounts for the test pod")
				command := []string{"mount"}
				out := e2epod.ExecCommandInContainer(f, testpod.Name, testpod.Spec.Containers[0].Name, command...)

				for _, mount := range mounts {
					ginkgo.By(fmt.Sprintf("checking that the hugetlb mount %s exists under the container", mount.MountPath))
					gomega.Expect(out).To(gomega.ContainSubstring(mount.MountPath))
				}

				for resourceName := range hugepages {
					verifyPod := makePodToVerifyHugePages(
						"pod"+string(testpod.UID),
						testpod.Spec.Containers[0].Resources.Limits[v1.ResourceName(resourceName)],
						resourceToCgroup[resourceName],
					)
					ginkgo.By("checking if the expected hugetlb settings were applied")
					e2epod.NewPodClient(f).Create(ctx, verifyPod)
					err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, verifyPod.Name, f.Namespace.Name)
					framework.ExpectNoError(err)
				}
			})
		}

		// setup
		ginkgo.JustBeforeEach(func(ctx context.Context) {
			setHugepages(ctx)

			ginkgo.By("restarting kubelet to pick up pre-allocated hugepages")
			restartKubelet(ctx, true)

			waitForHugepages(ctx)

			pod := getHugepagesTestPod(f, limits, mounts, volumes)

			ginkgo.By("by running a test pod that requests hugepages")
			testpod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
		})

		// we should use JustAfterEach because framework will teardown the client under the AfterEach method
		ginkgo.JustAfterEach(func(ctx context.Context) {
			ginkgo.By(fmt.Sprintf("deleting test pod %s", testpod.Name))
			e2epod.NewPodClient(f).DeleteSync(ctx, testpod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

			releaseHugepages(ctx)

			ginkgo.By("restarting kubelet to pick up pre-allocated hugepages")
			restartKubelet(ctx, true)

			waitForHugepages(ctx)
		})

		ginkgo.Context("with the resources requests that contain only one hugepages resource ", func() {
			ginkgo.Context("with the backward compatible API", func() {
				ginkgo.BeforeEach(func() {
					limits = v1.ResourceList{
						v1.ResourceCPU:           resource.MustParse("10m"),
						v1.ResourceMemory:        resource.MustParse("100Mi"),
						hugepagesResourceName2Mi: resource.MustParse("6Mi"),
					}
					mounts = []v1.VolumeMount{
						{
							Name:      "hugepages",
							MountPath: "/hugepages",
						},
					}
					volumes = []v1.Volume{
						{
							Name: "hugepages",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium: mediumHugepages,
								},
							},
						},
					}
					hugepages = map[string]int{hugepagesResourceName2Mi: 5}
				})
				// run tests
				runHugePagesTests()
			})

			ginkgo.Context("with the new API", func() {
				ginkgo.BeforeEach(func() {
					limits = v1.ResourceList{
						v1.ResourceCPU:           resource.MustParse("10m"),
						v1.ResourceMemory:        resource.MustParse("100Mi"),
						hugepagesResourceName2Mi: resource.MustParse("6Mi"),
					}
					mounts = []v1.VolumeMount{
						{
							Name:      "hugepages-2mi",
							MountPath: "/hugepages-2Mi",
						},
					}
					volumes = []v1.Volume{
						{
							Name: "hugepages-2mi",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium: mediumHugepages2Mi,
								},
							},
						},
					}
					hugepages = map[string]int{hugepagesResourceName2Mi: 5}
				})

				runHugePagesTests()
			})

			ginkgo.JustAfterEach(func() {
				hugepages = map[string]int{hugepagesResourceName2Mi: 0}
			})
		})

		ginkgo.Context("with the resources requests that contain multiple hugepages resources ", func() {
			ginkgo.BeforeEach(func() {
				hugepages = map[string]int{
					hugepagesResourceName2Mi: 5,
					hugepagesResourceName1Gi: 1,
				}
				limits = v1.ResourceList{
					v1.ResourceCPU:           resource.MustParse("10m"),
					v1.ResourceMemory:        resource.MustParse("100Mi"),
					hugepagesResourceName2Mi: resource.MustParse("6Mi"),
					hugepagesResourceName1Gi: resource.MustParse("1Gi"),
				}
				mounts = []v1.VolumeMount{
					{
						Name:      "hugepages-2mi",
						MountPath: "/hugepages-2Mi",
					},
					{
						Name:      "hugepages-1gi",
						MountPath: "/hugepages-1Gi",
					},
				}
				volumes = []v1.Volume{
					{
						Name: "hugepages-2mi",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{
								Medium: mediumHugepages2Mi,
							},
						},
					},
					{
						Name: "hugepages-1gi",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{
								Medium: mediumHugepages1Gi,
							},
						},
					},
				}
			})

			runHugePagesTests()

			ginkgo.JustAfterEach(func() {
				hugepages = map[string]int{
					hugepagesResourceName2Mi: 0,
					hugepagesResourceName1Gi: 0,
				}
			})
		})
	})
})
