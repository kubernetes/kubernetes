// +build linux

/*
Copyright 2016 The Kubernetes Authors.

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
	"io/ioutil"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/pborman/uuid"
)

func getOOMScoreForPid(pid int) (int, error) {
	procfsPath := path.Join("/proc", strconv.Itoa(pid), "oom_score_adj")
	out, err := exec.Command("sudo", "cat", procfsPath).CombinedOutput()
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(strings.TrimSpace(string(out)))
}

func validateOOMScoreAdjSetting(pid int, expectedOOMScoreAdj int) error {
	oomScore, err := getOOMScoreForPid(pid)
	if err != nil {
		return fmt.Errorf("failed to get oom_score_adj for %d: %v", pid, err)
	}
	if expectedOOMScoreAdj != oomScore {
		return fmt.Errorf("expected pid %d's oom_score_adj to be %d; found %d", pid, expectedOOMScoreAdj, oomScore)
	}
	return nil
}

func validateOOMScoreAdjSettingIsInRange(pid int, expectedMinOOMScoreAdj, expectedMaxOOMScoreAdj int) error {
	oomScore, err := getOOMScoreForPid(pid)
	if err != nil {
		return fmt.Errorf("failed to get oom_score_adj for %d", pid)
	}
	if oomScore < expectedMinOOMScoreAdj {
		return fmt.Errorf("expected pid %d's oom_score_adj to be >= %d; found %d", pid, expectedMinOOMScoreAdj, oomScore)
	}
	if oomScore < expectedMaxOOMScoreAdj {
		return fmt.Errorf("expected pid %d's oom_score_adj to be < %d; found %d", pid, expectedMaxOOMScoreAdj, oomScore)
	}
	return nil
}

func expectFileValToEqual(filePath string, expectedValue, delta int64) {
	out, err := ioutil.ReadFile(filePath)
	Expect(err).To(BeNil(), "failed to read file %q", filePath)
	actual, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	Expect(err).To(BeNil(), "failed to parse output %v", err)
	// Ensure that values are within a delta range to work arounding rounding errors.
	Expect((actual < (expectedValue-delta)) || (actual > (expectedValue+delta))).To(BeFalse(), "Expected value at %q to be between %d and %d. Got %d", filePath, (expectedValue - delta), (expectedValue + delta), actual)
}

var _ = framework.KubeDescribe("Kubelet Container Manager Node Allocatable Enforcement [Serial]", func() {
	f := framework.NewDefaultFramework("kubelet-container-manager-na")

	Describe("Validate Node Allocatable", func() {
		Context("once the node is setup", func() {
			tempSetCurrentKubeletConfig(f, func(initialConfig *componentconfig.KubeletConfiguration) {
				initialConfig.EnforceNodeAllocatable = []string{"pods"}
				initialConfig.SystemReserved = componentconfig.ConfigurationMap{
					"cpu":    "100m",
					"memory": "100Mi",
				}
				initialConfig.KubeReserved = componentconfig.ConfigurationMap{
					"cpu":    "100m",
					"memory": "100Mi",
				}
				initialConfig.EvictionHard = "memory.available<100Mi"
				// Necessary for allocatable enforcement.
				initialConfig.CgroupsPerQOS = true
			})

			It("Node Allocatable Pod Cgroup Is Found", func() {
				currentConfig, err := getCurrentKubeletConfig()
				Expect(err).To(BeNil())
				if !currentConfig.CgroupsPerQOS || len(currentConfig.EnforceNodeAllocatable) == 0 {
					Skip("Required configuration not found. Skipping test")
				}
				expectedNAPodCgroup := path.Join(currentConfig.CgroupRoot, "kubepods")
				subsystems, err := cm.GetCgroupSubsystems()
				Expect(err).To(BeNil())
				cgroupManager := cm.NewCgroupManager(subsystems, currentConfig.CgroupDriver)
				Expect(cgroupManager.Exists(cm.CgroupName(expectedNAPodCgroup))).To(BeTrue(), "Expected Node Allocatable Cgroup Does not exist")
				// TODO: Update cgroupManager to expose a Status interface to get current Cgroup Settings.
				capacity := getLocalNode(f).Status.Capacity
				Expect(err).To(BeNil())
				var allocatableCPU, allocatableMemory *resource.Quantity
				// Total cpu reservation is 200m.
				for k, v := range capacity {
					if k == v1.ResourceCPU {
						allocatableCPU = v.Copy()
						allocatableCPU.Sub(resource.MustParse("200m"))
					}
					if k == v1.ResourceMemory {
						allocatableMemory = v.Copy()
						allocatableMemory.Sub(resource.MustParse("200Mi"))
					}
				}
				// Total Memory reservation is 200Mi excluding eviction thresholds.
				// Expect CPU shares on node allocatable cgroup to equal allocatable.
				expectFileValToEqual(filepath.Join(subsystems.MountPoints["cpu"], "kubepods", "cpu.shares"), cm.MilliCPUToShares(allocatableCPU.MilliValue()), 10)
				// Expect Memory limit on node allocatable cgroup to equal allocatable.
				expectFileValToEqual(filepath.Join(subsystems.MountPoints["memory"], "kubepods", "memory.limit_in_bytes"), allocatableMemory.Value(), 0)
				// Check that Allocatable reported to scheduler includes eviction thresholds.
				schedulerAllocatable := getLocalNode(f).Status.Allocatable
				// CPU based evictions are not supported.
				Expect(allocatableCPU.Cmp(schedulerAllocatable["cpu"])).To(Equal(0), "Unexpected cpu allocatable value exposed by the node. Expected: %v, got: %v, capacity: %v", allocatableCPU, schedulerAllocatable["cpu"], capacity["cpu"])
				// Memory allocatable should take into account eviction thresholds.
				allocatableMemory.Sub(resource.MustParse("100Mi"))
				Expect(allocatableMemory.Cmp(schedulerAllocatable["memory"])).To(Equal(0), "Unexpected cpu allocatable value exposed by the node. Expected: %v, got: %v, capacity: %v", allocatableCPU, schedulerAllocatable["cpu"], capacity["memory"])

			})
			// TODO: Add test for enforcing System Reserved and Kube Reserved. Requires creation of dummy cgroups.
		})
	})
})

var _ = framework.KubeDescribe("Kubelet Container Manager OOM score Enforcement [Serial]", func() {
	f := framework.NewDefaultFramework("kubelet-container-manager-oom")

	Describe("Validate OOM score adjustments", func() {
		Context("once the node is setup", func() {
			It("docker daemon's oom-score-adj should be -999", func() {
				dockerPids, err := getPidsForProcess(dockerProcessName, dockerPidFile)
				Expect(err).To(BeNil(), "failed to get list of docker daemon pids")
				for _, pid := range dockerPids {
					Eventually(func() error {
						return validateOOMScoreAdjSetting(pid, -999)
					}, 5*time.Minute, 30*time.Second).Should(BeNil())
				}
			})
			It("Kubelet's oom-score-adj should be -999", func() {
				kubeletPids, err := getPidsForProcess(kubeletProcessName, "")
				Expect(err).To(BeNil(), "failed to get list of kubelet pids")
				Expect(len(kubeletPids)).To(Equal(1), "expected only one kubelet process; found %d", len(kubeletPids))
				Eventually(func() error {
					return validateOOMScoreAdjSetting(kubeletPids[0], -999)
				}, 5*time.Minute, 30*time.Second).Should(BeNil())
			})
			Context("", func() {
				It("pod infra containers oom-score-adj should be -998 and best effort container's should be 1000", func() {
					var err error
					podClient := f.PodClient()
					podName := "besteffort" + string(uuid.NewUUID())
					podClient.Create(&v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name: podName,
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Image: "gcr.io/google_containers/serve_hostname:v1.4",
									Name:  podName,
								},
							},
						},
					})
					var pausePids []int
					By("checking infra container's oom-score-adj")
					Eventually(func() error {
						pausePids, err = getPidsForProcess("pause", "")
						if err != nil {
							return fmt.Errorf("failed to get list of pause pids: %v", err)
						}
						for _, pid := range pausePids {
							if err := validateOOMScoreAdjSetting(pid, -998); err != nil {
								return err
							}
						}
						return nil
					}, 2*time.Minute, time.Second*4).Should(BeNil())
					var shPids []int
					By("checking besteffort container's oom-score-adj")
					Eventually(func() error {
						shPids, err = getPidsForProcess("serve_hostname", "")
						if err != nil {
							return fmt.Errorf("failed to get list of serve hostname process pids: %v", err)
						}
						if len(shPids) != 1 {
							return fmt.Errorf("expected only one serve_hostname process; found %d", len(shPids))
						}
						return validateOOMScoreAdjSetting(shPids[0], 1000)
					}, 2*time.Minute, time.Second*4).Should(BeNil())
				})
				// Log the running containers here to help debugging. Use `docker ps`
				// directly for now because the test is already docker specific.
				AfterEach(func() {
					if CurrentGinkgoTestDescription().Failed {
						By("Dump all running docker containers")
						output, err := exec.Command("docker", "ps").CombinedOutput()
						Expect(err).NotTo(HaveOccurred())
						framework.Logf("Running docker containers:\n%s", string(output))
					}
				})
			})
			It("guaranteed container's oom-score-adj should be -998", func() {
				podClient := f.PodClient()
				podName := "guaranteed" + string(uuid.NewUUID())
				podClient.Create(&v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: podName,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Image: "gcr.io/google_containers/nginx-slim:0.7",
								Name:  podName,
								Resources: v1.ResourceRequirements{
									Limits: v1.ResourceList{
										"cpu":    resource.MustParse("100m"),
										"memory": resource.MustParse("50Mi"),
									},
								},
							},
						},
					},
				})
				var (
					ngPids []int
					err    error
				)
				Eventually(func() error {
					ngPids, err = getPidsForProcess("nginx", "")
					if err != nil {
						return fmt.Errorf("failed to get list of nginx process pids: %v", err)
					}
					for _, pid := range ngPids {
						if err := validateOOMScoreAdjSetting(pid, -998); err != nil {
							return err
						}
					}

					return nil
				}, 2*time.Minute, time.Second*4).Should(BeNil())

			})
			It("burstable container's oom-score-adj should be between [2, 1000)", func() {
				podClient := f.PodClient()
				podName := "burstable" + string(uuid.NewUUID())
				podClient.Create(&v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: podName,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Image: "gcr.io/google_containers/test-webserver:e2e",
								Name:  podName,
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										"cpu":    resource.MustParse("100m"),
										"memory": resource.MustParse("50Mi"),
									},
								},
							},
						},
					},
				})
				var (
					wsPids []int
					err    error
				)
				Eventually(func() error {
					wsPids, err = getPidsForProcess("test-webserver", "")
					if err != nil {
						return fmt.Errorf("failed to get list of test-webserver process pids: %v", err)
					}
					for _, pid := range wsPids {
						if err := validateOOMScoreAdjSettingIsInRange(pid, 2, 1000); err != nil {
							return err
						}
					}
					return nil
				}, 2*time.Minute, time.Second*4).Should(BeNil())

				// TODO: Test the oom-score-adj logic for burstable more accurately.
			})
		})
	})
})
