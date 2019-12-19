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

package e2enode

import (
	"fmt"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
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

var _ = framework.KubeDescribe("Container Manager Misc [Serial]", func() {
	f := framework.NewDefaultFramework("kubelet-container-manager")
	ginkgo.Describe("Validate OOM score adjustments [NodeFeature:OOMScoreAdj]", func() {
		ginkgo.Context("once the node is setup", func() {
			ginkgo.It("container runtime's oom-score-adj should be -999", func() {
				runtimePids, err := getPidsForProcess(framework.TestContext.ContainerRuntimeProcessName, framework.TestContext.ContainerRuntimePidFile)
				framework.ExpectNoError(err, "failed to get list of container runtime pids")
				for _, pid := range runtimePids {
					gomega.Eventually(func() error {
						return validateOOMScoreAdjSetting(pid, -999)
					}, 5*time.Minute, 30*time.Second).Should(gomega.BeNil())
				}
			})
			ginkgo.It("Kubelet's oom-score-adj should be -999", func() {
				kubeletPids, err := getPidsForProcess(kubeletProcessName, "")
				framework.ExpectNoError(err, "failed to get list of kubelet pids")
				framework.ExpectEqual(len(kubeletPids), 1, "expected only one kubelet process; found %d", len(kubeletPids))
				gomega.Eventually(func() error {
					return validateOOMScoreAdjSetting(kubeletPids[0], -999)
				}, 5*time.Minute, 30*time.Second).Should(gomega.BeNil())
			})
			ginkgo.Context("", func() {
				ginkgo.It("pod infra containers oom-score-adj should be -998 and best effort container's should be 1000", func() {
					// Take a snapshot of existing pause processes. These were
					// created before this test, and may not be infra
					// containers. They should be excluded from the test.
					existingPausePIDs, err := getPidsForProcess("pause", "")
					framework.ExpectNoError(err, "failed to list all pause processes on the node")
					existingPausePIDSet := sets.NewInt(existingPausePIDs...)

					podClient := f.PodClient()
					podName := "besteffort" + string(uuid.NewUUID())
					podClient.Create(&v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name: podName,
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Image: framework.ServeHostnameImage,
									Name:  podName,
								},
							},
						},
					})

					var pausePids []int
					ginkgo.By("checking infra container's oom-score-adj")
					gomega.Eventually(func() error {
						pausePids, err = getPidsForProcess("pause", "")
						if err != nil {
							return fmt.Errorf("failed to get list of pause pids: %v", err)
						}
						for _, pid := range pausePids {
							if existingPausePIDSet.Has(pid) {
								// Not created by this test. Ignore it.
								continue
							}
							if err := validateOOMScoreAdjSetting(pid, -998); err != nil {
								return err
							}
						}
						return nil
					}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())
					var shPids []int
					ginkgo.By("checking besteffort container's oom-score-adj")
					gomega.Eventually(func() error {
						shPids, err = getPidsForProcess("agnhost", "")
						if err != nil {
							return fmt.Errorf("failed to get list of serve hostname process pids: %v", err)
						}
						if len(shPids) != 1 {
							return fmt.Errorf("expected only one agnhost process; found %d", len(shPids))
						}
						return validateOOMScoreAdjSetting(shPids[0], 1000)
					}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())
				})
				// Log the running containers here to help debugging.
				ginkgo.AfterEach(func() {
					if ginkgo.CurrentGinkgoTestDescription().Failed {
						ginkgo.By("Dump all running containers")
						runtime, _, err := getCRIClient()
						framework.ExpectNoError(err)
						containers, err := runtime.ListContainers(&runtimeapi.ContainerFilter{
							State: &runtimeapi.ContainerStateValue{
								State: runtimeapi.ContainerState_CONTAINER_RUNNING,
							},
						})
						framework.ExpectNoError(err)
						framework.Logf("Running containers:")
						for _, c := range containers {
							framework.Logf("%+v", c)
						}
					}
				})
			})
			ginkgo.It("guaranteed container's oom-score-adj should be -998", func() {
				podClient := f.PodClient()
				podName := "guaranteed" + string(uuid.NewUUID())
				podClient.Create(&v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: podName,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Image: imageutils.GetE2EImage(imageutils.Nginx),
								Name:  podName,
								Resources: v1.ResourceRequirements{
									Limits: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("100m"),
										v1.ResourceMemory: resource.MustParse("50Mi"),
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
				gomega.Eventually(func() error {
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
				}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())

			})
			ginkgo.It("burstable container's oom-score-adj should be between [2, 1000)", func() {
				podClient := f.PodClient()
				podName := "burstable" + string(uuid.NewUUID())
				podClient.Create(&v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: podName,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Image: imageutils.GetE2EImage(imageutils.TestWebserver),
								Name:  podName,
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("100m"),
										v1.ResourceMemory: resource.MustParse("50Mi"),
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
				gomega.Eventually(func() error {
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
				}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())

				// TODO: Test the oom-score-adj logic for burstable more accurately.
			})
		})
	})
})
