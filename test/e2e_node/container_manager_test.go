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
	"path"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	kubeletProcessname = "kubelet"
)

func getOOMScoreForPid(pid int) (int, error) {
	procfsPath := path.Join("/proc", strconv.Itoa(pid), "oom_score_adj")
	content, err := ioutil.ReadFile(procfsPath)
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(strings.TrimSpace(string(content)))
}

func validateOOMScoreAdjSetting(pid int, expectedOOMScoreAdj int) {
	oomScore, err := getOOMScoreForPid(pid)
	Expect(err).To(BeNil(), "failed to get oom_score_adj for %d", pid)
	Expect(oomScore).To(BeNumerically("==", expectedOOMScoreAdj), "expected pid %d's oom_score_adj to be %d; found %d", pid, expectedOOMScoreAdj, oomScore)
}

func validateOOMScoreAdjSettingIsInRange(pid int, expectedMinOOMScoreAdj, expectedMaxOOMScoreAdj int32) {
	oomScore, err := getOOMScoreForPid(pid)
	Expect(err).To(BeNil(), "failed to get oom_score_adj for %d", pid)
	Expect(oomScore).To(BeNumerically(">=", expectedMinOOMScoreAdj), "expected pid %d's oom_score_adj to be >= %d; found %d", pid, expectedMinOOMScoreAdj, oomScore)
	Expect(oomScore).To(BeNumerically("<", expectedMaxOOMScoreAdj), "expected pid %d's oom_score_adj to be < %d; found %d", pid, expectedMaxOOMScoreAdj, oomScore)
}

var _ = framework.KubeDescribe("Kubelet Container Manager [Serial]", func() {
	f := framework.NewDefaultFramework("kubelet-container-manager")

	Describe("Validate OOM score adjustments", func() {
		Context("once the node is setup", func() {
			It("docker daemon's oom-score-adj should be -999", func() {
				dockerPids, err := getPidsForProcess(dockerProcessName, dockerPidFile)
				Expect(err).To(BeNil(), "failed to get list of docker daemon pids")
				for _, pid := range dockerPids {
					validateOOMScoreAdjSetting(pid, -999)
				}
			})
			It("Kubelet's oom-score-adj should be -999", func() {
				kubeletPids, err := getPidsForProcess(kubeletProcessName, "")
				Expect(err).To(BeNil(), "failed to get list of kubelet pids")
				Expect(len(kubeletPids)).To(Equal(1), "expected only one kubelet process; found %d", len(kubeletPids))
				validateOOMScoreAdjSetting(kubeletPids[0], -999)
			})
			It("pod infra containers oom-score-adj should be -998 and best effort container's should be 1000", func() {
				var err error
				podClient := f.PodClient()
				podName := "besteffort" + string(uuid.NewUUID())
				podClient.Create(&api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: podName,
					},
					Spec: api.PodSpec{
						Containers: []api.Container{
							{
								Image: ImageRegistry[serveHostnameImage],
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
					return nil
				}, 2*time.Minute, time.Second*4).Should(BeNil())
				for _, pid := range pausePids {
					validateOOMScoreAdjSetting(pid, -998)
				}
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
					return nil
				}, 2*time.Minute, time.Second*4).Should(BeNil())
				validateOOMScoreAdjSetting(shPids[0], 1000)
			})
			It("guaranteed container's oom-score-adj should be -998", func() {
				podClient := f.PodClient()
				podName := "guaranteed" + string(uuid.NewUUID())
				podClient.Create(&api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: podName,
					},
					Spec: api.PodSpec{
						Containers: []api.Container{
							{
								Image: ImageRegistry[nginxImage],
								Name:  podName,
								Resources: api.ResourceRequirements{
									Limits: api.ResourceList{
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
					return nil
				}, 2*time.Minute, time.Second*4).Should(BeNil())

				for _, pid := range ngPids {
					validateOOMScoreAdjSetting(pid, -998)
				}
			})
			It("burstable container's oom-score-adj should be between [2, 1000)", func() {
				podClient := f.PodClient()
				podName := "burstable" + string(uuid.NewUUID())
				podClient.Create(&api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: podName,
					},
					Spec: api.PodSpec{
						Containers: []api.Container{
							{
								Image: ImageRegistry[testWebServer],
								Name:  podName,
								Resources: api.ResourceRequirements{
									Requests: api.ResourceList{
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
					return nil
				}, 2*time.Minute, time.Second*4).Should(BeNil())

				for _, pid := range wsPids {
					validateOOMScoreAdjSettingIsInRange(pid, 2, 1000)
				}
				// TODO: Test the oom-score-adj logic for burstable more accurately.
			})
		})
	})
})
