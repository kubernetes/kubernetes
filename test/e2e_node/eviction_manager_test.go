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
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

const (
	POD_CHECK_INTERVAL = 3
)

// TODO: Leverage config map to change kubelet option in e2e tests.
// For now it's marked as flaky to avoid affecting other e2e tests.
// To manuall trigger the test:
//   make test-e2e-node FOCUS="hard eviction test" TEST_ARGS="--eviction-hard=memory.available<1000Gi"
//   make test-e2e-node FOCUS="hard eviction test" TEST_ARGS="--eviction-hard=nodefs.available<1000Gi"
//   make test-e2e-node FOCUS="soft eviction test" TEST_ARGS="--eviction-soft=memory.available<1000Gi"
//   make test-e2e-node FOCUS="soft eviction test" TEST_ARGS="--eviction-soft=nodefs.available<1000Gi"
var _ = framework.KubeDescribe("Kubelet Eviction Manager [FLAKY]", func() {
	f := framework.NewDefaultFramework("kubelet-eviction-manager")
	var podClient *framework.PodClient

	BeforeEach(func() {
		podClient = f.PodClient()
	})

	Describe("hard eviction test", func() {
		Context("pod gets evicted when the resource usage is above the hard threshold", func() {
			var podName string

			BeforeEach(func() {
				podName = "idle-hard-evict" + string(util.NewUUID())
				podClient.Create(&api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: podName,
					},
					Spec: api.PodSpec{
						// Don't restart the Pod since it is expected to exit
						RestartPolicy: api.RestartPolicyNever,
						Containers: []api.Container{
							{
								Image: ImageRegistry[pauseImage],
								Name:  podName,
							},
						},
					},
				})
			})

			It("should be evicted", func() {
				Eventually(func() error {
					podData, err := podClient.Get(podName)
					if err != nil {
						return err
					}
					err = checkEviction(podData, f.Client)
					if err != nil {
						return err
					}
					return nil
				}, time.Second*30, time.Second*POD_CHECK_INTERVAL).Should(BeNil())
			})
		})
	})

	Describe("soft eviction test", func() {
		Context("pod gets evicted when when the resource usage keeps hovering above the soft threshold", func() {
			var podName string

			BeforeEach(func() {
				podName = "idle-soft-evict" + string(util.NewUUID())
				podClient.Create(&api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: podName,
					},
					Spec: api.PodSpec{
						// Don't restart the Pod since it is expected to exit
						RestartPolicy: api.RestartPolicyNever,
						Containers: []api.Container{
							{
								Image: ImageRegistry[pauseImage],
								Name:  podName,
							},
						},
					},
				})
			})

			It("should be evicted", func() {
				var transitionTimeThreshold time.Time
				Eventually(func() error {
					podData, err := podClient.Get(podName)
					if err != nil {
						return err
					}

					if transitionTimeThreshold.IsZero() {
						transitionTimeThreshold = time.Now().Add(time.Duration(SOFT_EVICTION_GRACE_PERIOD-POD_CHECK_INTERVAL) * time.Second)
					}

					err = checkEviction(podData, f.Client)
					if err != nil {
						return err
					}

					if time.Now().Before(transitionTimeThreshold) {
						return fmt.Errorf("Grace period %d is not honored", SOFT_EVICTION_GRACE_PERIOD)
					}
					return nil
				}, time.Second*50, time.Second*POD_CHECK_INTERVAL).Should(BeNil())
			})
		})
	})
})

func checkEviction(podData *api.Pod, c *client.Client) error {
	if podData.Status.Phase != "Failed" {
		return fmt.Errorf("expected phase to be failed. Got %+v", podData.Status.Phase)
	}
	if podData.Status.Reason != "Evicted" {
		return fmt.Errorf("expected state reason to be evicted. Got %+v", podData.Status.Reason)
	}

	memoryPressureConditionSet, diskPressureConditionSet := false, false
	nodeList := framework.GetReadySchedulableNodesOrDie(c)
	for _, c := range nodeList.Items[0].Status.Conditions {
		if c.Type == "MemoryPressure" {
			memoryPressureConditionSet = c.Status == "True"
		}
		if c.Type == "DiskPressure" {
			diskPressureConditionSet = c.Status == "True"
		}
	}

	if (strings.Contains(podData.Status.Message, "memory") && !memoryPressureConditionSet) ||
		(strings.Contains(podData.Status.Message, "disk") && !diskPressureConditionSet) {
		return fmt.Errorf("expected node condition is not set: %v %v %v",
			memoryPressureConditionSet, diskPressureConditionSet, podData.Status.Message)
	}
	return nil
}
