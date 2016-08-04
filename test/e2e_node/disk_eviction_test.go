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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

const (
	// The interval seconds between pod status checks
	POD_CHECK_INTERVAL = 3

	DUMMY_FILE = "dummy.file"
)

// TODO: Leverage config map to change kubelet eviction option based on available disk space in e2e tests.
// To manually trigger the test for a node with disk space just over 15G :
//   make test-e2e-node FOCUS="hard eviction test" TEST_ARGS="--eviction-hard=nodefs.available<15Gi"
var _ = framework.KubeDescribe("Kubelet Eviction Manager [FLAKY] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("kubelet-eviction-manager")
	var podClient *framework.PodClient
	var c *client.Client
	var n *api.Node

	BeforeEach(func() {
		podClient = f.PodClient()
		c = f.Client
		nodeList := framework.GetReadySchedulableNodesOrDie(c)
		n = &nodeList.Items[0]
	})

	Describe("hard eviction test", func() {
		Context("pod gets evicted when the disk usage is above the hard threshold", func() {
			var podName string
			BeforeEach(func() {
				podName = "idle-hard-evict" + string(util.NewUUID())
				podClient.Create(&api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: podName,
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyNever,
						Containers: []api.Container{
							{
								Image: ImageRegistry[busyBoxImage],
								Name:  podName,
								// Kepp writing to disk
								Command: []string{"sh", "-c", fmt.Sprintf("while true; do dd if=/dev/urandom of=%s bs=100000000 count=10; done",
									DUMMY_FILE)},
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
				}, time.Minute*5, time.Second*POD_CHECK_INTERVAL).Should(BeNil())
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

	diskPressureConditionSet := false
	nodeList := framework.GetReadySchedulableNodesOrDie(c)
	for _, c := range nodeList.Items[0].Status.Conditions {
		if c.Type == "DiskPressure" {
			diskPressureConditionSet = c.Status == "True"
			break
		}
	}

	if !diskPressureConditionSet {
		return fmt.Errorf("expected disk pressure condition is not set")
	}
	return nil
}
