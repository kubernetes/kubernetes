/*
Copyright 2015 Google Inc. All rights reserved.

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

package e2e

import (
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("PD", func() {
	var (
		c         *client.Client
		podClient client.PodInterface
		diskName  string
		host0Name string
		host1Name string
	)

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)

		podClient = c.Pods(api.NamespaceDefault)

		nodes, err := c.Nodes().List()
		expectNoError(err, "Failed to list nodes for e2e cluster.")
		Expect(len(nodes.Items) >= 2).To(BeTrue())

		diskName = fmt.Sprintf("e2e-%s", string(util.NewUUID()))
		host0Name = nodes.Items[0].ObjectMeta.Name
		host1Name = nodes.Items[1].ObjectMeta.Name
	})

	It("should schedule a pod w/ a RW PD, remove it, then schedule it on another host", func() {
		host0Pod := testPDPod(diskName, host0Name, false)
		host1Pod := testPDPod(diskName, host1Name, false)

		By(fmt.Sprintf("creating PD %q", diskName))
		expectNoError(createPD(diskName, testContext.gceConfig.Zone), "Error creating PD")

		defer func() {
			By("cleaning up PD-RW test environment")
			// Teardown pods, PD. Ignore errors.
			// Teardown should do nothing unless test failed.
			podClient.Delete(host0Pod.Name)
			podClient.Delete(host1Pod.Name)
			detachPD(host0Name, diskName, testContext.gceConfig.Zone)
			detachPD(host1Name, diskName, testContext.gceConfig.Zone)
			deletePD(diskName, testContext.gceConfig.Zone)
		}()

		By("submitting host0Pod to kubernetes")
		_, err := podClient.Create(host0Pod)
		expectNoError(err, fmt.Sprintf("Failed to create host0Pod: %v", err))

		expectNoError(waitForPodRunning(c, host0Pod.Name))

		By("deleting host0Pod")
		expectNoError(podClient.Delete(host0Pod.Name), "Failed to delete host0Pod")

		By("submitting host1Pod to kubernetes")
		_, err = podClient.Create(host1Pod)
		expectNoError(err, "Failed to create host1Pod")

		expectNoError(waitForPodRunning(c, host1Pod.Name))

		By("deleting host1Pod")
		expectNoError(podClient.Delete(host1Pod.Name), "Failed to delete host1Pod")

		By(fmt.Sprintf("deleting PD %q", diskName))
		for start := time.Now(); time.Since(start) < 180*time.Second; time.Sleep(5 * time.Second) {
			if err = deletePD(diskName, testContext.gceConfig.Zone); err != nil {
				Logf("Couldn't delete PD. Sleeping 5 seconds")
				continue
			}
			break
		}
		expectNoError(err, "Error deleting PD")

		return
	})

	It("should schedule a pod w/ a readonly PD on two hosts, then remove both.", func() {
		rwPod := testPDPod(diskName, host0Name, false)
		host0ROPod := testPDPod(diskName, host0Name, true)
		host1ROPod := testPDPod(diskName, host1Name, true)

		defer func() {
			By("cleaning up PD-RO test environment")
			// Teardown pods, PD. Ignore errors.
			// Teardown should do nothing unless test failed.
			podClient.Delete(rwPod.Name)
			podClient.Delete(host0ROPod.Name)
			podClient.Delete(host1ROPod.Name)
			detachPD(host0Name, diskName, testContext.gceConfig.Zone)
			detachPD(host1Name, diskName, testContext.gceConfig.Zone)
			deletePD(diskName, testContext.gceConfig.Zone)
		}()

		By(fmt.Sprintf("creating PD %q", diskName))
		expectNoError(createPD(diskName, testContext.gceConfig.Zone), "Error creating PD")

		By("submitting rwPod to ensure PD is formatted")
		_, err := podClient.Create(rwPod)
		expectNoError(err, "Failed to create rwPod")
		expectNoError(waitForPodRunning(c, rwPod.Name))
		expectNoError(podClient.Delete(rwPod.Name), "Failed to delete host0Pod")

		By("submitting host0ROPod to kubernetes")
		_, err = podClient.Create(host0ROPod)
		expectNoError(err, "Failed to create host0ROPod")

		By("submitting host1ROPod to kubernetes")
		_, err = podClient.Create(host1ROPod)
		expectNoError(err, "Failed to create host1ROPod")

		expectNoError(waitForPodRunning(c, host0ROPod.Name))

		expectNoError(waitForPodRunning(c, host1ROPod.Name))

		By("deleting host0ROPod")
		expectNoError(podClient.Delete(host0ROPod.Name), "Failed to delete host0ROPod")

		By("deleting host1ROPod")
		expectNoError(podClient.Delete(host1ROPod.Name), "Failed to delete host1ROPod")

		By(fmt.Sprintf("deleting PD %q", diskName))
		for start := time.Now(); time.Since(start) < 180*time.Second; time.Sleep(5 * time.Second) {
			if err = deletePD(diskName, testContext.gceConfig.Zone); err != nil {
				Logf("Couldn't delete PD. Sleeping 5 seconds")
				continue
			}
			break
		}
		expectNoError(err, "Error deleting PD")
	})
})

func createPD(pdName, zone string) error {
	// TODO: make this hit the compute API directly instread of shelling out to gcloud.
	return exec.Command("gcloud", "compute", "disks", "create", "--zone="+zone, "--size=10GB", pdName).Run()
}

func deletePD(pdName, zone string) error {
	// TODO: make this hit the compute API directly.
	return exec.Command("gcloud", "compute", "disks", "delete", "--zone="+zone, pdName).Run()
}

func detachPD(hostName, pdName, zone string) error {
	instanceName := strings.Split(hostName, ".")[0]
	// TODO: make this hit the compute API directly.
	return exec.Command("gcloud", "compute", "instances", "detach-disk", "--zone="+zone, "--disk="+pdName, instanceName).Run()
}

func testPDPod(diskName, targetHost string, readOnly bool) *api.Pod {
	return &api.Pod{
		TypeMeta: api.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1beta1",
		},
		ObjectMeta: api.ObjectMeta{
			Name: "pd-test-" + string(util.NewUUID()),
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "testpd",
					VolumeSource: api.VolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
							PDName:   diskName,
							FSType:   "ext4",
							ReadOnly: readOnly,
						},
					},
				},
			},
			Containers: []api.Container{
				{
					Name:  "testpd",
					Image: "kubernetes/pause",
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "testpd",
							MountPath: "/testpd",
						},
					},
				},
			},
			Host: targetHost,
		},
	}
}
