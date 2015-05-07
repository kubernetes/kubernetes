/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/aws"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("PD", func() {
	var (
		c         *client.Client
		podClient client.PodInterface
		host0Name string
		host1Name string
		numHosts  int
	)

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)

		podClient = c.Pods(api.NamespaceDefault)

		nodes, err := c.Nodes().List(labels.Everything(), fields.Everything())
		expectNoError(err, "Failed to list nodes for e2e cluster.")

		numHosts = len(nodes.Items)

		if len(nodes.Items) >= 2 {
			host1Name = nodes.Items[1].ObjectMeta.Name
		}
		if len(nodes.Items) >= 1 {
			host0Name = nodes.Items[0].ObjectMeta.Name
		}
	})

	It("should schedule a pod w/ a RW PD, remove it, then schedule it on another host", func() {
		if !providerIs("gce", "aws") {
			By(fmt.Sprintf("Skipping PD test, which is only supported for providers gce & aws (not %s)",
				testContext.Provider))
			return
		}

		Expect(numHosts >= 2).To(BeTrue(), "At least 2 nodes required")

		By("creating PD")
		diskName, err := createPD()
		expectNoError(err, "Error creating PD")

		host0Pod := testPDPod(diskName, host0Name, false)
		host1Pod := testPDPod(diskName, host1Name, false)

		defer func() {
			By("cleaning up PD-RW test environment")
			// Teardown pods, PD. Ignore errors.
			// Teardown should do nothing unless test failed.
			podClient.Delete(host0Pod.Name, nil)
			podClient.Delete(host1Pod.Name, nil)
			detachPD(host0Name, diskName)
			detachPD(host1Name, diskName)
			deletePD(diskName)
		}()

		By("submitting host0Pod to kubernetes")
		_, err = podClient.Create(host0Pod)
		expectNoError(err, fmt.Sprintf("Failed to create host0Pod: %v", err))

		expectNoError(waitForPodRunning(c, host0Pod.Name))

		By("deleting host0Pod")
		expectNoError(podClient.Delete(host0Pod.Name, nil), "Failed to delete host0Pod")

		By("submitting host1Pod to kubernetes")
		_, err = podClient.Create(host1Pod)
		expectNoError(err, "Failed to create host1Pod")

		expectNoError(waitForPodRunning(c, host1Pod.Name))

		By("deleting host1Pod")
		expectNoError(podClient.Delete(host1Pod.Name, nil), "Failed to delete host1Pod")

		By(fmt.Sprintf("deleting PD %q", diskName))
		for start := time.Now(); time.Since(start) < 180*time.Second; time.Sleep(5 * time.Second) {
			if err = deletePD(diskName); err != nil {
				Logf("Couldn't delete PD. Sleeping 5 seconds (%v)", err)
				continue
			}
			Logf("Deleted PD %v", diskName)
			break
		}
		expectNoError(err, "Error deleting PD")

		return
	})

	It("should schedule a pod w/ a readonly PD on two hosts, then remove both.", func() {
		if testContext.Provider != "gce" {
			By(fmt.Sprintf("Skipping PD test, which is only supported for provider gce (not %s)",
				testContext.Provider))
			return
		}

		Expect(numHosts >= 2).To(BeTrue(), "At least 2 nodes required")

		By("creating PD")
		diskName, err := createPD()
		expectNoError(err, "Error creating PD")

		rwPod := testPDPod(diskName, host0Name, false)
		host0ROPod := testPDPod(diskName, host0Name, true)
		host1ROPod := testPDPod(diskName, host1Name, true)

		defer func() {
			By("cleaning up PD-RO test environment")
			// Teardown pods, PD. Ignore errors.
			// Teardown should do nothing unless test failed.
			podClient.Delete(rwPod.Name, nil)
			podClient.Delete(host0ROPod.Name, nil)
			podClient.Delete(host1ROPod.Name, nil)

			detachPD(host0Name, diskName)
			detachPD(host1Name, diskName)
			deletePD(diskName)
		}()

		By("submitting rwPod to ensure PD is formatted")
		_, err = podClient.Create(rwPod)
		expectNoError(err, "Failed to create rwPod")
		expectNoError(waitForPodRunning(c, rwPod.Name))
		expectNoError(podClient.Delete(rwPod.Name, nil), "Failed to delete host0Pod")

		By("submitting host0ROPod to kubernetes")
		_, err = podClient.Create(host0ROPod)
		expectNoError(err, "Failed to create host0ROPod")

		By("submitting host1ROPod to kubernetes")
		_, err = podClient.Create(host1ROPod)
		expectNoError(err, "Failed to create host1ROPod")

		expectNoError(waitForPodRunning(c, host0ROPod.Name))

		expectNoError(waitForPodRunning(c, host1ROPod.Name))

		By("deleting host0ROPod")
		expectNoError(podClient.Delete(host0ROPod.Name, nil), "Failed to delete host0ROPod")

		By("deleting host1ROPod")
		expectNoError(podClient.Delete(host1ROPod.Name, nil), "Failed to delete host1ROPod")

		By(fmt.Sprintf("deleting PD %q", diskName))
		for start := time.Now(); time.Since(start) < 180*time.Second; time.Sleep(5 * time.Second) {
			if err = deletePD(diskName); err != nil {
				Logf("Couldn't delete PD. Sleeping 5 seconds")
				continue
			}
			break
		}
		expectNoError(err, "Error deleting PD")
	})
})

func createPD() (string, error) {
	if testContext.Provider == "gce" {
		pdName := fmt.Sprintf("e2e-%s", string(util.NewUUID()))

		zone := testContext.CloudConfig.Zone
		// TODO: make this hit the compute API directly instread of shelling out to gcloud.
		err := exec.Command("gcloud", "compute", "disks", "create", "--zone="+zone, "--size=10GB", pdName).Run()
		if err != nil {
			return "", err
		}
		return pdName, nil
	} else {
		volumes, ok := testContext.CloudConfig.Provider.(aws_cloud.Volumes)
		if !ok {
			return "", fmt.Errorf("Provider does not support volumes")
		}
		volumeOptions := &aws_cloud.VolumeOptions{}
		volumeOptions.CapacityMB = 10 * 1024
		return volumes.CreateVolume(volumeOptions)
	}
}

func deletePD(pdName string) error {
	if testContext.Provider == "gce" {
		zone := testContext.CloudConfig.Zone

		// TODO: make this hit the compute API directly.
		return exec.Command("gcloud", "compute", "disks", "delete", "--zone="+zone, pdName).Run()
	} else {
		volumes, ok := testContext.CloudConfig.Provider.(aws_cloud.Volumes)
		if !ok {
			return fmt.Errorf("Provider does not support volumes")
		}
		return volumes.DeleteVolume(pdName)
	}
}

func detachPD(hostName, pdName string) error {
	if testContext.Provider == "gce" {
		instanceName := strings.Split(hostName, ".")[0]

		zone := testContext.CloudConfig.Zone

		// TODO: make this hit the compute API directly.
		return exec.Command("gcloud", "compute", "instances", "detach-disk", "--zone="+zone, "--disk="+pdName, instanceName).Run()
	} else {
		volumes, ok := testContext.CloudConfig.Provider.(aws_cloud.Volumes)
		if !ok {
			return fmt.Errorf("Provider does not support volumes")
		}
		return volumes.DetachDisk(hostName, pdName)
	}
}

func testPDPod(diskName, targetHost string, readOnly bool) *api.Pod {
	pod := &api.Pod{
		TypeMeta: api.TypeMeta{
			Kind:       "Pod",
			APIVersion: latest.Version,
		},
		ObjectMeta: api.ObjectMeta{
			Name: "pd-test-" + string(util.NewUUID()),
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "testpd",
					Image: "gcr.io/google_containers/pause",
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

	if testContext.Provider == "gce" {
		pod.Spec.Volumes = []api.Volume{
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
		}
	} else if testContext.Provider == "aws" {
		pod.Spec.Volumes = []api.Volume{
			{
				Name: "testpd",
				VolumeSource: api.VolumeSource{
					AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
						VolumeID: diskName,
						FSType:   "ext4",
						ReadOnly: readOnly,
					},
				},
			},
		}
	} else {
		panic("Unknown provider: " + testContext.Provider)
	}

	return pod
}
