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

package scheduling

import (
	"fmt"
	"strconv"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	compute "google.golang.org/api/compute/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = SIGDescribe("Multi-AZ Cluster Volumes [sig-storage]", func() {
	f := framework.NewDefaultFramework("multi-az")
	var zoneCount int
	var err error
	image := framework.ServeHostnameImage
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke")
		if zoneCount <= 0 {
			zoneCount, err = getZoneCount(f.ClientSet)
			Expect(err).NotTo(HaveOccurred())
		}
		By(fmt.Sprintf("Checking for multi-zone cluster.  Zone count = %d", zoneCount))
		msg := fmt.Sprintf("Zone count is %d, only run for multi-zone clusters, skipping test", zoneCount)
		framework.SkipUnlessAtLeast(zoneCount, 2, msg)
		// TODO: SkipUnlessDefaultScheduler() // Non-default schedulers might not spread
	})
	It("should schedule pods in the same zones as statically provisioned PVs", func() {
		PodsUseStaticPVsOrFail(f, (2*zoneCount)+1, image)
	})

	It("should only be allowed to provision PDs in zones where nodes exist", func() {
		OnlyAllowNodeZones(f, zoneCount, image)
	})
})

// OnlyAllowNodeZones tests that GetAllCurrentZones returns only zones with Nodes
func OnlyAllowNodeZones(f *framework.Framework, zoneCount int, image string) {
	gceCloud, err := framework.GetGCECloud()
	Expect(err).NotTo(HaveOccurred())

	// Get all the zones that the nodes are in
	expectedZones, err := gceCloud.GetAllZonesFromCloudProvider()
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Expected zones: %v", expectedZones)

	// Get all the zones in this current region
	region := gceCloud.Region()
	allZonesInRegion, err := gceCloud.ListZonesInRegion(region)
	Expect(err).NotTo(HaveOccurred())

	var extraZone string
	for _, zone := range allZonesInRegion {
		if !expectedZones.Has(zone.Name) {
			extraZone = zone.Name
			break
		}
	}
	Expect(extraZone).NotTo(Equal(""), fmt.Sprintf("No extra zones available in region %s", region))

	By(fmt.Sprintf("starting a compute instance in unused zone: %v\n", extraZone))
	project := framework.TestContext.CloudConfig.ProjectID
	zone := extraZone
	myuuid := string(uuid.NewUUID())
	name := "compute-" + myuuid
	imageURL := "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/debian-7-wheezy-v20140606"

	rb := &compute.Instance{
		MachineType: "zones/" + zone + "/machineTypes/f1-micro",
		Disks: []*compute.AttachedDisk{
			{
				AutoDelete: true,
				Boot:       true,
				Type:       "PERSISTENT",
				InitializeParams: &compute.AttachedDiskInitializeParams{
					DiskName:    "my-root-pd-" + myuuid,
					SourceImage: imageURL,
				},
			},
		},
		NetworkInterfaces: []*compute.NetworkInterface{
			{
				AccessConfigs: []*compute.AccessConfig{
					{
						Type: "ONE_TO_ONE_NAT",
						Name: "External NAT",
					},
				},
				Network: "/global/networks/default",
			},
		},
		Name: name,
	}

	err = gceCloud.InsertInstance(project, zone, rb)
	Expect(err).NotTo(HaveOccurred())

	defer func() {
		// Teardown of the compute instance
		framework.Logf("Deleting compute resource: %v", name)
		err := gceCloud.DeleteInstance(project, zone, name)
		Expect(err).NotTo(HaveOccurred())
	}()

	By("Creating zoneCount+1 PVCs and making sure PDs are only provisioned in zones with nodes")
	// Create some (zoneCount+1) PVCs with names of form "pvc-x" where x is 1...zoneCount+1
	// This will exploit ChooseZoneForVolume in pkg/volume/util.go to provision them in all the zones it "sees"
	var pvcList []*v1.PersistentVolumeClaim
	c := f.ClientSet
	ns := f.Namespace.Name

	for index := 1; index <= zoneCount+1; index++ {
		pvc := newNamedDefaultClaim(ns, index)
		pvc, err = framework.CreatePVC(c, ns, pvc)
		Expect(err).NotTo(HaveOccurred())
		pvcList = append(pvcList, pvc)

		// Defer the cleanup
		defer func() {
			framework.Logf("deleting claim %q/%q", pvc.Namespace, pvc.Name)
			err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, nil)
			if err != nil {
				framework.Failf("Error deleting claim %q. Error: %v", pvc.Name, err)
			}
		}()
	}

	// Wait for all claims bound
	for _, claim := range pvcList {
		err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, claim.Namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred())
	}

	pvZones := sets.NewString()
	By("Checking that PDs have been provisioned in only the expected zones")
	for _, claim := range pvcList {
		// Get a new copy of the claim to have all fields populated
		claim, err = c.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		// Get the related PV
		pv, err := c.CoreV1().PersistentVolumes().Get(claim.Spec.VolumeName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		pvZone, ok := pv.ObjectMeta.Labels[kubeletapis.LabelZoneFailureDomain]
		Expect(ok).To(BeTrue(), "PV has no LabelZone to be found")
		pvZones.Insert(pvZone)
	}
	Expect(pvZones.Equal(expectedZones)).To(BeTrue(), fmt.Sprintf("PDs provisioned in unwanted zones. We want zones: %v, got: %v", expectedZones, pvZones))
}

type staticPVTestConfig struct {
	pvSource *v1.PersistentVolumeSource
	pv       *v1.PersistentVolume
	pvc      *v1.PersistentVolumeClaim
	pod      *v1.Pod
}

// Check that the pods using statically created PVs get scheduled to the same zone that the PV is in.
func PodsUseStaticPVsOrFail(f *framework.Framework, podCount int, image string) {
	var err error
	c := f.ClientSet
	ns := f.Namespace.Name

	zones, err := framework.GetClusterZones(c)
	Expect(err).NotTo(HaveOccurred())
	zonelist := zones.List()
	By("Creating static PVs across zones")
	configs := make([]*staticPVTestConfig, podCount)
	for i := range configs {
		configs[i] = &staticPVTestConfig{}
	}

	defer func() {
		By("Cleaning up pods and PVs")
		for _, config := range configs {
			framework.DeletePodOrFail(c, ns, config.pod.Name)
		}
		for _, config := range configs {
			framework.WaitForPodNoLongerRunningInNamespace(c, config.pod.Name, ns)
			framework.PVPVCCleanup(c, ns, config.pv, config.pvc)
			err = framework.DeletePVSource(config.pvSource)
			Expect(err).NotTo(HaveOccurred())
		}
	}()

	for i, config := range configs {
		zone := zonelist[i%len(zones)]
		config.pvSource, err = framework.CreatePVSource(zone)
		Expect(err).NotTo(HaveOccurred())

		pvConfig := framework.PersistentVolumeConfig{
			NamePrefix: "multizone-pv",
			PVSource:   *config.pvSource,
			Prebind:    nil,
		}
		className := ""
		pvcConfig := framework.PersistentVolumeClaimConfig{StorageClassName: &className}

		config.pv, config.pvc, err = framework.CreatePVPVC(c, pvConfig, pvcConfig, ns, true)
		Expect(err).NotTo(HaveOccurred())
	}

	By("Waiting for all PVCs to be bound")
	for _, config := range configs {
		framework.WaitOnPVandPVC(c, ns, config.pv, config.pvc)
	}

	By("Creating pods for each static PV")
	for _, config := range configs {
		podConfig := framework.MakePod(ns, nil, []*v1.PersistentVolumeClaim{config.pvc}, false, "")
		config.pod, err = c.CoreV1().Pods(ns).Create(podConfig)
		Expect(err).NotTo(HaveOccurred())
	}

	By("Waiting for all pods to be running")
	for _, config := range configs {
		err = framework.WaitForPodRunningInNamespace(c, config.pod)
		Expect(err).NotTo(HaveOccurred())
	}
}

func newNamedDefaultClaim(ns string, index int) *v1.PersistentVolumeClaim {
	claim := v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pvc-" + strconv.Itoa(index),
			Namespace: ns,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("1Gi"),
				},
			},
		},
	}

	return &claim
}
