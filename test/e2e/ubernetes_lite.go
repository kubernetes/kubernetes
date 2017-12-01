/*
Copyright 2015 The Kubernetes Authors.

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
	"math"
	"strconv"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	compute "google.golang.org/api/compute/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
)

var _ = framework.KubeDescribe("Multi-AZ Clusters", func() {
	f := framework.NewDefaultFramework("multi-az")
	var zoneCount int
	var err error
	image := framework.ServeHostnameImage
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke", "aws")
		if zoneCount <= 0 {
			zoneCount, err = getZoneCount(f.ClientSet)
			Expect(err).NotTo(HaveOccurred())
		}
		By(fmt.Sprintf("Checking for multi-zone cluster.  Zone count = %d", zoneCount))
		msg := fmt.Sprintf("Zone count is %d, only run for multi-zone clusters, skipping test", zoneCount)
		framework.SkipUnlessAtLeast(zoneCount, 2, msg)
		// TODO: SkipUnlessDefaultScheduler() // Non-default schedulers might not spread
	})
	It("should spread the pods of a service across zones", func() {
		SpreadServiceOrFail(f, (2*zoneCount)+1, image)
	})

	It("should spread the pods of a replication controller across zones", func() {
		SpreadRCOrFail(f, int32((2*zoneCount)+1), image)
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
	framework.Logf("Expected zones: %v\n", expectedZones)

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
		resp, err := gceCloud.DeleteInstance(project, zone, name)
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("Compute deletion response: %v\n", resp)
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

// Check that the pods comprising a service get spread evenly across available zones
func SpreadServiceOrFail(f *framework.Framework, replicaCount int, image string) {
	// First create the service
	serviceName := "test-service"
	serviceSpec := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName,
			Namespace: f.Namespace.Name,
		},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{
				"service": serviceName,
			},
			Ports: []v1.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt(80),
			}},
		},
	}
	_, err := f.ClientSet.Core().Services(f.Namespace.Name).Create(serviceSpec)
	Expect(err).NotTo(HaveOccurred())

	// Now create some pods behind the service
	podSpec := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   serviceName,
			Labels: map[string]string{"service": serviceName},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "test",
					Image: framework.GetPauseImageName(f.ClientSet),
				},
			},
		},
	}

	// Caution: StartPods requires at least one pod to replicate.
	// Based on the callers, replicas is always positive number: zoneCount >= 0 implies (2*zoneCount)+1 > 0.
	// Thus, no need to test for it. Once the precondition changes to zero number of replicas,
	// test for replicaCount > 0. Otherwise, StartPods panics.
	framework.ExpectNoError(testutils.StartPods(f.ClientSet, replicaCount, f.Namespace.Name, serviceName, *podSpec, false, framework.Logf))

	// Wait for all of them to be scheduled
	selector := labels.SelectorFromSet(labels.Set(map[string]string{"service": serviceName}))
	pods, err := framework.WaitForPodsWithLabelScheduled(f.ClientSet, f.Namespace.Name, selector)
	Expect(err).NotTo(HaveOccurred())

	// Now make sure they're spread across zones
	zoneNames, err := getZoneNames(f.ClientSet)
	Expect(err).NotTo(HaveOccurred())
	Expect(checkZoneSpreading(f.ClientSet, pods, zoneNames)).To(Equal(true))
}

// Find the name of the zone in which a Node is running
func getZoneNameForNode(node v1.Node) (string, error) {
	for key, value := range node.Labels {
		if key == kubeletapis.LabelZoneFailureDomain {
			return value, nil
		}
	}
	return "", fmt.Errorf("Zone name for node %s not found. No label with key %s",
		node.Name, kubeletapis.LabelZoneFailureDomain)
}

// Find the names of all zones in which we have nodes in this cluster.
func getZoneNames(c clientset.Interface) ([]string, error) {
	zoneNames := sets.NewString()
	nodes, err := c.Core().Nodes().List(metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	for _, node := range nodes.Items {
		zoneName, err := getZoneNameForNode(node)
		Expect(err).NotTo(HaveOccurred())
		zoneNames.Insert(zoneName)
	}
	return zoneNames.List(), nil
}

// Return the number of zones in which we have nodes in this cluster.
func getZoneCount(c clientset.Interface) (int, error) {
	zoneNames, err := getZoneNames(c)
	if err != nil {
		return -1, err
	}
	return len(zoneNames), nil
}

// Find the name of the zone in which the pod is scheduled
func getZoneNameForPod(c clientset.Interface, pod v1.Pod) (string, error) {
	By(fmt.Sprintf("Getting zone name for pod %s, on node %s", pod.Name, pod.Spec.NodeName))
	node, err := c.Core().Nodes().Get(pod.Spec.NodeName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	return getZoneNameForNode(*node)
}

// Determine whether a set of pods are approximately evenly spread
// across a given set of zones
func checkZoneSpreading(c clientset.Interface, pods *v1.PodList, zoneNames []string) (bool, error) {
	podsPerZone := make(map[string]int)
	for _, zoneName := range zoneNames {
		podsPerZone[zoneName] = 0
	}
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp != nil {
			continue
		}
		zoneName, err := getZoneNameForPod(c, pod)
		Expect(err).NotTo(HaveOccurred())
		podsPerZone[zoneName] = podsPerZone[zoneName] + 1
	}
	minPodsPerZone := math.MaxInt32
	maxPodsPerZone := 0
	for _, podCount := range podsPerZone {
		if podCount < minPodsPerZone {
			minPodsPerZone = podCount
		}
		if podCount > maxPodsPerZone {
			maxPodsPerZone = podCount
		}
	}
	Expect(minPodsPerZone).To(BeNumerically("~", maxPodsPerZone, 1),
		"Pods were not evenly spread across zones.  %d in one zone and %d in another zone",
		minPodsPerZone, maxPodsPerZone)
	return true, nil
}

// Check that the pods comprising a replication controller get spread evenly across available zones
func SpreadRCOrFail(f *framework.Framework, replicaCount int32, image string) {
	name := "ubelite-spread-rc-" + string(uuid.NewUUID())
	By(fmt.Sprintf("Creating replication controller %s", name))
	controller, err := f.ClientSet.Core().ReplicationControllers(f.Namespace.Name).Create(&v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: f.Namespace.Name,
			Name:      name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &replicaCount,
			Selector: map[string]string{
				"name": name,
			},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": name},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  name,
							Image: image,
							Ports: []v1.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())
	// Cleanup the replication controller when we are done.
	defer func() {
		// Resize the replication controller to zero to get rid of pods.
		if err := framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, controller.Name); err != nil {
			framework.Logf("Failed to cleanup replication controller %v: %v.", controller.Name, err)
		}
	}()
	// List the pods, making sure we observe all the replicas.
	selector := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	pods, err := framework.PodsCreated(f.ClientSet, f.Namespace.Name, name, replicaCount)
	Expect(err).NotTo(HaveOccurred())

	// Wait for all of them to be scheduled
	By(fmt.Sprintf("Waiting for %d replicas of %s to be scheduled.  Selector: %v", replicaCount, name, selector))
	pods, err = framework.WaitForPodsWithLabelScheduled(f.ClientSet, f.Namespace.Name, selector)
	Expect(err).NotTo(HaveOccurred())

	// Now make sure they're spread across zones
	zoneNames, err := getZoneNames(f.ClientSet)
	Expect(err).NotTo(HaveOccurred())
	Expect(checkZoneSpreading(f.ClientSet, pods, zoneNames)).To(Equal(true))
}

type StaticPVTestConfig struct {
	pvSource *v1.PersistentVolumeSource
	pv       *v1.PersistentVolume
	pvc      *v1.PersistentVolumeClaim
	pod      *v1.Pod
}

// Check that the pods using statically created PVs get scheduled to the same zone that the PV is in.
func PodsUseStaticPVsOrFail(f *framework.Framework, podCount int, image string) {
	// TODO: add GKE after enabling admission plugin in GKE
	// TODO: add AWS
	framework.SkipUnlessProviderIs("gce")

	var err error
	c := f.ClientSet
	ns := f.Namespace.Name

	zones, err := getZoneNames(c)
	Expect(err).NotTo(HaveOccurred())

	By("Creating static PVs across zones")
	configs := make([]*StaticPVTestConfig, podCount)
	for i := range configs {
		configs[i] = &StaticPVTestConfig{}
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
		zone := zones[i%len(zones)]
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
		podConfig := framework.MakePod(ns, []*v1.PersistentVolumeClaim{config.pvc}, false, "")
		config.pod, err = c.Core().Pods(ns).Create(podConfig)
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
