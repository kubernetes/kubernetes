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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
)

var _ = framework.KubeDescribe("Multi-AZ Clusters", func() {
	f := framework.NewDefaultFramework("multi-az")
	var zoneCount int
	var err error
	image := "gcr.io/google_containers/serve_hostname:v1.4"
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke", "aws")
		if zoneCount <= 0 {
			zoneCount, err = getZoneCount(f.ClientSet)
			Expect(err).NotTo(HaveOccurred())
		}
		By(fmt.Sprintf("Checking for multi-zone cluster.  Zone count = %d", zoneCount))
		framework.SkipUnlessAtLeast(zoneCount, 2, "Zone count is %d, only run for multi-zone clusters, skipping test")
		// TODO: SkipUnlessDefaultScheduler() // Non-default schedulers might not spread
	})
	It("should spread the pods of a service across zones", func() {
		SpreadServiceOrFail(f, (2*zoneCount)+1, image)
	})

	It("should spread the pods of a replication controller across zones", func() {
		SpreadRCOrFail(f, int32((2*zoneCount)+1), image)
	})
})

// Check that the pods comprising a service get spread evenly across available zones
func SpreadServiceOrFail(f *framework.Framework, replicaCount int, image string) {
	// First create the service
	serviceName := "test-service"
	serviceSpec := &v1.Service{
		ObjectMeta: v1.ObjectMeta{
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
		ObjectMeta: v1.ObjectMeta{
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
		if key == metav1.LabelZoneFailureDomain {
			return value, nil
		}
	}
	return "", fmt.Errorf("Zone name for node %s not found. No label with key %s",
		node.Name, metav1.LabelZoneFailureDomain)
}

// Find the names of all zones in which we have nodes in this cluster.
func getZoneNames(c clientset.Interface) ([]string, error) {
	zoneNames := sets.NewString()
	nodes, err := c.Core().Nodes().List(v1.ListOptions{})
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
	node, err := c.Core().Nodes().Get(pod.Spec.NodeName)
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
		ObjectMeta: v1.ObjectMeta{
			Namespace: f.Namespace.Name,
			Name:      name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &replicaCount,
			Selector: map[string]string{
				"name": name,
			},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
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
