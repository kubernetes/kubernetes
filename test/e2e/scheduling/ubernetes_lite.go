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

package scheduling

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Multi-AZ Clusters", func() {
	f := framework.NewDefaultFramework("multi-az")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	var zoneCount int
	var err error
	var zoneNames sets.Set[string]
	ginkgo.BeforeEach(func(ctx context.Context) {
		cs := f.ClientSet

		if zoneCount <= 0 {
			zoneNames, err = e2enode.GetSchedulableClusterZones(ctx, cs)
			framework.ExpectNoError(err)
			zoneCount = len(zoneNames)
		}
		ginkgo.By(fmt.Sprintf("Checking for multi-zone cluster. Schedulable zone count = %d", zoneCount))
		msg := fmt.Sprintf("Schedulable zone count is %d, only run for multi-zone clusters, skipping test", zoneCount)
		e2eskipper.SkipUnlessAtLeast(zoneCount, 2, msg)
		// TODO: SkipUnlessDefaultScheduler() // Non-default schedulers might not spread

		e2enode.WaitForTotalHealthy(ctx, cs, time.Minute)
		nodeList, err := e2enode.GetReadySchedulableNodes(ctx, cs)
		framework.ExpectNoError(err)

		// make the nodes have balanced cpu,mem usage
		err = createBalancedPodForNodes(ctx, f, cs, f.Namespace.Name, nodeList.Items, podRequestedResource, 0.0)
		framework.ExpectNoError(err)
	})
	f.It("should spread the pods of a service across zones", f.WithSerial(), func(ctx context.Context) {
		SpreadServiceOrFail(ctx, f, 5*zoneCount, zoneNames, imageutils.GetPauseImageName())
	})

	f.It("should spread the pods of a replication controller across zones", f.WithSerial(), func(ctx context.Context) {
		SpreadRCOrFail(ctx, f, int32(5*zoneCount), zoneNames, imageutils.GetE2EImage(imageutils.Agnhost), []string{"serve-hostname"})
	})
})

// SpreadServiceOrFail check that the pods comprising a service
// get spread evenly across available zones
func SpreadServiceOrFail(ctx context.Context, f *framework.Framework, replicaCount int, zoneNames sets.Set[string], image string) {
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
				TargetPort: intstr.FromInt32(80),
			}},
		},
	}
	_, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(ctx, serviceSpec, metav1.CreateOptions{})
	framework.ExpectNoError(err)

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
					Image: image,
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
	pods, err := e2epod.WaitForPodsWithLabelScheduled(ctx, f.ClientSet, f.Namespace.Name, selector)
	framework.ExpectNoError(err)

	// Now make sure they're spread across zones
	checkZoneSpreading(ctx, f.ClientSet, pods, sets.List(zoneNames))
}

// Find the name of the zone in which a Node is running
func getZoneNameForNode(node v1.Node) (string, error) {
	if z, ok := node.Labels[v1.LabelFailureDomainBetaZone]; ok {
		return z, nil
	} else if z, ok := node.Labels[v1.LabelTopologyZone]; ok {
		return z, nil
	}
	return "", fmt.Errorf("node %s doesn't have zone label %s or %s",
		node.Name, v1.LabelFailureDomainBetaZone, v1.LabelTopologyZone)
}

// Find the name of the zone in which the pod is scheduled
func getZoneNameForPod(ctx context.Context, c clientset.Interface, pod v1.Pod) (string, error) {
	ginkgo.By(fmt.Sprintf("Getting zone name for pod %s, on node %s", pod.Name, pod.Spec.NodeName))
	node, err := c.CoreV1().Nodes().Get(ctx, pod.Spec.NodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	return getZoneNameForNode(*node)
}

// Determine whether a set of pods are approximately evenly spread
// across a given set of zones
func checkZoneSpreading(ctx context.Context, c clientset.Interface, pods *v1.PodList, zoneNames []string) {
	podsPerZone := make(map[string]int)
	for _, zoneName := range zoneNames {
		podsPerZone[zoneName] = 0
	}
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp != nil {
			continue
		}
		zoneName, err := getZoneNameForPod(ctx, c, pod)
		framework.ExpectNoError(err)
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
	gomega.Expect(maxPodsPerZone-minPodsPerZone).To(gomega.BeNumerically("~", 0, 2),
		"Pods were not evenly spread across zones.  %d in one zone and %d in another zone",
		minPodsPerZone, maxPodsPerZone)
}

// SpreadRCOrFail Check that the pods comprising a replication
// controller get spread evenly across available zones
func SpreadRCOrFail(ctx context.Context, f *framework.Framework, replicaCount int32, zoneNames sets.Set[string], image string, args []string) {
	name := "ubelite-spread-rc-" + string(uuid.NewUUID())
	rcLabels := map[string]string{"name": name}

	ginkgo.By(fmt.Sprintf("Creating replication controller %s", name))
	controller, err := f.ClientSet.CoreV1().ReplicationControllers(f.Namespace.Name).Create(ctx, &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: f.Namespace.Name,
			Name:      name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &replicaCount,
			Selector: rcLabels,
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: rcLabels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  name,
							Image: image,
							Args:  args,
							Ports: []v1.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	// Cleanup the replication controller when we are done.
	defer func() {
		// Resize the replication controller to zero to get rid of pods.
		if err := e2erc.DeleteRCAndWaitForGC(ctx, f.ClientSet, f.Namespace.Name, controller.Name); err != nil {
			framework.Logf("Failed to cleanup replication controller %v: %v.", controller.Name, err)
		}
	}()
	// List the pods, making sure we observe all the replicas.
	selector := labels.SelectorFromSet(rcLabels)
	_, err = e2epod.PodsCreatedByLabel(ctx, f.ClientSet, f.Namespace.Name, name, replicaCount, selector)
	framework.ExpectNoError(err)

	// Wait for all of them to be scheduled
	ginkgo.By(fmt.Sprintf("Waiting for %d replicas of %s to be scheduled.  Selector: %v", replicaCount, name, selector))
	pods, err := e2epod.WaitForPodsWithLabelScheduled(ctx, f.ClientSet, f.Namespace.Name, selector)
	framework.ExpectNoError(err)

	// Now make sure they're spread across zones
	checkZoneSpreading(ctx, f.ClientSet, pods, sets.List(zoneNames))
}
