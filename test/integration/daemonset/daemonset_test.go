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

package daemonset

import (
	"fmt"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	corev1typed "k8s.io/client-go/kubernetes/typed/core/v1"
	typedv1 "k8s.io/client-go/kubernetes/typed/core/v1"
	extensionsv1beta1typed "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/util/metrics"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	pollInterval = 100 * time.Millisecond
	pollTimeout  = 60 * time.Second

	fakeContainerName = "fake-name"
	fakeImage         = "fakeimage"
)

func setup(t *testing.T) (*httptest.Server, framework.CloseFunc, *daemon.DaemonSetsController, informers.SharedInformerFactory, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, server, closeFn := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: server.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error in creating clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "daemonset-informers")), resyncPeriod)
	metrics.UnregisterMetricAndUntrackRateLimiterUsage("daemon_controller")
	dc, err := daemon.NewDaemonSetsController(
		informers.Extensions().V1beta1().DaemonSets(),
		informers.Apps().V1beta1().ControllerRevisions(),
		informers.Core().V1().Pods(),
		informers.Core().V1().Nodes(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "daemonset-controller")),
	)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}

	return server, closeFn, dc, informers, clientSet
}

func testLabels() map[string]string {
	return map[string]string{"name": "test"}
}

func newDaemonSet(name, namespace string) *v1beta1.DaemonSet {
	two := int32(2)
	return &v1beta1.DaemonSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "DaemonSet",
			APIVersion: "extensions/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: v1beta1.DaemonSetSpec{
			RevisionHistoryLimit: &two,
			Selector:             &metav1.LabelSelector{MatchLabels: testLabels()},
			UpdateStrategy: v1beta1.DaemonSetUpdateStrategy{
				Type: v1beta1.OnDeleteDaemonSetStrategyType,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: testLabels(),
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: fakeContainerName, Image: fakeImage}},
				},
			},
		},
	}
}

func newRollbackStrategy() *v1beta1.DaemonSetUpdateStrategy {
	one := intstr.FromInt(1)
	return &v1beta1.DaemonSetUpdateStrategy{
		Type:          v1beta1.RollingUpdateDaemonSetStrategyType,
		RollingUpdate: &v1beta1.RollingUpdateDaemonSet{MaxUnavailable: &one},
	}
}

func newOnDeleteStrategy() *v1beta1.DaemonSetUpdateStrategy {
	return &v1beta1.DaemonSetUpdateStrategy{
		Type: v1beta1.OnDeleteDaemonSetStrategyType,
	}
}

func updateStrategies() []*v1beta1.DaemonSetUpdateStrategy {
	return []*v1beta1.DaemonSetUpdateStrategy{newOnDeleteStrategy(), newRollbackStrategy()}
}

func allocatableResources(memory, cpu string) v1.ResourceList {
	return v1.ResourceList{
		v1.ResourceMemory: resource.MustParse(memory),
		v1.ResourceCPU:    resource.MustParse(cpu),
		v1.ResourcePods:   resource.MustParse("100"),
	}
}

func resourcePodSpec(nodeName, memory, cpu string) v1.PodSpec {
	return v1.PodSpec{
		NodeName: nodeName,
		Containers: []v1.Container{
			{
				Name:  fakeContainerName,
				Image: fakeImage,
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse(memory),
						v1.ResourceCPU:    resource.MustParse(cpu),
					},
				},
			},
		},
	}
}

func newNode(name string, label map[string]string) *v1.Node {
	return &v1.Node{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Node",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Labels:    label,
			Namespace: metav1.NamespaceDefault,
		},
		Status: v1.NodeStatus{
			Conditions:  []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
			Allocatable: v1.ResourceList{v1.ResourcePods: resource.MustParse("100")},
		},
	}
}

func addNodes(nodeClient corev1typed.NodeInterface, startIndex, numNodes int, label map[string]string, t *testing.T) {
	for i := startIndex; i < startIndex+numNodes; i++ {
		_, err := nodeClient.Create(newNode(fmt.Sprintf("node-%d", i), label))
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}
	}
}

func validateDaemonSetPodsAndMarkReady(
	podClient corev1typed.PodInterface,
	podInformer cache.SharedIndexInformer,
	numberPods int,
	t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		if len(objects) != numberPods {
			return false, nil
		}

		for _, object := range objects {
			pod := object.(*v1.Pod)

			ownerReferences := pod.ObjectMeta.OwnerReferences
			if len(ownerReferences) != 1 {
				return false, fmt.Errorf("Pod %s has %d OwnerReferences, expected only 1", pod.Name, len(ownerReferences))
			}
			controllerRef := ownerReferences[0]
			if got, want := controllerRef.APIVersion, "extensions/v1beta1"; got != want {
				t.Errorf("controllerRef.APIVersion = %q, want %q", got, want)
			}
			if got, want := controllerRef.Kind, "DaemonSet"; got != want {
				t.Errorf("controllerRef.Kind = %q, want %q", got, want)
			}
			if controllerRef.Controller == nil || *controllerRef.Controller != true {
				t.Errorf("controllerRef.Controller is not set to true")
			}

			if !podutil.IsPodReady(pod) {
				podCopy := pod.DeepCopy()
				podCopy.Status = v1.PodStatus{
					Phase:      v1.PodRunning,
					Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}},
				}
				_, err := podClient.UpdateStatus(podCopy)
				if err != nil {
					return false, err
				}
			}
		}

		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func validateDaemonSetStatus(
	dsClient extensionsv1beta1typed.DaemonSetInterface,
	dsName string,
	dsNamespace string,
	expectedNumberReady int32,
	t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		ds, err := dsClient.Get(dsName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return ds.Status.NumberReady == expectedNumberReady, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func validateFailedPlacementEvent(eventClient corev1typed.EventInterface, t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		eventList, err := eventClient.List(metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		if len(eventList.Items) == 0 {
			return false, nil
		}
		if len(eventList.Items) > 1 {
			t.Errorf("Expected 1 event got %d", len(eventList.Items))
		}
		event := eventList.Items[0]
		if event.Type != v1.EventTypeWarning {
			t.Errorf("Event type expected %s got %s", v1.EventTypeWarning, event.Type)
		}
		if event.Reason != daemon.FailedPlacementReason {
			t.Errorf("Event reason expected %s got %s", daemon.FailedPlacementReason, event.Reason)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestOneNodeDaemonLaunchesPod(t *testing.T) {
	for _, strategy := range updateStrategies() {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("one-node-daemonset-test", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.ExtensionsV1beta1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()
		stopCh := make(chan struct{})
		informers.Start(stopCh)
		go dc.Run(5, stopCh)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(ds)
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}
		_, err = nodeClient.Create(newNode("single-node", nil))
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 1, t)
		validateDaemonSetStatus(dsClient, ds.Name, ds.Namespace, 1, t)

		close(stopCh)
	}
}

func TestSimpleDaemonSetLaunchesPods(t *testing.T) {
	for _, strategy := range updateStrategies() {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("simple-daemonset-test", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.ExtensionsV1beta1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()
		stopCh := make(chan struct{})
		informers.Start(stopCh)
		go dc.Run(5, stopCh)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(ds)
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}
		addNodes(nodeClient, 0, 5, nil, t)

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 5, t)
		validateDaemonSetStatus(dsClient, ds.Name, ds.Namespace, 5, t)

		close(stopCh)
	}
}

func TestNotReadyNodeDaemonDoesLaunchPod(t *testing.T) {
	for _, strategy := range updateStrategies() {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("simple-daemonset-test", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.ExtensionsV1beta1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()
		stopCh := make(chan struct{})
		informers.Start(stopCh)
		go dc.Run(5, stopCh)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(ds)
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}
		node := newNode("single-node", nil)
		node.Status.Conditions = []v1.NodeCondition{
			{Type: v1.NodeReady, Status: v1.ConditionFalse},
		}
		_, err = nodeClient.Create(node)
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 1, t)
		validateDaemonSetStatus(dsClient, ds.Name, ds.Namespace, 1, t)

		close(stopCh)
	}
}

func TestInsufficientCapacityNodeDaemonDoesNotLaunchPod(t *testing.T) {
	for _, strategy := range updateStrategies() {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("insufficient-capacity", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.ExtensionsV1beta1().DaemonSets(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		eventClient := corev1typed.New(clientset.CoreV1().RESTClient()).Events(ns.Namespace)
		stopCh := make(chan struct{})
		informers.Start(stopCh)
		go dc.Run(5, stopCh)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.Template.Spec = resourcePodSpec("node-with-limited-memory", "120M", "75m")
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(ds)
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}
		node := newNode("node-with-limited-memory", nil)
		node.Status.Allocatable = allocatableResources("100M", "200m")
		_, err = nodeClient.Create(node)
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		validateFailedPlacementEvent(eventClient, t)

		close(stopCh)
	}
}

func getPods(t *testing.T, podClient corev1typed.PodInterface, labelMap map[string]string) *v1.PodList {
	podSelector := labels.Set(labelMap).AsSelector()
	options := metav1.ListOptions{LabelSelector: podSelector.String()}
	pods, err := podClient.List(options)
	if err != nil {
		t.Fatalf("failed obtaining a list of pods that match the pod labels %v: %v", labelMap, err)
	}
	return pods
}

func updatePod(t *testing.T, podClient typedv1.PodInterface, podName string, updateFunc func(*v1.Pod)) *v1.Pod {
	var pod *v1.Pod
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newPod, err := podClient.Get(podName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateFunc(newPod)
		pod, err = podClient.Update(newPod)
		return err
	}); err != nil {
		t.Fatalf("failed to update status of pod %q: %v", pod.Name, err)
	}
	return pod
}

func updatePodStatus(t *testing.T, podClient typedv1.PodInterface, podName string, updateStatusFunc func(*v1.Pod)) *v1.Pod {
	var pod *v1.Pod
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newPod, err := podClient.Get(podName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateStatusFunc(newPod)
		pod, err = podClient.UpdateStatus(newPod)
		return err
	}); err != nil {
		t.Fatalf("failed to update status of pod %q: %v", pod.Name, err)
	}
	return pod
}

// waitForAllPodsPending waits for all pods to be in pending phase
func waitForAllPodsPending(podClient corev1typed.PodInterface, t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		pods := getPods(t, podClient, testLabels())
		for _, pod := range pods.Items {
			if pod.Status.Phase != v1.PodPending {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatalf("failed waiting for all pods to be in pending phase: %v", err)
	}
}

// checkAtLeastOnePodWithNewImage checks if there is at least one pod with specific image
func checkAtLeastOnePodWithImage(podClient corev1typed.PodInterface, image string, t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		pods := getPods(t, podClient, testLabels())
		for _, pod := range pods.Items {
			if pod.Spec.Containers[0].Image == image {
				return true, nil
			}
		}
		return false, nil
	}); err != nil {
		t.Fatal(err)
	}
}

// canScheduleOnNode checks if a given DaemonSet can schedule pods on the given node
func canScheduleOnNode(node v1.Node, ds *v1beta1.DaemonSet) bool {
	newPod := daemon.NewPod(ds, node.Name)
	nodeInfo := schedulercache.NewNodeInfo()
	nodeInfo.SetNode(&node)
	fit, _, err := daemon.Predicates(newPod, nodeInfo)
	if err != nil {
		return false
	}
	return fit
}

// schedulableNodes returns name list of schedulable nodes for a given DaemonSet
func schedulableNodes(c clientset.Interface, ds *v1beta1.DaemonSet, t *testing.T) []string {
	nodeList, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("failed to get a list of schedulable nodes for daemonset %q: %v", ds.Name, err)
	}
	nodeNames := make([]string, 0)
	for _, node := range nodeList.Items {
		if !canScheduleOnNode(node, ds) {
			continue
		}
		nodeNames = append(nodeNames, node.Name)
	}
	return nodeNames
}

// checkPodsImageAndAvailability checks whether pods have desired image and
// maxUnavailable requirement is met after a rolling update operation completes
func checkPodsImageAndAvailability(c clientset.Interface, ds *v1beta1.DaemonSet, image string, maxUnavailable int, podClient corev1typed.PodInterface, t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		pods := getPods(t, podClient, testLabels())
		unavailablePods := 0
		nodesToUpdatedPodCount := make(map[string]int)
		for _, pod := range pods.Items {
			if !metav1.IsControlledBy(&pod, ds) {
				continue
			}
			podImage := pod.Spec.Containers[0].Image
			if podImage == image {
				nodesToUpdatedPodCount[pod.Spec.NodeName]++
			}
			if !podutil.IsPodAvailable(&pod, ds.Spec.MinReadySeconds, metav1.Now()) {
				unavailablePods++
			}
		}
		if unavailablePods > maxUnavailable {
			return false, fmt.Errorf("number of unavailable pods %d is greater than maxUnavailable %d", unavailablePods, maxUnavailable)
		}
		// Make sure every pod on the node has been updated
		nodeNames := schedulableNodes(c, ds, t)
		for _, node := range nodeNames {
			if nodesToUpdatedPodCount[node] == 0 {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

// A DaemonSet should retry creating failed daemon pods
func TestRetryCreatingFailedPod(t *testing.T) {
	for _, strategy := range updateStrategies() {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("retry-creating-failed-pod-test", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.ExtensionsV1beta1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()
		stopCh := make(chan struct{})
		defer close(stopCh)
		informers.Start(stopCh)
		go dc.Run(5, stopCh)

		dsName := "daemonset"
		ds := newDaemonSet(dsName, ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		ds, err := dsClient.Create(ds)
		if err != nil {
			t.Fatalf("failed to create daemonset %q: %v", dsName, err)
		}

		nodeName := "single-node"
		node := newNode(nodeName, nil)
		_, err = nodeClient.Create(node)
		if err != nil {
			t.Fatalf("failed to create node %q: %v", nodeName, err)
		}

		// Validate everything works correctly, include marking pods
		// as ready and ensure they launch on every node
		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 1, t)
		validateDaemonSetStatus(dsClient, ds.Name, ds.Namespace, 1, t)

		// Set the pod's phase to "Failed"
		pods := getPods(t, podClient, testLabels())
		updatePodStatus(t, podClient, pods.Items[0].Name, func(pod *v1.Pod) {
			pod.Status.Phase = v1.PodFailed
		})

		// Wait for the failed pod to be created again and in pending phase
		waitForAllPodsPending(podClient, t)
	}
}

// A RollingUpdate DaemonSet should rollback without unnecessary restarts
func TestRollbackWithoutUnnecessaryRestarts(t *testing.T) {
	server, closeFn, dc, informers, clientset := setup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("rollback-without-unnecessary-restarts-test", server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	dsClient := clientset.ExtensionsV1beta1().DaemonSets(ns.Name)
	podClient := clientset.CoreV1().Pods(ns.Name)
	nodeClient := clientset.CoreV1().Nodes()
	podInformer := informers.Core().V1().Pods().Informer()
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go dc.Run(5, stopCh)

	// Create a rolling update daemonset
	dsName := "daemonset"
	ds := newDaemonSet(dsName, ns.Name)
	ds.Spec.UpdateStrategy = *newRollbackStrategy()
	oldImage := "image"
	ds.Spec.Template.Spec.Containers[0].Image = oldImage
	ds, err := dsClient.Create(ds)
	if err != nil {
		t.Fatalf("failed to create daemonset %q: %v", dsName, err)
	}

	// Create 2 nodes
	addNodes(nodeClient, 0, 2, nil, t)

	// Validate everything works correctly, include marking pods
	// as ready and ensure they launch on every node
	validateDaemonSetPodsAndMarkReady(podClient, podInformer, 2, t)
	validateDaemonSetStatus(dsClient, ds.Name, ds.Namespace, 2, t)

	// Manually update image of first pod to the new image
	// Note: daemonset image update does not propagate to its pods
	newImage := "non-existent-image"
	pods := getPods(t, podClient, testLabels())
	pod := updatePod(t, podClient, pods.Items[0].Name, func(pod *v1.Pod) {
		pod.Spec.Containers[0].Image = newImage
	})

	// Ensure we are in the middle of a rollout by checking whether
	// there is at least one pod with new image
	checkAtLeastOnePodWithImage(podClient, newImage, t)
	var existingPods, newPods []*v1.Pod
	pods = getPods(t, podClient, testLabels())
	for i := range pods.Items {
		pod := pods.Items[i]
		image := pod.Spec.Containers[0].Image
		switch image {
		case oldImage:
			existingPods = append(existingPods, &pod)
		case newImage:
			newPods = append(newPods, &pod)
		default:
			t.Fatalf("unexpected pod image %q found", image)
		}
	}
	if len(existingPods) == 0 {
		t.Fatalf("expected at least one pod with old image exists, but found none")
	}
	if len(newPods) == 0 {
		t.Fatalf("expected at least one pod with new image exists, but found none")
	}

	// Manually update image of first pod back to old image
	updatePod(t, podClient, pod.Name, func(pod *v1.Pod) {
		pod.Spec.Containers[0].Image = oldImage
	})

	// Ensure the rollback is complete
	checkPodsImageAndAvailability(clientset, ds, oldImage, 1, podClient, t)

	// Ensure pods are not restarted during rollback by comparing current and old pods
	pods = getPods(t, podClient, testLabels())
	m := map[string]bool{}
	for _, pod := range pods.Items {
		m[pod.Name] = true
	}
	for _, pod := range existingPods {
		if _, ok := m[pod.Name]; !ok {
			t.Fatalf("unexpected pod %q found due to pod restart", pod.Name)
		}
	}
}
