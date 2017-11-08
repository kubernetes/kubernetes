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
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	corev1typed "k8s.io/client-go/kubernetes/typed/core/v1"
	extensionsv1beta1typed "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/util/metrics"
	"k8s.io/kubernetes/test/integration/framework"
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
					Containers: []v1.Container{{Name: "foo", Image: "bar"}},
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
				Name:  "foo",
				Image: "bar",
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
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
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
	if err := wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) {
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
	if err := wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) {
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
