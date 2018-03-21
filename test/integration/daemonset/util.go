/*
Copyright 2018 The Kubernetes Authors.

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
	"strings"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/discovery"
	cacheddiscovery "k8s.io/client-go/discovery/cached"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	appstyped "k8s.io/client-go/kubernetes/typed/apps/v1"
	corev1typed "k8s.io/client-go/kubernetes/typed/core/v1"
	typedv1 "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
)

const (
	pollInterval = 100 * time.Millisecond
	pollTimeout  = 60 * time.Second
)

func testLabels() map[string]string {
	return map[string]string{"name": "test"}
}

func newDaemonSet(name, namespace string) *apps.DaemonSet {
	two := int32(2)
	return &apps.DaemonSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "DaemonSet",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: apps.DaemonSetSpec{
			RevisionHistoryLimit: &two,
			Selector:             &metav1.LabelSelector{MatchLabels: testLabels()},
			UpdateStrategy: apps.DaemonSetUpdateStrategy{
				Type: apps.OnDeleteDaemonSetStrategyType,
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

func newRollbackStrategy() *apps.DaemonSetUpdateStrategy {
	one := intstr.FromInt(1)
	return &apps.DaemonSetUpdateStrategy{
		Type:          apps.RollingUpdateDaemonSetStrategyType,
		RollingUpdate: &apps.RollingUpdateDaemonSet{MaxUnavailable: &one},
	}
}

func newOnDeleteStrategy() *apps.DaemonSetUpdateStrategy {
	return &apps.DaemonSetUpdateStrategy{
		Type: apps.OnDeleteDaemonSetStrategyType,
	}
}

func updateStrategies() []*apps.DaemonSetUpdateStrategy {
	return []*apps.DaemonSetUpdateStrategy{newOnDeleteStrategy(), newRollbackStrategy()}
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
			t.Fatalf("failed to create node: %v", err)
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
				return false, fmt.Errorf("pod %s has %d OwnerReferences, expected only 1", pod.Name, len(ownerReferences))
			}
			controllerRef := ownerReferences[0]
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
	dsClient appstyped.DaemonSetInterface,
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
			t.Errorf("expected 1 event got %d", len(eventList.Items))
		}
		event := eventList.Items[0]
		if event.Type != v1.EventTypeWarning {
			t.Errorf("event type expected %s got %s", v1.EventTypeWarning, event.Type)
		}
		if event.Reason != daemon.FailedPlacementReason {
			t.Errorf("event reason expected %s got %s", daemon.FailedPlacementReason, event.Reason)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
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

func boolptr(b bool) *bool { return &b }

func checkDaemonSetPodsOrphaned(podClient corev1typed.PodInterface, t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		pods := getPods(t, podClient, testLabels())
		for _, pod := range pods.Items {
			// This pod is orphaned only when its controllerRef is nil
			if controllerRef := metav1.GetControllerOf(&pod); controllerRef != nil {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func checkDaemonSetPodsAdopted(podClient corev1typed.PodInterface, podInformer cache.SharedIndexInformer, dsUID types.UID, t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		pods := podInformer.GetIndexer().List()
		for _, object := range pods {
			pod := object.(*v1.Pod)
			// This pod is adopted only when its controller ref is update
			if controllerRef := metav1.GetControllerOf(pod); controllerRef == nil || controllerRef.UID != dsUID {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func listDaemonSetHistories(controllerRevisionClient appstyped.ControllerRevisionInterface, t *testing.T) *apps.ControllerRevisionList {
	selector := labels.Set(testLabels()).AsSelector()
	options := metav1.ListOptions{LabelSelector: selector.String()}
	historyList, err := controllerRevisionClient.List(options)
	if err != nil {
		t.Fatalf("failed to list daemonset histories: %v", err)
	}
	if len(historyList.Items) == 0 {
		t.Fatalf("failed to locate any daemonset history")
	}
	return historyList
}

func checkDaemonSetHistoryOrphaned(controllerRevisionClient appstyped.ControllerRevisionInterface, t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		histories := listDaemonSetHistories(controllerRevisionClient, t)
		for _, history := range histories.Items {
			// This history is orphaned only when its controllerRef is nil
			if controllerRef := metav1.GetControllerOf(&history); controllerRef != nil {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func checkDaemonSetHistoryAdopted(controllerRevisionClient appstyped.ControllerRevisionInterface, dsUID types.UID, t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		histories := listDaemonSetHistories(controllerRevisionClient, t)
		for _, history := range histories.Items {
			// This history is adopted only when its controller ref is update
			if controllerRef := metav1.GetControllerOf(&history); controllerRef == nil || controllerRef.UID != dsUID {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func checkDaemonSetDeleted(dsClient appstyped.DaemonSetInterface, dsName string, t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		_, err := dsClient.Get(dsName, metav1.GetOptions{})
		if !apierrs.IsNotFound(err) {
			return false, err
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
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

func updateHistory(t *testing.T, controllerRevisionClient appstyped.ControllerRevisionInterface, historyName string, updateFunc func(*apps.ControllerRevision)) *apps.ControllerRevision {
	var history *apps.ControllerRevision
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newHistory, err := controllerRevisionClient.Get(historyName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateFunc(newHistory)
		history, err = controllerRevisionClient.Update(newHistory)
		return err
	}); err != nil {
		t.Fatalf("failed to update status of history %q: %v", history.Name, err)
	}
	return history
}

// deleteDaemonSetAndOrphanPods deletes the given DaemonSet and orphans all its dependents.
// It also checks that all dependents are orphaned, and the DaemonSet is deleted.
func deleteDaemonSetAndOrphanPods(
	dsClient appstyped.DaemonSetInterface,
	podClient corev1typed.PodInterface,
	controllerRevisionClient appstyped.ControllerRevisionInterface,
	podInformer cache.SharedIndexInformer,
	ds *apps.DaemonSet,
	t *testing.T) {

	ds, err := dsClient.Get(ds.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get daemonset %q: %v", ds.Name, err)
	}

	deletePropagationOrphanPolicy := metav1.DeletePropagationOrphan
	deleteOptions := &metav1.DeleteOptions{
		PropagationPolicy: &deletePropagationOrphanPolicy,
		Preconditions:     metav1.NewUIDPreconditions(string(ds.UID)),
	}
	if err = dsClient.Delete(ds.Name, deleteOptions); err != nil {
		t.Fatalf("failed deleting daemonset %q: %v", ds.Name, err)
	}

	checkDaemonSetDeleted(dsClient, ds.Name, t)
	checkDaemonSetPodsOrphaned(podClient, t)
	checkDaemonSetHistoryOrphaned(controllerRevisionClient, t)
}

func checkDaemonSetPodsName(podInformer cache.SharedIndexInformer, podNamePrefix string, t *testing.T) {
	pods := podInformer.GetIndexer().List()
	for _, object := range pods {
		pod := object.(*v1.Pod)
		if !strings.HasPrefix(pod.Name, podNamePrefix) {
			t.Fatalf("expected pod %q has name prefix %q", pod.Name, podNamePrefix)
		}
	}
}

func waitDaemonSetAdoption(
	podClient corev1typed.PodInterface,
	controllerRevisionClient appstyped.ControllerRevisionInterface,
	podInformer cache.SharedIndexInformer,
	ds *apps.DaemonSet,
	podNamePrefix string,
	t *testing.T) {
	checkDaemonSetPodsAdopted(podClient, podInformer, ds.UID, t)
	checkDaemonSetHistoryAdopted(controllerRevisionClient, ds.UID, t)

	// Ensure no pod is re-created by checking their names
	checkDaemonSetPodsName(podInformer, podNamePrefix, t)
}

// setupGC starts an owner reference garbage collector for given test server.
// The function returns a tear down function to defer shutting down the GC.
// When deleting a DaemonSet to orphan its pods, the GC is used to remove
// finalizer from the DaemonSet to complete deletion of the controller.
func setupGC(t *testing.T, server *httptest.Server) func() {
	config := restclient.Config{Host: server.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("error creating clientset: %v", err)
	}

	discoveryClient := cacheddiscovery.NewMemCacheClient(clientSet.Discovery())
	restMapper := discovery.NewDeferredDiscoveryRESTMapper(discoveryClient, meta.InterfacesForUnstructured)
	restMapper.Reset()
	deletableResources := garbagecollector.GetDeletableResources(discoveryClient)
	config.ContentConfig = dynamic.ContentConfig()
	metaOnlyClientPool := dynamic.NewClientPool(&config, restMapper, dynamic.LegacyAPIPathResolverFunc)
	clientPool := dynamic.NewClientPool(&config, restMapper, dynamic.LegacyAPIPathResolverFunc)
	sharedInformers := informers.NewSharedInformerFactory(clientSet, 0)
	alwaysStarted := make(chan struct{})
	close(alwaysStarted)
	gc, err := garbagecollector.NewGarbageCollector(
		metaOnlyClientPool,
		clientPool,
		restMapper,
		deletableResources,
		garbagecollector.DefaultIgnoredResources(),
		sharedInformers,
		alwaysStarted,
	)
	if err != nil {
		t.Fatalf("failed to create garbage collector: %v", err)
	}

	stopCh := make(chan struct{})
	tearDown := func() {
		close(stopCh)
	}
	syncPeriod := 5 * time.Second
	startGC := func(workers int) {
		go gc.Run(workers, stopCh)
		go gc.Sync(clientSet.Discovery(), syncPeriod, stopCh)
	}

	startGC(5)

	return tearDown
}

func createNamespaceOrDie(name string, c clientset.Interface, t *testing.T) *v1.Namespace {
	ns := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}
	if _, err := c.CoreV1().Namespaces().Create(ns); err != nil {
		t.Fatalf("failed to create namespace: %v", err)
	}
	falseVar := false
	_, err := c.CoreV1().ServiceAccounts(ns.Name).Create(&v1.ServiceAccount{
		ObjectMeta:                   metav1.ObjectMeta{Name: "default"},
		AutomountServiceAccountToken: &falseVar,
	})
	if err != nil {
		t.Fatalf("failed to create service account: %v", err)
	}
	return ns
}
