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
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	corev1typed "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
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
		t.Fatalf("error in creating clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	dsInformers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "daemonset-informers")), resyncPeriod)
	metrics.UnregisterMetricAndUntrackRateLimiterUsage("daemon_controller")
	dc, err := daemon.NewDaemonSetsController(
		dsInformers.Apps().V1().DaemonSets(),
		dsInformers.Apps().V1().ControllerRevisions(),
		dsInformers.Core().V1().Pods(),
		dsInformers.Core().V1().Nodes(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "daemonset-controller")),
	)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}

	return server, closeFn, dc, dsInformers, clientSet
}

func TestOneNodeDaemonLaunchesPod(t *testing.T) {
	for _, strategy := range updateStrategies() {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("one-node-daemonset-test", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
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
			t.Fatalf("failed to create DaemonSet: %v", err)
		}
		_, err = nodeClient.Create(newNode("single-node", nil))
		if err != nil {
			t.Fatalf("failed to create node: %v", err)
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

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
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
			t.Fatalf("failed to create DaemonSet: %v", err)
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

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
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
			t.Fatalf("failed to create DaemonSet: %v", err)
		}
		node := newNode("single-node", nil)
		node.Status.Conditions = []v1.NodeCondition{
			{Type: v1.NodeReady, Status: v1.ConditionFalse},
		}
		_, err = nodeClient.Create(node)
		if err != nil {
			t.Fatalf("failed to create node: %v", err)
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

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
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
			t.Fatalf("failed to create DaemonSet: %v", err)
		}
		node := newNode("node-with-limited-memory", nil)
		node.Status.Allocatable = allocatableResources("100M", "200m")
		_, err = nodeClient.Create(node)
		if err != nil {
			t.Fatalf("failed to create node: %v", err)
		}

		validateFailedPlacementEvent(eventClient, t)

		close(stopCh)
	}
}

// A RollingUpdate DaemonSet should adopt existing pods when it is created
func TestRollingUpdateDaemonSetExistingPodAdoption(t *testing.T) {
	server, closeFn, dc, informers, clientset := setup(t)
	defer closeFn()

	tearDownFn := setupGC(t, server)
	defer tearDownFn()

	ns := framework.CreateTestingNamespace("rolling-update-daemonset-existing-pod-adoption-test", server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	dsClient := clientset.AppsV1().DaemonSets(ns.Name)
	podClient := clientset.CoreV1().Pods(ns.Name)
	nodeClient := clientset.CoreV1().Nodes()
	podInformer := informers.Core().V1().Pods().Informer()
	controllerRevisionClient := clientset.AppsV1().ControllerRevisions(ns.Name)
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go dc.Run(5, stopCh)

	// Step 1: create a RollingUpdate DaemonSet
	dsName := "daemonset"
	ds := newDaemonSet(dsName, ns.Name)
	ds.Spec.UpdateStrategy = *newRollbackStrategy()
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

	// Validate everything works correctly (include marking daemon pod as ready)
	validateDaemonSetPodsAndMarkReady(podClient, podInformer, 1, t)
	validateDaemonSetStatus(dsClient, ds.Name, ds.Namespace, 1, t)

	// Step 2: delete daemonset and orphan its pods
	deleteDaemonSetAndOrphanPods(dsClient, podClient, controllerRevisionClient, podInformer, ds, t)

	// Step 3: create 2rd daemonset to adopt the pods (no restart) as long as template matches
	dsName2 := "daemonset-adopt-template-matches"
	ds2 := newDaemonSet(dsName2, ns.Name)
	ds2.Spec.UpdateStrategy = *newRollbackStrategy()
	ds2, err = dsClient.Create(ds2)
	if err != nil {
		t.Fatalf("failed to create daemonset %q: %v", dsName2, err)
	}
	if !apiequality.Semantic.DeepEqual(ds2.Spec.Template, ds.Spec.Template) {
		t.Fatalf(".spec.template of new daemonset %q and old daemonset %q are not the same", dsName2, dsName)
	}

	// Wait for pods and history to be adopted by 2nd daemonset
	waitDaemonSetAdoption(podClient, controllerRevisionClient, podInformer, ds2, ds.Name, t)
	validateDaemonSetStatus(dsClient, ds2.Name, ds2.Namespace, 1, t)
}
