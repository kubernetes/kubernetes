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

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	appstyped "k8s.io/client-go/kubernetes/typed/apps/v1"
	clientv1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corev1typed "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
	_ "k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
	"k8s.io/kubernetes/pkg/scheduler/factory"
	"k8s.io/kubernetes/pkg/util/metrics"
	"k8s.io/kubernetes/test/integration/framework"
)

var zero = int64(0)

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
		informers.Apps().V1().DaemonSets(),
		informers.Apps().V1().ControllerRevisions(),
		informers.Core().V1().Pods(),
		informers.Core().V1().Nodes(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "daemonset-controller")),
	)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}

	return server, closeFn, dc, informers, clientSet
}

func setupScheduler(
	t *testing.T,
	cs clientset.Interface,
	informerFactory informers.SharedInformerFactory,
	stopCh chan struct{},
) {
	// If ScheduleDaemonSetPods is disabled, do not start scheduler.
	if !utilfeature.DefaultFeatureGate.Enabled(features.ScheduleDaemonSetPods) {
		return
	}

	schedulerConfigFactory := factory.NewConfigFactory(
		v1.DefaultSchedulerName,
		cs,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		informerFactory.Storage().V1().StorageClasses(),
		v1.DefaultHardPodAffinitySymmetricWeight,
		true,
		false,
	)

	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		t.Fatalf("Couldn't create scheduler config: %v", err)
	}

	schedulerConfig.StopEverything = stopCh

	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(
		legacyscheme.Scheme,
		v1.EventSource{Component: v1.DefaultSchedulerName},
	)
	eventBroadcaster.StartRecordingToSink(&clientv1core.EventSinkImpl{
		Interface: cs.CoreV1().Events(""),
	})

	sched, err := scheduler.NewFromConfigurator(
		&scheduler.FakeConfigurator{Config: schedulerConfig}, nil...)
	if err != nil {
		t.Fatalf("error creating scheduler: %v", err)
	}

	algorithmprovider.ApplyFeatureGates()

	go sched.Run()
}

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
					Containers:                    []v1.Container{{Name: "foo", Image: "bar"}},
					TerminationGracePeriodSeconds: &zero,
				},
			},
		},
	}
}

func cleanupDaemonSets(t *testing.T, cs clientset.Interface, ds *apps.DaemonSet) {
	ds, err := cs.AppsV1().DaemonSets(ds.Namespace).Get(ds.Name, metav1.GetOptions{})
	if err != nil {
		t.Errorf("Failed to get DaemonSet %s/%s: %v", ds.Namespace, ds.Name, err)
		return
	}

	// We set the nodeSelector to a random label. This label is nearly guaranteed
	// to not be set on any node so the DameonSetController will start deleting
	// daemon pods. Once it's done deleting the daemon pods, it's safe to delete
	// the DaemonSet.
	ds.Spec.Template.Spec.NodeSelector = map[string]string{
		string(uuid.NewUUID()): string(uuid.NewUUID()),
	}
	// force update to avoid version conflict
	ds.ResourceVersion = ""

	if ds, err = cs.AppsV1().DaemonSets(ds.Namespace).Update(ds); err != nil {
		t.Errorf("Failed to update DaemonSet %s/%s: %v", ds.Namespace, ds.Name, err)
		return
	}

	// Wait for the daemon set controller to kill all the daemon pods.
	if err := wait.Poll(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		updatedDS, err := cs.AppsV1().DaemonSets(ds.Namespace).Get(ds.Name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return updatedDS.Status.CurrentNumberScheduled+updatedDS.Status.NumberMisscheduled == 0, nil
	}); err != nil {
		t.Errorf("Failed to kill the pods of DaemonSet %s/%s: %v", ds.Namespace, ds.Name, err)
		return
	}

	falseVar := false
	deleteOptions := &metav1.DeleteOptions{OrphanDependents: &falseVar}
	if err := cs.AppsV1().DaemonSets(ds.Namespace).Delete(ds.Name, deleteOptions); err != nil {
		t.Errorf("Failed to delete DaemonSet %s/%s: %v", ds.Namespace, ds.Name, err)
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

func featureGates() []utilfeature.Feature {
	return []utilfeature.Feature{
		features.ScheduleDaemonSetPods,
	}
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
		TerminationGracePeriodSeconds: &zero,
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
			if got, want := controllerRef.Kind, "DaemonSet"; got != want {
				t.Errorf("controllerRef.Kind = %q, want %q", got, want)
			}
			if controllerRef.Controller == nil || *controllerRef.Controller != true {
				t.Errorf("controllerRef.Controller is not set to true")
			}

			if !podutil.IsPodReady(pod) && len(pod.Spec.NodeName) != 0 {
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

// podUnschedulable returns a condition function that returns true if the given pod
// gets unschedulable status.
func podUnschedulable(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(podName, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
		return cond != nil && cond.Status == v1.ConditionFalse &&
			cond.Reason == v1.PodReasonUnschedulable, nil
	}
}

// waitForPodUnscheduleWithTimeout waits for a pod to fail scheduling and returns
// an error if it does not become unschedulable within the given timeout.
func waitForPodUnschedulableWithTimeout(cs clientset.Interface, pod *v1.Pod, timeout time.Duration) error {
	return wait.Poll(100*time.Millisecond, timeout, podUnschedulable(cs, pod.Namespace, pod.Name))
}

// waitForPodUnschedule waits for a pod to fail scheduling and returns
// an error if it does not become unschedulable within the timeout duration (30 seconds).
func waitForPodUnschedulable(cs clientset.Interface, pod *v1.Pod) error {
	return waitForPodUnschedulableWithTimeout(cs, pod, 10*time.Second)
}

// waitForPodsCreated waits for number of pods are created.
func waitForPodsCreated(podInformer cache.SharedIndexInformer, num int) error {
	return wait.Poll(100*time.Millisecond, 10*time.Second, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		return len(objects) == num, nil
	})
}

func validateDaemonSetStatus(
	dsClient appstyped.DaemonSetInterface,
	dsName string,
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

func forEachFeatureGate(t *testing.T, tf func(t *testing.T)) {
	for _, fg := range featureGates() {
		func() {
			enabled := utilfeature.DefaultFeatureGate.Enabled(fg)
			defer func() {
				utilfeature.DefaultFeatureGate.Set(fmt.Sprintf("%v=%t", fg, enabled))
			}()

			for _, f := range []bool{true, false} {
				utilfeature.DefaultFeatureGate.Set(fmt.Sprintf("%v=%t", fg, f))
				t.Run(fmt.Sprintf("%v (%t)", fg, f), tf)
			}
		}()
	}
}

func forEachStrategy(t *testing.T, tf func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy)) {
	for _, strategy := range updateStrategies() {
		t.Run(fmt.Sprintf("%s (%v)", t.Name(), strategy),
			func(tt *testing.T) { tf(tt, strategy) })
	}
}

func TestOneNodeDaemonLaunchesPod(t *testing.T) {
	forEachFeatureGate(t, func(t *testing.T) {
		forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
			server, closeFn, dc, informers, clientset := setup(t)
			defer closeFn()
			ns := framework.CreateTestingNamespace("one-node-daemonset-test", server, t)
			defer framework.DeleteTestingNamespace(ns, server, t)

			dsClient := clientset.AppsV1().DaemonSets(ns.Name)
			podClient := clientset.CoreV1().Pods(ns.Name)
			nodeClient := clientset.CoreV1().Nodes()
			podInformer := informers.Core().V1().Pods().Informer()

			stopCh := make(chan struct{})
			defer close(stopCh)

			informers.Start(stopCh)
			go dc.Run(5, stopCh)

			// Start Scheduler
			setupScheduler(t, clientset, informers, stopCh)

			ds := newDaemonSet("foo", ns.Name)
			ds.Spec.UpdateStrategy = *strategy
			_, err := dsClient.Create(ds)
			if err != nil {
				t.Fatalf("Failed to create DaemonSet: %v", err)
			}
			defer cleanupDaemonSets(t, clientset, ds)

			_, err = nodeClient.Create(newNode("single-node", nil))
			if err != nil {
				t.Fatalf("Failed to create node: %v", err)
			}

			validateDaemonSetPodsAndMarkReady(podClient, podInformer, 1, t)
			validateDaemonSetStatus(dsClient, ds.Name, 1, t)
		})
	})
}

func TestSimpleDaemonSetLaunchesPods(t *testing.T) {
	forEachFeatureGate(t, func(t *testing.T) {
		forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
			server, closeFn, dc, informers, clientset := setup(t)
			defer closeFn()
			ns := framework.CreateTestingNamespace("simple-daemonset-test", server, t)
			defer framework.DeleteTestingNamespace(ns, server, t)

			dsClient := clientset.AppsV1().DaemonSets(ns.Name)
			podClient := clientset.CoreV1().Pods(ns.Name)
			nodeClient := clientset.CoreV1().Nodes()
			podInformer := informers.Core().V1().Pods().Informer()

			stopCh := make(chan struct{})
			defer close(stopCh)

			informers.Start(stopCh)
			go dc.Run(5, stopCh)

			// Start Scheduler
			setupScheduler(t, clientset, informers, stopCh)

			ds := newDaemonSet("foo", ns.Name)
			ds.Spec.UpdateStrategy = *strategy
			_, err := dsClient.Create(ds)
			if err != nil {
				t.Fatalf("Failed to create DaemonSet: %v", err)
			}
			defer cleanupDaemonSets(t, clientset, ds)

			addNodes(nodeClient, 0, 5, nil, t)

			validateDaemonSetPodsAndMarkReady(podClient, podInformer, 5, t)
			validateDaemonSetStatus(dsClient, ds.Name, 5, t)
		})
	})
}

func TestDaemonSetWithNodeSelectorLaunchesPods(t *testing.T) {
	forEachFeatureGate(t, func(t *testing.T) {
		forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
			server, closeFn, dc, informers, clientset := setup(t)
			defer closeFn()
			ns := framework.CreateTestingNamespace("simple-daemonset-test", server, t)
			defer framework.DeleteTestingNamespace(ns, server, t)

			dsClient := clientset.AppsV1().DaemonSets(ns.Name)
			podClient := clientset.CoreV1().Pods(ns.Name)
			nodeClient := clientset.CoreV1().Nodes()
			podInformer := informers.Core().V1().Pods().Informer()

			stopCh := make(chan struct{})
			defer close(stopCh)

			informers.Start(stopCh)
			go dc.Run(5, stopCh)

			// Start Scheduler
			setupScheduler(t, clientset, informers, stopCh)

			ds := newDaemonSet("foo", ns.Name)
			ds.Spec.UpdateStrategy = *strategy

			ds.Spec.Template.Spec.Affinity = &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "zone",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"test"},
									},
								},
							},
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      algorithm.NodeFieldSelectorKeyNodeName,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"node-1"},
									},
								},
							},
						},
					},
				},
			}

			_, err := dsClient.Create(ds)
			if err != nil {
				t.Fatalf("Failed to create DaemonSet: %v", err)
			}
			defer cleanupDaemonSets(t, clientset, ds)

			addNodes(nodeClient, 0, 2, nil, t)
			// Two nodes with labels
			addNodes(nodeClient, 2, 2, map[string]string{
				"zone": "test",
			}, t)
			addNodes(nodeClient, 4, 2, nil, t)

			validateDaemonSetPodsAndMarkReady(podClient, podInformer, 3, t)
			validateDaemonSetStatus(dsClient, ds.Name, 3, t)
		})
	})
}

func TestNotReadyNodeDaemonDoesLaunchPod(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("simple-daemonset-test", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()

		stopCh := make(chan struct{})
		defer close(stopCh)

		informers.Start(stopCh)
		go dc.Run(5, stopCh)

		// Start Scheduler
		setupScheduler(t, clientset, informers, stopCh)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(ds)
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}

		defer cleanupDaemonSets(t, clientset, ds)

		node := newNode("single-node", nil)
		node.Status.Conditions = []v1.NodeCondition{
			{Type: v1.NodeReady, Status: v1.ConditionFalse},
		}
		_, err = nodeClient.Create(node)
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 1, t)
		validateDaemonSetStatus(dsClient, ds.Name, 1, t)
	})
}

func TestInsufficientCapacityNodeDaemonDoesNotLaunchPod(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("insufficient-capacity", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		eventClient := clientset.CoreV1().Events(ns.Namespace)

		stopCh := make(chan struct{})
		defer close(stopCh)

		informers.Start(stopCh)
		go dc.Run(5, stopCh)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.Template.Spec = resourcePodSpec("node-with-limited-memory", "120M", "75m")
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(ds)
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}
		defer cleanupDaemonSets(t, clientset, ds)

		node := newNode("node-with-limited-memory", nil)
		node.Status.Allocatable = allocatableResources("100M", "200m")
		_, err = nodeClient.Create(node)
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		validateFailedPlacementEvent(eventClient, t)
	})
}

// TestInsufficientCapacityNodeDaemonSetCreateButNotLaunchPod tests that when "ScheduleDaemonSetPods"
// feature is enabled, the DaemonSet should create Pods for all the nodes regardless of available resource
// on the nodes, and kube-scheduler should not schedule Pods onto the nodes with insufficient resource.
func TestInsufficientCapacityNodeWhenScheduleDaemonSetPodsEnabled(t *testing.T) {
	enabled := utilfeature.DefaultFeatureGate.Enabled(features.ScheduleDaemonSetPods)
	defer func() {
		utilfeature.DefaultFeatureGate.Set(fmt.Sprintf("%s=%t",
			features.ScheduleDaemonSetPods, enabled))
	}()

	utilfeature.DefaultFeatureGate.Set(fmt.Sprintf("%s=%t", features.ScheduleDaemonSetPods, true))

	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("insufficient-capacity", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		podInformer := informers.Core().V1().Pods().Informer()
		nodeClient := clientset.CoreV1().Nodes()
		stopCh := make(chan struct{})
		defer close(stopCh)

		informers.Start(stopCh)
		go dc.Run(5, stopCh)

		// Start Scheduler
		setupScheduler(t, clientset, informers, stopCh)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.Template.Spec = resourcePodSpec("", "120M", "75m")
		ds.Spec.UpdateStrategy = *strategy
		ds, err := dsClient.Create(ds)
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}

		defer cleanupDaemonSets(t, clientset, ds)

		node := newNode("node-with-limited-memory", nil)
		node.Status.Allocatable = allocatableResources("100M", "200m")
		_, err = nodeClient.Create(node)
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		if err := waitForPodsCreated(podInformer, 1); err != nil {
			t.Errorf("Failed to wait for pods created: %v", err)
		}

		objects := podInformer.GetIndexer().List()
		for _, object := range objects {
			pod := object.(*v1.Pod)
			if err := waitForPodUnschedulable(clientset, pod); err != nil {
				t.Errorf("Failed to wait for unschedulable status of pod %+v", pod)
			}
		}

		node1 := newNode("node-with-enough-memory", nil)
		node1.Status.Allocatable = allocatableResources("200M", "2000m")
		_, err = nodeClient.Create(node1)
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// When ScheduleDaemonSetPods enabled, 2 pods are created. But only one
		// of two Pods is scheduled by default scheduler.
		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 2, t)
		validateDaemonSetStatus(dsClient, ds.Name, 1, t)
	})
}
