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
	"context"
	"fmt"
	"net/http/httptest"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	appstyped "k8s.io/client-go/kubernetes/typed/apps/v1"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/client-go/util/retry"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
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
	dc, err := daemon.NewDaemonSetsController(
		informers.Apps().V1().DaemonSets(),
		informers.Apps().V1().ControllerRevisions(),
		informers.Core().V1().Pods(),
		informers.Core().V1().Nodes(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "daemonset-controller")),
		flowcontrol.NewBackOff(5*time.Second, 15*time.Minute),
	)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}

	return server, closeFn, dc, informers, clientSet
}

func setupScheduler(
	ctx context.Context,
	t *testing.T,
	cs clientset.Interface,
	informerFactory informers.SharedInformerFactory,
) {
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{
		Interface: cs.EventsV1beta1().Events(""),
	})

	sched, err := scheduler.New(
		cs,
		informerFactory,
		informerFactory.Core().V1().Pods(),
		profile.NewRecorderFactory(eventBroadcaster),
		ctx.Done(),
	)
	if err != nil {
		t.Fatalf("Couldn't create scheduler: %v", err)
	}

	eventBroadcaster.StartRecordingToSink(ctx.Done())

	go sched.Run(ctx)
	return
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
	ds, err := cs.AppsV1().DaemonSets(ds.Namespace).Get(context.TODO(), ds.Name, metav1.GetOptions{})
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

	if ds, err = cs.AppsV1().DaemonSets(ds.Namespace).Update(context.TODO(), ds, metav1.UpdateOptions{}); err != nil {
		t.Errorf("Failed to update DaemonSet %s/%s: %v", ds.Namespace, ds.Name, err)
		return
	}

	// Wait for the daemon set controller to kill all the daemon pods.
	if err := wait.Poll(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		updatedDS, err := cs.AppsV1().DaemonSets(ds.Namespace).Get(context.TODO(), ds.Name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return updatedDS.Status.CurrentNumberScheduled+updatedDS.Status.NumberMisscheduled == 0, nil
	}); err != nil {
		t.Errorf("Failed to kill the pods of DaemonSet %s/%s: %v", ds.Namespace, ds.Name, err)
		return
	}

	falseVar := false
	deleteOptions := metav1.DeleteOptions{OrphanDependents: &falseVar}
	if err := cs.AppsV1().DaemonSets(ds.Namespace).Delete(context.TODO(), ds.Name, deleteOptions); err != nil {
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
			Namespace: metav1.NamespaceNone,
		},
		Status: v1.NodeStatus{
			Conditions:  []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
			Allocatable: v1.ResourceList{v1.ResourcePods: resource.MustParse("100")},
		},
	}
}

func addNodes(nodeClient corev1client.NodeInterface, startIndex, numNodes int, label map[string]string, t *testing.T) {
	for i := startIndex; i < startIndex+numNodes; i++ {
		_, err := nodeClient.Create(context.TODO(), newNode(fmt.Sprintf("node-%d", i), label), metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}
	}
}

func validateDaemonSetPodsAndMarkReady(
	podClient corev1client.PodInterface,
	podInformer cache.SharedIndexInformer,
	numberPods int,
	t *testing.T,
) {
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
				_, err := podClient.UpdateStatus(context.TODO(), podCopy, metav1.UpdateOptions{})
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
		pod, err := c.CoreV1().Pods(podNamespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
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

func waitForDaemonSetAndControllerRevisionCreated(c clientset.Interface, name string, namespace string) error {
	return wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (bool, error) {
		ds, err := c.AppsV1().DaemonSets(namespace).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if ds == nil {
			return false, nil
		}

		revs, err := c.AppsV1().ControllerRevisions(namespace).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		if revs.Size() == 0 {
			return false, nil
		}

		for _, rev := range revs.Items {
			for _, oref := range rev.OwnerReferences {
				if oref.Kind == "DaemonSet" && oref.UID == ds.UID {
					return true, nil
				}
			}
		}
		return false, nil
	})
}

func hashAndNameForDaemonSet(ds *apps.DaemonSet) (string, string) {
	hash := fmt.Sprint(controller.ComputeHash(&ds.Spec.Template, ds.Status.CollisionCount))
	name := ds.Name + "-" + hash
	return hash, name
}

func validateDaemonSetCollisionCount(dsClient appstyped.DaemonSetInterface, dsName string, expCount int32, t *testing.T) {
	ds, err := dsClient.Get(context.TODO(), dsName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to look up DaemonSet: %v", err)
	}
	collisionCount := ds.Status.CollisionCount
	if *collisionCount != expCount {
		t.Fatalf("Expected collisionCount to be %d, but found %d", expCount, *collisionCount)
	}
}

func validateDaemonSetStatus(
	dsClient appstyped.DaemonSetInterface,
	dsName string,
	expectedNumberReady int32,
	t *testing.T) {
	if err := wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) {
		ds, err := dsClient.Get(context.TODO(), dsName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return ds.Status.NumberReady == expectedNumberReady, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func updateDS(t *testing.T, dsClient appstyped.DaemonSetInterface, dsName string, updateFunc func(*apps.DaemonSet)) *apps.DaemonSet {
	var ds *apps.DaemonSet
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newDS, err := dsClient.Get(context.TODO(), dsName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateFunc(newDS)
		ds, err = dsClient.Update(context.TODO(), newDS, metav1.UpdateOptions{})
		return err
	}); err != nil {
		t.Fatalf("Failed to update DaemonSet: %v", err)
	}
	return ds
}

func forEachStrategy(t *testing.T, tf func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy)) {
	for _, strategy := range updateStrategies() {
		t.Run(fmt.Sprintf("%s (%v)", t.Name(), strategy),
			func(tt *testing.T) { tf(tt, strategy) })
	}
}

func TestOneNodeDaemonLaunchesPod(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("one-node-daemonset-test", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		// Start Scheduler
		setupScheduler(ctx, t, clientset, informers)

		informers.Start(ctx.Done())
		go dc.Run(5, ctx.Done())

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(context.TODO(), ds, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}
		defer cleanupDaemonSets(t, clientset, ds)

		_, err = nodeClient.Create(context.TODO(), newNode("single-node", nil), metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 1, t)
		validateDaemonSetStatus(dsClient, ds.Name, 1, t)
	})
}

func TestSimpleDaemonSetLaunchesPods(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("simple-daemonset-test", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		informers.Start(ctx.Done())
		go dc.Run(5, ctx.Done())

		// Start Scheduler
		setupScheduler(ctx, t, clientset, informers)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(context.TODO(), ds, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}
		defer cleanupDaemonSets(t, clientset, ds)

		addNodes(nodeClient, 0, 5, nil, t)

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 5, t)
		validateDaemonSetStatus(dsClient, ds.Name, 5, t)
	})
}

func TestDaemonSetWithNodeSelectorLaunchesPods(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("simple-daemonset-test", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		informers.Start(ctx.Done())
		go dc.Run(5, ctx.Done())

		// Start Scheduler
		setupScheduler(ctx, t, clientset, informers)

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
									Key:      api.ObjectNameField,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"node-1"},
								},
							},
						},
					},
				},
			},
		}

		_, err := dsClient.Create(context.TODO(), ds, metav1.CreateOptions{})
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

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		informers.Start(ctx.Done())
		go dc.Run(5, ctx.Done())

		// Start Scheduler
		setupScheduler(ctx, t, clientset, informers)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(context.TODO(), ds, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}

		defer cleanupDaemonSets(t, clientset, ds)

		node := newNode("single-node", nil)
		node.Status.Conditions = []v1.NodeCondition{
			{Type: v1.NodeReady, Status: v1.ConditionFalse},
		}
		_, err = nodeClient.Create(context.TODO(), node, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 1, t)
		validateDaemonSetStatus(dsClient, ds.Name, 1, t)
	})
}

// TestInsufficientCapacityNodeDaemonSetCreateButNotLaunchPod tests thaat the DaemonSet should create
// Pods for all the nodes regardless of available resource on the nodes, and kube-scheduler should
// not schedule Pods onto the nodes with insufficient resource.
func TestInsufficientCapacityNode(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("insufficient-capacity", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		podInformer := informers.Core().V1().Pods().Informer()
		nodeClient := clientset.CoreV1().Nodes()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		informers.Start(ctx.Done())
		go dc.Run(5, ctx.Done())

		// Start Scheduler
		setupScheduler(ctx, t, clientset, informers)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.Template.Spec = resourcePodSpec("", "120M", "75m")
		ds.Spec.UpdateStrategy = *strategy
		ds, err := dsClient.Create(context.TODO(), ds, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}

		defer cleanupDaemonSets(t, clientset, ds)

		node := newNode("node-with-limited-memory", nil)
		node.Status.Allocatable = allocatableResources("100M", "200m")
		_, err = nodeClient.Create(context.TODO(), node, metav1.CreateOptions{})
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
		_, err = nodeClient.Create(context.TODO(), node1, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// 2 pods are created. But only one of two Pods is scheduled by default scheduler.
		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 2, t)
		validateDaemonSetStatus(dsClient, ds.Name, 1, t)
	})
}

// TestLaunchWithHashCollision tests that a DaemonSet can be updated even if there is a
// hash collision with an existing ControllerRevision
func TestLaunchWithHashCollision(t *testing.T) {
	server, closeFn, dc, informers, clientset := setup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("one-node-daemonset-test", server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	dsClient := clientset.AppsV1().DaemonSets(ns.Name)
	podInformer := informers.Core().V1().Pods().Informer()
	nodeClient := clientset.CoreV1().Nodes()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	informers.Start(ctx.Done())
	go dc.Run(5, ctx.Done())

	// Start Scheduler
	setupScheduler(ctx, t, clientset, informers)

	// Create single node
	_, err := nodeClient.Create(context.TODO(), newNode("single-node", nil), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// Create new DaemonSet with RollingUpdate strategy
	orgDs := newDaemonSet("foo", ns.Name)
	oneIntString := intstr.FromInt(1)
	orgDs.Spec.UpdateStrategy = apps.DaemonSetUpdateStrategy{
		Type: apps.RollingUpdateDaemonSetStrategyType,
		RollingUpdate: &apps.RollingUpdateDaemonSet{
			MaxUnavailable: &oneIntString,
		},
	}
	ds, err := dsClient.Create(context.TODO(), orgDs, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create DaemonSet: %v", err)
	}

	// Wait for the DaemonSet to be created before proceeding
	err = waitForDaemonSetAndControllerRevisionCreated(clientset, ds.Name, ds.Namespace)
	if err != nil {
		t.Fatalf("Failed to create DaemonSet: %v", err)
	}

	ds, err = dsClient.Get(context.TODO(), ds.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get DaemonSet: %v", err)
	}
	var orgCollisionCount int32
	if ds.Status.CollisionCount != nil {
		orgCollisionCount = *ds.Status.CollisionCount
	}

	// Look up the ControllerRevision for the DaemonSet
	_, name := hashAndNameForDaemonSet(ds)
	revision, err := clientset.AppsV1().ControllerRevisions(ds.Namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil || revision == nil {
		t.Fatalf("Failed to look up ControllerRevision: %v", err)
	}

	// Create a "fake" ControllerRevision that we know will create a hash collision when we make
	// the next update
	one := int64(1)
	ds.Spec.Template.Spec.TerminationGracePeriodSeconds = &one

	newHash, newName := hashAndNameForDaemonSet(ds)
	newRevision := &apps.ControllerRevision{
		ObjectMeta: metav1.ObjectMeta{
			Name:            newName,
			Namespace:       ds.Namespace,
			Labels:          labelsutil.CloneAndAddLabel(ds.Spec.Template.Labels, apps.DefaultDaemonSetUniqueLabelKey, newHash),
			Annotations:     ds.Annotations,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(ds, apps.SchemeGroupVersion.WithKind("DaemonSet"))},
		},
		Data:     revision.Data,
		Revision: revision.Revision + 1,
	}
	_, err = clientset.AppsV1().ControllerRevisions(ds.Namespace).Create(context.TODO(), newRevision, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create ControllerRevision: %v", err)
	}

	// Make an update of the DaemonSet which we know will create a hash collision when
	// the next ControllerRevision is created.
	ds = updateDS(t, dsClient, ds.Name, func(updateDS *apps.DaemonSet) {
		updateDS.Spec.Template.Spec.TerminationGracePeriodSeconds = &one
	})

	// Wait for any pod with the latest Spec to exist
	err = wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		for _, object := range objects {
			pod := object.(*v1.Pod)
			if *pod.Spec.TerminationGracePeriodSeconds == *ds.Spec.Template.Spec.TerminationGracePeriodSeconds {
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Failed to wait for Pods with the latest Spec to be created: %v", err)
	}

	validateDaemonSetCollisionCount(dsClient, ds.Name, orgCollisionCount+1, t)
}

// TestTaintedNode tests tainted node isn't expected to have pod scheduled
func TestTaintedNode(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("tainted-node", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		podInformer := informers.Core().V1().Pods().Informer()
		nodeClient := clientset.CoreV1().Nodes()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		informers.Start(ctx.Done())
		go dc.Run(5, ctx.Done())

		// Start Scheduler
		setupScheduler(ctx, t, clientset, informers)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		ds, err := dsClient.Create(context.TODO(), ds, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}

		defer cleanupDaemonSets(t, clientset, ds)

		nodeWithTaint := newNode("node-with-taint", nil)
		nodeWithTaint.Spec.Taints = []v1.Taint{{Key: "key1", Value: "val1", Effect: "NoSchedule"}}
		_, err = nodeClient.Create(context.TODO(), nodeWithTaint, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create nodeWithTaint: %v", err)
		}

		nodeWithoutTaint := newNode("node-without-taint", nil)
		_, err = nodeClient.Create(context.TODO(), nodeWithoutTaint, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create nodeWithoutTaint: %v", err)
		}

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 1, t)
		validateDaemonSetStatus(dsClient, ds.Name, 1, t)

		// remove taint from nodeWithTaint
		nodeWithTaint, err = nodeClient.Get(context.TODO(), "node-with-taint", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to retrieve nodeWithTaint: %v", err)
		}
		nodeWithTaintCopy := nodeWithTaint.DeepCopy()
		nodeWithTaintCopy.Spec.Taints = []v1.Taint{}
		_, err = nodeClient.Update(context.TODO(), nodeWithTaintCopy, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to update nodeWithTaint: %v", err)
		}

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 2, t)
		validateDaemonSetStatus(dsClient, ds.Name, 2, t)
	})
}

// TestUnschedulableNodeDaemonDoesLaunchPod tests that the DaemonSet Pods can still be scheduled
// to the Unschedulable nodes.
func TestUnschedulableNodeDaemonDoesLaunchPod(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		server, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("daemonset-unschedulable-test", server, t)
		defer framework.DeleteTestingNamespace(ns, server, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		informers.Start(ctx.Done())
		go dc.Run(5, ctx.Done())

		// Start Scheduler
		setupScheduler(ctx, t, clientset, informers)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		ds.Spec.Template.Spec.HostNetwork = true
		_, err := dsClient.Create(context.TODO(), ds, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}

		defer cleanupDaemonSets(t, clientset, ds)

		// Creates unschedulable node.
		node := newNode("unschedulable-node", nil)
		node.Spec.Unschedulable = true
		node.Spec.Taints = []v1.Taint{
			{
				Key:    v1.TaintNodeUnschedulable,
				Effect: v1.TaintEffectNoSchedule,
			},
		}

		_, err = nodeClient.Create(context.TODO(), node, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// Creates network-unavailable node.
		nodeNU := newNode("network-unavailable-node", nil)
		nodeNU.Status.Conditions = []v1.NodeCondition{
			{Type: v1.NodeReady, Status: v1.ConditionFalse},
			{Type: v1.NodeNetworkUnavailable, Status: v1.ConditionTrue},
		}
		nodeNU.Spec.Taints = []v1.Taint{
			{
				Key:    v1.TaintNodeNetworkUnavailable,
				Effect: v1.TaintEffectNoSchedule,
			},
		}

		_, err = nodeClient.Create(context.TODO(), nodeNU, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 2, t)
		validateDaemonSetStatus(dsClient, ds.Name, 2, t)
	})
}
