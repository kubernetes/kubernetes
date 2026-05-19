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
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
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
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/kubernetes/test/utils/ktesting"
)

var zero = int64(0)

func setup(t *testing.T) (context.Context, kubeapiservertesting.TearDownFunc, *daemon.DaemonSetsController, informers.SharedInformerFactory, clientset.Interface) {
	return setupWithServerSetup(t, framework.TestServerSetup{})
}

func setupWithServerSetup(t *testing.T, serverSetup framework.TestServerSetup) (context.Context, kubeapiservertesting.TearDownFunc, *daemon.DaemonSetsController, informers.SharedInformerFactory, clientset.Interface) {
	tCtx := ktesting.Init(t)
	modifyServerRunOptions := serverSetup.ModifyServerRunOptions
	serverSetup.ModifyServerRunOptions = func(opts *options.ServerRunOptions) {
		if modifyServerRunOptions != nil {
			modifyServerRunOptions(opts)
		}

		opts.Admission.GenericAdmission.DisablePlugins = append(opts.Admission.GenericAdmission.DisablePlugins,
			// Disable ServiceAccount admission plugin as we don't have
			// serviceaccount controller running.
			"ServiceAccount",
			"TaintNodesByCondition",
		)
	}

	clientSet, config, closeFn := framework.StartTestServer(tCtx, t, serverSetup)

	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "daemonset-informers")), resyncPeriod)
	dc, err := daemon.NewDaemonSetsController(
		tCtx,
		informers.Apps().V1().DaemonSets(),
		informers.Apps().V1().ControllerRevisions(),
		informers.Core().V1().Pods(),
		informers.Core().V1().Nodes(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "daemonset-controller")),
		flowcontrol.NewBackOff(5*time.Second, 15*time.Minute),
	)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}

	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{
		Interface: clientSet.EventsV1(),
	})

	sched, err := scheduler.New(
		tCtx,
		clientSet,
		informers,
		nil,
		profile.NewRecorderFactory(eventBroadcaster),
	)
	if err != nil {
		t.Fatalf("Couldn't create scheduler: %v", err)
	}

	eventBroadcaster.StartRecordingToSink(tCtx.Done())
	go sched.Run(tCtx)

	tearDownFn := func() {
		tCtx.Cancel("tearing down apiserver")
		closeFn()
		eventBroadcaster.Shutdown()
	}

	return tCtx, tearDownFn, dc, informers, clientSet
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
	t.Helper()
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

	if len(ds.Spec.Template.Finalizers) > 0 {
		testutils.RemovePodFinalizersInNamespace(context.TODO(), cs, t, ds.Namespace)
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
	one := intstr.FromInt32(1)
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
	if err := wait.Poll(time.Second, 60*time.Second, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		nonTerminatedPods := 0

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

			if podutil.IsPodPhaseTerminal(pod.Status.Phase) {
				continue
			}
			nonTerminatedPods++
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

		return nonTerminatedPods == numberPods, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func validateDaemonSetPodsActive(
	podClient corev1client.PodInterface,
	podInformer cache.SharedIndexInformer,
	numberPods int,
	t *testing.T,
) {
	if err := wait.Poll(time.Second, 60*time.Second, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		if len(objects) < numberPods {
			return false, nil
		}
		podsActiveCount := 0
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
			if pod.Status.Phase == v1.PodRunning || pod.Status.Phase == v1.PodPending {
				podsActiveCount += 1
			}
		}
		return podsActiveCount == numberPods, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func validateDaemonSetPodsTolerations(
	podClient corev1client.PodInterface,
	podInformer cache.SharedIndexInformer,
	expectedTolerations []v1.Toleration,
	prefix string,
	t *testing.T,
) {
	objects := podInformer.GetIndexer().List()
	for _, object := range objects {
		var prefixedPodToleration []v1.Toleration
		pod := object.(*v1.Pod)
		ownerReferences := pod.ObjectMeta.OwnerReferences
		if len(ownerReferences) != 1 {
			t.Errorf("Pod %s has %d OwnerReferences, expected only 1", pod.Name, len(ownerReferences))
		}
		controllerRef := ownerReferences[0]
		if got, want := controllerRef.Kind, "DaemonSet"; got != want {
			t.Errorf("controllerRef.Kind = %q, want %q", got, want)
		}
		if controllerRef.Controller == nil || *controllerRef.Controller != true {
			t.Errorf("controllerRef.Controller is not set to true")
		}
		for _, podToleration := range pod.Spec.Tolerations {
			if strings.HasPrefix(podToleration.Key, prefix) {
				prefixedPodToleration = append(prefixedPodToleration, podToleration)
			}
		}
		if diff := cmp.Diff(expectedTolerations, prefixedPodToleration); diff != "" {
			t.Fatalf("Unexpected tolerations (-want,+got):\n%s", diff)
		}
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
	if err := wait.Poll(time.Second, 60*time.Second, func() (bool, error) {
		ds, err := dsClient.Get(context.TODO(), dsName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return ds.Status.NumberReady == expectedNumberReady, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func validateUpdatedNumberScheduled(
	ctx context.Context,
	dsClient appstyped.DaemonSetInterface,
	dsName string,
	expectedUpdatedNumberScheduled int32,
	t *testing.T) {
	if err := wait.PollUntilContextTimeout(ctx, time.Second, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		ds, err := dsClient.Get(context.TODO(), dsName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return ds.Status.UpdatedNumberScheduled == expectedUpdatedNumberScheduled, nil
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
		t.Run(string(strategy.Type), func(t *testing.T) {
			tf(t, strategy)
		})
	}
}

func TestOneNodeDaemonLaunchesPod(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		ctx, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateNamespaceOrDie(clientset, "one-node-daemonset-test", t)
		defer framework.DeleteNamespaceOrDie(clientset, ns, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()

		informers.Start(ctx.Done())
		go dc.Run(ctx, 2)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(ctx, ds, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}
		defer cleanupDaemonSets(t, clientset, ds)

		_, err = nodeClient.Create(ctx, newNode("single-node", nil), metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 1, t)
		validateDaemonSetStatus(dsClient, ds.Name, 1, t)
	})
}

func TestSimpleDaemonSetLaunchesPods(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		ctx, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateNamespaceOrDie(clientset, "simple-daemonset-test", t)
		defer framework.DeleteNamespaceOrDie(clientset, ns, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()

		informers.Start(ctx.Done())
		go dc.Run(ctx, 2)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(ctx, ds, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}
		defer cleanupDaemonSets(t, clientset, ds)

		addNodes(nodeClient, 0, 5, nil, t)

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 5, t)
		validateDaemonSetStatus(dsClient, ds.Name, 5, t)
	})
}

func TestSimpleDaemonSetRestartsPodsOnTerminalPhase(t *testing.T) {
	cases := map[string]struct {
		phase     v1.PodPhase
		finalizer bool
	}{
		"Succeeded": {
			phase: v1.PodSucceeded,
		},
		"Failed": {
			phase: v1.PodFailed,
		},
		"Succeeded with finalizer": {
			phase:     v1.PodSucceeded,
			finalizer: true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
				ctx, closeFn, dc, informers, clientset := setup(t)
				defer closeFn()
				ns := framework.CreateNamespaceOrDie(clientset, "daemonset-restart-terminal-pod-test", t)
				defer framework.DeleteNamespaceOrDie(clientset, ns, t)

				dsClient := clientset.AppsV1().DaemonSets(ns.Name)
				podClient := clientset.CoreV1().Pods(ns.Name)
				nodeClient := clientset.CoreV1().Nodes()
				podInformer := informers.Core().V1().Pods().Informer()

				informers.Start(ctx.Done())
				go dc.Run(ctx, 2)

				ds := newDaemonSet("restart-terminal-pod", ns.Name)
				if tc.finalizer {
					ds.Spec.Template.Finalizers = append(ds.Spec.Template.Finalizers, "test.k8s.io/finalizer")
				}
				ds.Spec.UpdateStrategy = *strategy
				if _, err := dsClient.Create(ctx, ds, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create DaemonSet: %v", err)
				}
				defer cleanupDaemonSets(t, clientset, ds)

				numNodes := 3
				addNodes(nodeClient, 0, numNodes, nil, t)

				validateDaemonSetPodsAndMarkReady(podClient, podInformer, numNodes, t)
				validateDaemonSetStatus(dsClient, ds.Name, int32(numNodes), t)
				podToMarkAsTerminal := podInformer.GetIndexer().List()[0].(*v1.Pod)
				podCopy := podToMarkAsTerminal.DeepCopy()
				podCopy.Status.Phase = tc.phase
				if _, err := podClient.UpdateStatus(ctx, podCopy, metav1.UpdateOptions{}); err != nil {
					t.Fatalf("Failed to mark the pod as terminal with phase: %v. Error: %v", tc.phase, err)
				}
				// verify all pods are active. They either continue Running or are Pending after restart
				validateDaemonSetPodsActive(podClient, podInformer, numNodes, t)
				validateDaemonSetPodsAndMarkReady(podClient, podInformer, numNodes, t)
				validateDaemonSetStatus(dsClient, ds.Name, int32(numNodes), t)
			})
		})
	}
}

func TestDaemonSetWithNodeSelectorLaunchesPods(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		ctx, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateNamespaceOrDie(clientset, "simple-daemonset-test", t)
		defer framework.DeleteNamespaceOrDie(clientset, ns, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()

		informers.Start(ctx.Done())
		go dc.Run(ctx, 2)

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
									Key:      metav1.ObjectNameField,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"node-1"},
								},
							},
						},
					},
				},
			},
		}

		_, err := dsClient.Create(ctx, ds, metav1.CreateOptions{})
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
		ctx, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateNamespaceOrDie(clientset, "simple-daemonset-test", t)
		defer framework.DeleteNamespaceOrDie(clientset, ns, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()

		informers.Start(ctx.Done())
		go dc.Run(ctx, 2)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(ctx, ds, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}

		defer cleanupDaemonSets(t, clientset, ds)

		node := newNode("single-node", nil)
		node.Status.Conditions = []v1.NodeCondition{
			{Type: v1.NodeReady, Status: v1.ConditionFalse},
		}
		_, err = nodeClient.Create(ctx, node, metav1.CreateOptions{})
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
		ctx, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateNamespaceOrDie(clientset, "insufficient-capacity", t)
		defer framework.DeleteNamespaceOrDie(clientset, ns, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		podInformer := informers.Core().V1().Pods().Informer()
		nodeClient := clientset.CoreV1().Nodes()

		informers.Start(ctx.Done())
		go dc.Run(ctx, 2)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.Template.Spec = resourcePodSpec("", "120M", "75m")
		ds.Spec.UpdateStrategy = *strategy
		ds, err := dsClient.Create(ctx, ds, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}

		defer cleanupDaemonSets(t, clientset, ds)

		node := newNode("node-with-limited-memory", nil)
		node.Status.Allocatable = allocatableResources("100M", "200m")
		_, err = nodeClient.Create(ctx, node, metav1.CreateOptions{})
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
		_, err = nodeClient.Create(ctx, node1, metav1.CreateOptions{})
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
	ctx, closeFn, dc, informers, clientset := setup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(clientset, "one-node-daemonset-test", t)
	defer framework.DeleteNamespaceOrDie(clientset, ns, t)

	dsClient := clientset.AppsV1().DaemonSets(ns.Name)
	podInformer := informers.Core().V1().Pods().Informer()
	nodeClient := clientset.CoreV1().Nodes()

	informers.Start(ctx.Done())
	go dc.Run(ctx, 2)

	// Create single node
	_, err := nodeClient.Create(ctx, newNode("single-node", nil), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// Create new DaemonSet with RollingUpdate strategy
	orgDs := newDaemonSet("foo", ns.Name)
	oneIntString := intstr.FromInt32(1)
	orgDs.Spec.UpdateStrategy = apps.DaemonSetUpdateStrategy{
		Type: apps.RollingUpdateDaemonSetStrategyType,
		RollingUpdate: &apps.RollingUpdateDaemonSet{
			MaxUnavailable: &oneIntString,
		},
	}
	ds, err := dsClient.Create(ctx, orgDs, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create DaemonSet: %v", err)
	}

	// Wait for the DaemonSet to be created before proceeding
	err = waitForDaemonSetAndControllerRevisionCreated(clientset, ds.Name, ds.Namespace)
	if err != nil {
		t.Fatalf("Failed to create DaemonSet: %v", err)
	}

	ds, err = dsClient.Get(ctx, ds.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get DaemonSet: %v", err)
	}
	var orgCollisionCount int32
	if ds.Status.CollisionCount != nil {
		orgCollisionCount = *ds.Status.CollisionCount
	}

	// Look up the ControllerRevision for the DaemonSet
	_, name := hashAndNameForDaemonSet(ds)
	revision, err := clientset.AppsV1().ControllerRevisions(ds.Namespace).Get(ctx, name, metav1.GetOptions{})
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
	_, err = clientset.AppsV1().ControllerRevisions(ds.Namespace).Create(ctx, newRevision, metav1.CreateOptions{})
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

// Test DaemonSet Controller updates label of the pod after "DedupCurHistories". The scenario is
// 1. Create an another controllerrevision owned by the daemonset but with higher revision and different hash
// 2. Add a node to ensure the controller sync
// 3. The dsc is expected to "PATCH" the existing pod label with new hash and deletes the old controllerrevision once finishes the update
func TestDSCUpdatesPodLabelAfterDedupCurHistories(t *testing.T) {
	ctx, closeFn, dc, informers, clientset := setup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(clientset, "one-node-daemonset-test", t)
	defer framework.DeleteNamespaceOrDie(clientset, ns, t)

	dsClient := clientset.AppsV1().DaemonSets(ns.Name)
	podInformer := informers.Core().V1().Pods().Informer()
	nodeClient := clientset.CoreV1().Nodes()

	informers.Start(ctx.Done())
	go dc.Run(ctx, 2)

	// Create single node
	_, err := nodeClient.Create(ctx, newNode("single-node", nil), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// Create new DaemonSet with RollingUpdate strategy
	orgDs := newDaemonSet("foo", ns.Name)
	oneIntString := intstr.FromInt32(1)
	orgDs.Spec.UpdateStrategy = apps.DaemonSetUpdateStrategy{
		Type: apps.RollingUpdateDaemonSetStrategyType,
		RollingUpdate: &apps.RollingUpdateDaemonSet{
			MaxUnavailable: &oneIntString,
		},
	}
	ds, err := dsClient.Create(ctx, orgDs, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create DaemonSet: %v", err)
	}
	t.Logf("ds created")
	// Wait for the DaemonSet to be created before proceeding
	err = waitForDaemonSetAndControllerRevisionCreated(clientset, ds.Name, ds.Namespace)
	if err != nil {
		t.Fatalf("Failed to create DaemonSet: %v", err)
	}

	ds, err = dsClient.Get(ctx, ds.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get DaemonSet: %v", err)
	}

	// Look up the ControllerRevision for the DaemonSet
	_, name := hashAndNameForDaemonSet(ds)
	revision, err := clientset.AppsV1().ControllerRevisions(ds.Namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil || revision == nil {
		t.Fatalf("Failed to look up ControllerRevision: %v", err)
	}
	t.Logf("revision: %v", revision.Name)

	// Create a "fake" ControllerRevision which is owned by the same daemonset
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
	_, err = clientset.AppsV1().ControllerRevisions(ds.Namespace).Create(ctx, newRevision, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create ControllerRevision: %v", err)
	}
	t.Logf("revision: %v", newName)

	// ensure the daemonset to be synced
	_, err = nodeClient.Create(ctx, newNode("second-node", nil), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// check whether the pod label is updated after controllerrevision is created
	err = wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		for _, object := range objects {
			pod := object.(*v1.Pod)
			t.Logf("newHash: %v, label: %v", newHash, pod.ObjectMeta.Labels[apps.DefaultDaemonSetUniqueLabelKey])
			for _, oref := range pod.OwnerReferences {
				if oref.Name == ds.Name && oref.UID == ds.UID && oref.Kind == "DaemonSet" {
					if pod.ObjectMeta.Labels[apps.DefaultDaemonSetUniqueLabelKey] != newHash {
						return false, nil
					}
				}
			}
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Failed to update the pod label after new controllerrevision is created: %v", err)
	}

	err = wait.PollImmediate(1*time.Second, 10*time.Second, func() (bool, error) {
		revs, err := clientset.AppsV1().ControllerRevisions(ds.Namespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, fmt.Errorf("failed to list controllerrevision: %v", err)
		}
		if revs.Size() == 0 {
			return false, fmt.Errorf("no avaialable controllerrevision")
		}

		for _, rev := range revs.Items {
			t.Logf("revision: %v;hash: %v", rev.Name, rev.ObjectMeta.Labels[apps.DefaultDaemonSetUniqueLabelKey])
			for _, oref := range rev.OwnerReferences {
				if oref.Kind == "DaemonSet" && oref.UID == ds.UID {
					if rev.Name != newName {
						t.Logf("waiting for duplicate controllerrevision %v to be deleted", newName)
						return false, nil
					}
				}
			}
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Failed to check that duplicate controllerrevision is not deleted: %v", err)
	}
}

// TestTaintedNode tests tainted node isn't expected to have pod scheduled
func TestTaintedNode(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		ctx, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateNamespaceOrDie(clientset, "tainted-node", t)
		defer framework.DeleteNamespaceOrDie(clientset, ns, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		podInformer := informers.Core().V1().Pods().Informer()
		nodeClient := clientset.CoreV1().Nodes()

		informers.Start(ctx.Done())
		go dc.Run(ctx, 2)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		ds, err := dsClient.Create(ctx, ds, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}

		defer cleanupDaemonSets(t, clientset, ds)

		nodeWithTaint := newNode("node-with-taint", nil)
		nodeWithTaint.Spec.Taints = []v1.Taint{{Key: "key1", Value: "val1", Effect: "NoSchedule"}}
		_, err = nodeClient.Create(ctx, nodeWithTaint, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create nodeWithTaint: %v", err)
		}

		nodeWithoutTaint := newNode("node-without-taint", nil)
		_, err = nodeClient.Create(ctx, nodeWithoutTaint, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create nodeWithoutTaint: %v", err)
		}

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 1, t)
		validateDaemonSetStatus(dsClient, ds.Name, 1, t)

		// remove taint from nodeWithTaint
		nodeWithTaint, err = nodeClient.Get(ctx, "node-with-taint", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to retrieve nodeWithTaint: %v", err)
		}
		nodeWithTaintCopy := nodeWithTaint.DeepCopy()
		nodeWithTaintCopy.Spec.Taints = []v1.Taint{}
		_, err = nodeClient.Update(ctx, nodeWithTaintCopy, metav1.UpdateOptions{})
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
		ctx, closeFn, dc, informers, clientset := setup(t)
		defer closeFn()
		ns := framework.CreateNamespaceOrDie(clientset, "daemonset-unschedulable-test", t)
		defer framework.DeleteNamespaceOrDie(clientset, ns, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()

		informers.Start(ctx.Done())
		go dc.Run(ctx, 2)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		ds.Spec.Template.Spec.HostNetwork = true
		_, err := dsClient.Create(ctx, ds, metav1.CreateOptions{})
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

		_, err = nodeClient.Create(ctx, node, metav1.CreateOptions{})
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

		_, err = nodeClient.Create(ctx, nodeNU, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, 2, t)
		validateDaemonSetStatus(dsClient, ds.Name, 2, t)
	})
}

func TestUpdateStatusDespitePodCreationFailure(t *testing.T) {
	forEachStrategy(t, func(t *testing.T, strategy *apps.DaemonSetUpdateStrategy) {
		limitedPodNumber := 2
		ctx, closeFn, dc, informers, clientset := setupWithServerSetup(t, framework.TestServerSetup{
			ModifyServerConfig: func(config *controlplane.Config) {
				config.ControlPlane.Generic.AdmissionControl = &fakePodFailAdmission{
					limitedPodNumber: limitedPodNumber,
				}
			},
		})
		defer closeFn()
		ns := framework.CreateNamespaceOrDie(clientset, "update-status-despite-pod-failure", t)
		defer framework.DeleteNamespaceOrDie(clientset, ns, t)

		dsClient := clientset.AppsV1().DaemonSets(ns.Name)
		podClient := clientset.CoreV1().Pods(ns.Name)
		nodeClient := clientset.CoreV1().Nodes()
		podInformer := informers.Core().V1().Pods().Informer()

		informers.Start(ctx.Done())
		go dc.Run(ctx, 2)

		ds := newDaemonSet("foo", ns.Name)
		ds.Spec.UpdateStrategy = *strategy
		_, err := dsClient.Create(ctx, ds, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create DaemonSet: %v", err)
		}
		defer cleanupDaemonSets(t, clientset, ds)

		addNodes(nodeClient, 0, 5, nil, t)

		validateDaemonSetPodsAndMarkReady(podClient, podInformer, limitedPodNumber, t)
		validateDaemonSetStatus(dsClient, ds.Name, int32(limitedPodNumber), t)
	})
}

func TestDaemonSetRollingUpdateWithTolerations(t *testing.T) {
	var taints []v1.Taint
	var node *v1.Node
	var tolerations []v1.Toleration
	ctx, closeFn, dc, informers, clientset := setup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(clientset, "daemonset-rollingupdate-with-tolerations-test", t)
	defer framework.DeleteNamespaceOrDie(clientset, ns, t)

	dsClient := clientset.AppsV1().DaemonSets(ns.Name)
	podClient := clientset.CoreV1().Pods(ns.Name)
	nodeClient := clientset.CoreV1().Nodes()
	podInformer := informers.Core().V1().Pods().Informer()
	informers.Start(ctx.Done())
	go dc.Run(ctx, 2)

	zero := intstr.FromInt32(0)
	maxSurge := intstr.FromInt32(1)
	ds := newDaemonSet("foo", ns.Name)
	ds.Spec.UpdateStrategy = apps.DaemonSetUpdateStrategy{
		Type: apps.RollingUpdateDaemonSetStrategyType,
		RollingUpdate: &apps.RollingUpdateDaemonSet{
			MaxUnavailable: &zero,
			MaxSurge:       &maxSurge,
		},
	}

	// Add six nodes with zone-y, zone-z or common taint
	for i := 0; i < 6; i++ {
		if i < 2 {
			taints = []v1.Taint{
				{Key: "zone-y", Effect: v1.TaintEffectNoSchedule},
			}
		} else if i < 4 {
			taints = []v1.Taint{
				{Key: "zone-z", Effect: v1.TaintEffectNoSchedule},
			}
		} else {
			taints = []v1.Taint{
				{Key: "zone-common", Effect: v1.TaintEffectNoSchedule},
			}
		}
		node = newNode(fmt.Sprintf("node-%d", i), nil)
		node.Spec.Taints = taints
		_, err := nodeClient.Create(context.TODO(), node, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}
	}

	// Create DaemonSet with zone-y toleration
	tolerations = []v1.Toleration{
		{Key: "zone-y", Operator: v1.TolerationOpExists},
		{Key: "zone-common", Operator: v1.TolerationOpExists},
	}
	ds.Spec.Template.Spec.Tolerations = tolerations
	_, err := dsClient.Create(ctx, ds, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create DaemonSet: %v", err)
	}
	defer cleanupDaemonSets(t, clientset, ds)
	validateDaemonSetPodsActive(podClient, podInformer, 4, t)
	validateDaemonSetPodsAndMarkReady(podClient, podInformer, 4, t)
	validateDaemonSetStatus(dsClient, ds.Name, 4, t)
	validateUpdatedNumberScheduled(ctx, dsClient, ds.Name, 4, t)
	validateDaemonSetPodsTolerations(podClient, podInformer, tolerations, "zone-", t)

	// Update DaemonSet with zone-z toleration
	tolerations = []v1.Toleration{
		{Key: "zone-z", Operator: v1.TolerationOpExists},
		{Key: "zone-common", Operator: v1.TolerationOpExists},
	}
	ds.Spec.Template.Spec.Tolerations = tolerations
	_, err = dsClient.Update(ctx, ds, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update DaemonSet: %v", err)
	}

	// Expected numberPods of validateDaemonSetPodsActive is 7 when update DaemonSet
	// and before updated pods become ready because:
	//   - New 2 pods are created and Pending on Zone Z nodes
	//   - New 1 pod are created as surge and Pending on Zone Common node
	//   - Old 2 pods that violate scheduling constraints on Zone Y nodes will remain existing and Running
	//     until other new pods become available
	validateDaemonSetPodsActive(podClient, podInformer, 7, t)
	validateDaemonSetPodsAndMarkReady(podClient, podInformer, 4, t)
	validateDaemonSetStatus(dsClient, ds.Name, 4, t)
	validateUpdatedNumberScheduled(ctx, dsClient, ds.Name, 4, t)
	validateDaemonSetPodsTolerations(podClient, podInformer, tolerations, "zone-", t)

	// Update DaemonSet with zone-y and zone-z toleration
	tolerations = []v1.Toleration{
		{Key: "zone-y", Operator: v1.TolerationOpExists},
		{Key: "zone-z", Operator: v1.TolerationOpExists},
		{Key: "zone-common", Operator: v1.TolerationOpExists},
	}
	ds.Spec.Template.Spec.Tolerations = tolerations
	_, err = dsClient.Update(ctx, ds, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update DaemonSet: %v", err)
	}
	validateDaemonSetPodsActive(podClient, podInformer, 7, t)
	validateDaemonSetPodsAndMarkReady(podClient, podInformer, 6, t)
	validateDaemonSetStatus(dsClient, ds.Name, 6, t)
	validateUpdatedNumberScheduled(ctx, dsClient, ds.Name, 6, t)
	validateDaemonSetPodsTolerations(podClient, podInformer, tolerations, "zone-", t)

	// Update DaemonSet with no toleration
	ds.Spec.Template.Spec.Tolerations = nil
	_, err = dsClient.Update(ctx, ds, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update DaemonSet: %v", err)
	}
	validateDaemonSetPodsActive(podClient, podInformer, 0, t)
	validateDaemonSetStatus(dsClient, ds.Name, 0, t)
	validateUpdatedNumberScheduled(ctx, dsClient, ds.Name, 0, t)
}
