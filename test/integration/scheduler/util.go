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

package scheduler

import (
	"context"
	"fmt"
	"slices"
	"time"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/ktesting"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/scale"
	"k8s.io/klog/v2"
	configv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/disruption"
	"k8s.io/kubernetes/pkg/scheduler"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpreemption"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/integration/util"
	utiltesting "k8s.io/kubernetes/test/utils"
	"k8s.io/utils/ptr"
)

// ShutdownFunc represents the function handle to be called, typically in a defer handler, to shutdown a running module
type ShutdownFunc func()

// The returned shutdown func will delete created resources and scheduler, resources should be those
// that will affect the scheduling result, like nodes, pods, etc.. Namespaces should not be
// deleted here because it's created together with the apiserver, they should be deleted
// simultaneously or we'll have no namespace.
// This should only be called when you want to kill the scheduler alone, away from apiserver.
// For example, in scheduler integration tests, recreating apiserver is performance consuming,
// then shutdown the scheduler and recreate it between each test case is a better approach.
func InitTestSchedulerForFrameworkTest(t ktesting.TB, testCtx *testutils.TestContext, nodeCount int, runScheduler bool, opts ...scheduler.Option) (*testutils.TestContext, ShutdownFunc) {
	tCtx := ktesting.Init(t)
	testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, 0, opts...)
	testutils.SyncSchedulerInformerFactory(testCtx)
	if runScheduler {
		go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
	}

	if nodeCount > 0 {
		if _, err := CreateAndWaitForNodesInCache(testCtx, "test-node", st.MakeNode(), nodeCount); err != nil {
			// Make sure to cleanup the resources when initializing error.
			testutils.CleanupTest(tCtx, testCtx)
			t.Fatal(err)
		}
	}

	teardown := func() {
		err := testCtx.ClientSet.CoreV1().Nodes().DeleteCollection(testCtx.SchedulerCtx, *metav1.NewDeleteOptions(0), metav1.ListOptions{})
		if err != nil {
			t.Errorf("error while deleting all nodes: %v", err)
		}
		err = testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).DeleteCollection(testCtx.SchedulerCtx, *metav1.NewDeleteOptions(0), metav1.ListOptions{})
		if err != nil {
			t.Errorf("error while deleting pod: %v", err)
		}
		// Wait for all pods to be deleted, or will failed to create same name pods
		// required in other test cases.
		err = wait.PollUntilContextTimeout(testCtx.SchedulerCtx, time.Millisecond, wait.ForeverTestTimeout, true,
			PodsCleanedUp(testCtx.SchedulerCtx, testCtx.ClientSet, testCtx.NS.Name))
		if err != nil {
			t.Errorf("error while waiting for all pods to be deleted: %v", err)
		}
		// Kill the scheduler.
		testCtx.SchedulerCloseFn()
	}

	return testCtx, teardown
}

// NewPlugin returns a plugin factory with specified Plugin.
func NewPlugin(plugin fwk.Plugin) frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return plugin, nil
	}
}

// InitRegistryAndConfig returns registry and plugins config based on give plugins.
func InitRegistryAndConfig(t utiltesting.TB, factory func(plugin fwk.Plugin) frameworkruntime.PluginFactory, plugins ...fwk.Plugin) (frameworkruntime.Registry, schedulerconfig.KubeSchedulerProfile) {
	if len(plugins) == 0 {
		return frameworkruntime.Registry{}, schedulerconfig.KubeSchedulerProfile{}
	}

	if factory == nil {
		factory = NewPlugin
	}

	registry := frameworkruntime.Registry{}
	pls := &configv1.Plugins{}

	for _, p := range plugins {
		registry.Register(p.Name(), factory(p))
		plugin := configv1.Plugin{Name: p.Name()}

		switch p.(type) {
		case fwk.QueueSortPlugin:
			pls.QueueSort.Enabled = append(pls.QueueSort.Enabled, plugin)
			// It's intentional to disable the PrioritySort plugin.
			pls.QueueSort.Disabled = []configv1.Plugin{{Name: queuesort.Name}}
		case fwk.PreEnqueuePlugin:
			pls.PreEnqueue.Enabled = append(pls.PreEnqueue.Enabled, plugin)
		case fwk.PreFilterPlugin:
			pls.PreFilter.Enabled = append(pls.PreFilter.Enabled, plugin)
		case fwk.FilterPlugin:
			pls.Filter.Enabled = append(pls.Filter.Enabled, plugin)
		case fwk.PreScorePlugin:
			pls.PreScore.Enabled = append(pls.PreScore.Enabled, plugin)
		case fwk.ScorePlugin:
			pls.Score.Enabled = append(pls.Score.Enabled, plugin)
		case fwk.ReservePlugin:
			pls.Reserve.Enabled = append(pls.Reserve.Enabled, plugin)
		case fwk.PreBindPlugin:
			pls.PreBind.Enabled = append(pls.PreBind.Enabled, plugin)
		case fwk.BindPlugin:
			pls.Bind.Enabled = append(pls.Bind.Enabled, plugin)
			// It's intentional to disable the DefaultBind plugin. Otherwise, DefaultBinder's failure would fail
			// a pod's scheduling, as well as the test BindPlugin's execution.
			pls.Bind.Disabled = []configv1.Plugin{{Name: defaultbinder.Name}}
		case fwk.PostBindPlugin:
			pls.PostBind.Enabled = append(pls.PostBind.Enabled, plugin)
		case fwk.PermitPlugin:
			pls.Permit.Enabled = append(pls.Permit.Enabled, plugin)
		}
	}

	versionedCfg := configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
			Plugins:       pls,
		}},
	}
	cfg := configtesting.V1ToInternalWithDefaults(t, versionedCfg)
	return registry, cfg.Profiles[0]
}

// PodsCleanedUp returns true if all pods are deleted in the specific namespace.
func PodsCleanedUp(ctx context.Context, c clientset.Interface, namespace string) wait.ConditionWithContextFunc {
	return func(context.Context) (bool, error) {
		list, err := c.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		return len(list.Items) == 0, nil
	}
}

// CleanupPodGroups deletes the given pod groups and waits for them to be actually deleted.
func CleanupPodGroups(ctx context.Context, cs clientset.Interface, t utiltesting.TB, podGroups []*schedulingv1beta1.PodGroup) {
	for _, pg := range podGroups {
		patch := []byte(`{"metadata":{"finalizers":null}}`)
		if _, err := cs.SchedulingV1beta1().PodGroups(pg.Namespace).Patch(ctx, pg.Name, types.MergePatchType, patch, metav1.PatchOptions{}); err != nil && !apierrors.IsNotFound(err) {
			t.Errorf("error while removing finalizers from podgroup %v/%v: %v", pg.Namespace, pg.Name, err)
		}
		if err := cs.SchedulingV1beta1().PodGroups(pg.Namespace).Delete(ctx, pg.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			t.Errorf("error while deleting podgroup %v/%v: %v", pg.Namespace, pg.Name, err)
		}
	}
	for _, pg := range podGroups {
		if err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
			_, err := cs.SchedulingV1beta1().PodGroups(pg.Namespace).Get(ctx, pg.Name, metav1.GetOptions{})
			return apierrors.IsNotFound(err), nil
		}); err != nil {
			t.Errorf("error while waiting for podgroup %v/%v to get deleted: %v", pg.Namespace, pg.Name, err)
		}
	}
}

// CleanupCompositePodGroups deletes the given composite pod groups and waits for them to be actually deleted.
func CleanupCompositePodGroups(ctx context.Context, cs clientset.Interface, t utiltesting.TB, compositePodGroups []*schedulingv1beta1.CompositePodGroup) {
	for _, cpg := range compositePodGroups {
		patch := []byte(`{"metadata":{"finalizers":null}}`)
		if _, err := cs.SchedulingV1beta1().CompositePodGroups(cpg.Namespace).Patch(ctx, cpg.Name, types.MergePatchType, patch, metav1.PatchOptions{}); err != nil && !apierrors.IsNotFound(err) {
			t.Errorf("error while removing finalizers from compositepodgroup %v/%v: %v", cpg.Namespace, cpg.Name, err)
		}
		if err := cs.SchedulingV1beta1().CompositePodGroups(cpg.Namespace).Delete(ctx, cpg.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			t.Errorf("error while deleting compositepodgroup %v/%v: %v", cpg.Namespace, cpg.Name, err)
		}
	}
	for _, cpg := range compositePodGroups {
		if err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
			_, err := cs.SchedulingV1beta1().CompositePodGroups(cpg.Namespace).Get(ctx, cpg.Name, metav1.GetOptions{})
			return apierrors.IsNotFound(err), nil
		}); err != nil {
			t.Errorf("error while waiting for compositepodgroup %v/%v to get deleted: %v", cpg.Namespace, cpg.Name, err)
		}
	}
}

// WaitForSchedulerCacheCleanup waits for cleanup of scheduler's cache to complete
func WaitForSchedulerCacheCleanup(ctx context.Context, sched *scheduler.Scheduler, t utiltesting.TB) {
	schedulerCacheIsEmpty := func(context.Context) (bool, error) {
		dump := sched.Cache.Dump()

		return len(dump.Nodes) == 0 && len(dump.AssumedPods) == 0, nil
	}

	if err := wait.PollUntilContextTimeout(ctx, time.Second, wait.ForeverTestTimeout, false, schedulerCacheIsEmpty); err != nil {
		t.Errorf("Failed to wait for scheduler cache cleanup: %v", err)
	}
}

// WithNewNamespace creates a child TestContext that shares the API server from
// parent but gets a fresh namespace. Only the namespace is deleted on t.Cleanup;
// the API server lifecycle is managed by the caller. This is useful when
// multiple subtests share one API server to avoid the per-subtest startup cost.
func WithNewNamespace(t utiltesting.TB, parent *testutils.TestContext, nsPrefix string) *testutils.TestContext {
	t.Helper()
	child := &testutils.TestContext{
		ClientSet:  parent.ClientSet,
		KubeConfig: parent.KubeConfig,
		Ctx:        parent.Ctx,
		CloseFn:    func() {}, // API server is owned by parent; do not tear it down here.
	}
	child.NS = framework.CreateNamespaceOrDie(child.ClientSet, nsPrefix+string(uuid.NewUUID()), t)
	t.Cleanup(func() {
		framework.DeleteNamespaceOrDie(child.ClientSet, child.NS, t)
	})
	return child
}

// InitTestScheduler initializes a test environment and creates a scheduler with default
// configuration.
func InitTestScheduler(
	t utiltesting.TB,
	testCtx *testutils.TestContext,
) *testutils.TestContext {
	// Pod preemption is enabled by default scheduler configuration.
	return testutils.InitTestSchedulerWithOptions(t, testCtx, 0)
}

// WaitForPodToScheduleWithTimeout waits for a pod to get scheduled and returns
// an error if it does not scheduled within the given timeout.
func WaitForPodToScheduleWithTimeout(ctx context.Context, cs clientset.Interface, pod *v1.Pod, timeout time.Duration) error {
	return wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, timeout, false, PodScheduled(cs, pod.Namespace, pod.Name))
}

// WaitForPodToSchedule waits for a pod to get scheduled and returns an error if
// it does not get scheduled within the timeout duration (30 seconds).
func WaitForPodToSchedule(ctx context.Context, cs clientset.Interface, pod *v1.Pod) error {
	return WaitForPodToScheduleWithTimeout(ctx, cs, pod, 30*time.Second)
}

// PodScheduled checks if the pod has been scheduled
func PodScheduled(c clientset.Interface, podNamespace, podName string) wait.ConditionWithContextFunc {
	return func(ctx context.Context) (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		if pod.Spec.NodeName == "" {
			return false, nil
		}
		return true, nil
	}
}

// InitDisruptionController initializes and runs a Disruption Controller to properly
// update PodDisuptionBudget objects.
func InitDisruptionController(t utiltesting.TB, testCtx *testutils.TestContext) *disruption.DisruptionController {
	informers := informers.NewSharedInformerFactory(testCtx.ClientSet, 12*time.Hour)

	discoveryClient := cacheddiscovery.NewMemCacheClient(testCtx.ClientSet.Discovery())
	mapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)

	config := restclient.CopyConfig(testCtx.KubeConfig)
	scaleKindResolver := scale.NewDiscoveryScaleKindResolver(testCtx.ClientSet.Discovery())
	scaleClient, err := scale.NewForConfig(config, mapper, dynamic.LegacyAPIPathResolverFunc, scaleKindResolver)
	if err != nil {
		t.Fatalf("Error in create scaleClient: %v", err)
	}

	dc := disruption.NewDisruptionController(
		testCtx.Ctx,
		informers.Core().V1().Pods(),
		informers.Policy().V1().PodDisruptionBudgets(),
		informers.Core().V1().ReplicationControllers(),
		informers.Apps().V1().ReplicaSets(),
		informers.Apps().V1().Deployments(),
		informers.Apps().V1().StatefulSets(),
		testCtx.ClientSet,
		mapper,
		scaleClient,
		testCtx.ClientSet.Discovery())

	informers.Start(testCtx.Scheduler.StopEverything)
	informers.WaitForCacheSync(testCtx.Scheduler.StopEverything)
	go dc.Run(testCtx.Ctx, 1)
	return dc
}

// InitTestSchedulerWithNS initializes a test environment and creates API server and scheduler with default
// configuration.
func InitTestSchedulerWithNS(t ktesting.TB, nsPrefix string, opts ...scheduler.Option) *testutils.TestContext {
	testCtx := testutils.InitTestSchedulerWithOptions(t, testutils.InitTestAPIServer(t, nsPrefix, nil), 0, opts...)
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
	return testCtx
}

// InitTestDisablePreemption initializes a test environment and creates API server and scheduler with default
// configuration but with pod preemption disabled.
func InitTestDisablePreemption(t ktesting.TB, nsPrefix string) *testutils.TestContext {
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				MultiPoint: configv1.PluginSet{
					Disabled: []configv1.Plugin{
						{Name: defaultpreemption.Name},
					},
				},
			},
		}},
	})
	testCtx := testutils.InitTestSchedulerWithOptions(
		t, testutils.InitTestAPIServer(t, nsPrefix, nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...))
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
	return testCtx
}

// WaitForReflection waits till the passFunc confirms that the object it expects
// to see is in the store. Used to observe reflected events.
func WaitForReflection(ctx context.Context, t utiltesting.TB, nodeLister corelisters.NodeLister, key string,
	passFunc func(n interface{}) bool) error {
	var nodes []*v1.Node
	err := wait.PollUntilContextTimeout(ctx, time.Millisecond*100, wait.ForeverTestTimeout, false, func(context.Context) (bool, error) {
		n, err := nodeLister.Get(key)

		switch {
		case err == nil && passFunc(n):
			return true, nil
		case apierrors.IsNotFound(err):
			nodes = append(nodes, nil)
		case err != nil:
			t.Errorf("Unexpected error: %v", err)
		default:
			nodes = append(nodes, n)
		}

		return false, nil
	})
	if err != nil {
		t.Logf("Logging consecutive node versions received from store:")
		for i, n := range nodes {
			t.Logf("%d: %#v", i, n)
		}
	}
	return err
}

func UpdateNode(cs clientset.Interface, node *v1.Node) (*v1.Node, error) {
	return cs.CoreV1().Nodes().Update(context.TODO(), node, metav1.UpdateOptions{})
}

func CreateNode(cs clientset.Interface, node *v1.Node) (*v1.Node, error) {
	return cs.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
}

// RemoveTaintOffNode removes a specific taint from a node
func RemoveTaintOffNode(cs clientset.Interface, nodeName string, taint v1.Taint) error {
	node, err := cs.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	var taints []v1.Taint
	for _, t := range node.Spec.Taints {
		if !t.MatchTaint(&taint) {
			taints = append(taints, t)
		}
	}
	node.Spec.Taints = taints
	_, err = cs.CoreV1().Nodes().Update(context.TODO(), node, metav1.UpdateOptions{})
	return err
}

func createNodes(cs clientset.Interface, prefix string, wrapper *st.NodeWrapper, numNodes int) ([]*v1.Node, error) {
	nodes := make([]*v1.Node, numNodes)
	for i := range numNodes {
		nodeName := fmt.Sprintf("%v-%d", prefix, i)
		node, err := CreateNode(cs, wrapper.Name(nodeName).Label("kubernetes.io/hostname", nodeName).Obj())
		if err != nil {
			return nodes, err
		}
		nodes[i] = node
	}
	return nodes, nil
}

// CreateAndWaitForNodesInCache calls createNodes(), and wait for the created
// nodes to be present in scheduler cache.
func CreateAndWaitForNodesInCache(testCtx *testutils.TestContext, prefix string, wrapper *st.NodeWrapper, numNodes int) ([]*v1.Node, error) {
	existingNodes := testCtx.Scheduler.Cache.NodeCount()
	nodes, err := createNodes(testCtx.ClientSet, prefix, wrapper, numNodes)
	if err != nil {
		return nodes, fmt.Errorf("cannot create nodes: %w", err)
	}
	return nodes, WaitForNodesInCache(testCtx.Ctx, testCtx.Scheduler, numNodes+existingNodes)
}

// WaitForNodesInCache ensures at least <nodeCount> nodes are present in scheduler cache
// within 30 seconds; otherwise returns false.
func WaitForNodesInCache(ctx context.Context, sched *scheduler.Scheduler, nodeCount int) error {
	err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false, func(context.Context) (bool, error) {
		return sched.Cache.NodeCount() >= nodeCount, nil
	})
	if err != nil {
		return fmt.Errorf("cannot obtain available nodes in scheduler cache: %w", err)
	}
	return nil
}

// CreatePausePodWithResource creates a pod with "Pause" image and the given
// resources and returns its pointer and error status. The resource list can be
// nil.
func CreatePausePodWithResource(cs clientset.Interface, podName string,
	nsName string, res *v1.ResourceList) (*v1.Pod, error) {
	var conf testutils.PausePodConfig
	if res == nil {
		conf = testutils.PausePodConfig{
			Name:      podName,
			Namespace: nsName,
		}
	} else {
		conf = testutils.PausePodConfig{
			Name:      podName,
			Namespace: nsName,
			Resources: &v1.ResourceRequirements{
				Requests: *res,
			},
		}
	}
	return testutils.CreatePausePod(cs, testutils.InitPausePod(&conf))
}

// CreatePVC creates a PersistentVolumeClaim with the given config and returns
// its pointer and error status.
func CreatePVC(cs clientset.Interface, pvc *v1.PersistentVolumeClaim) (*v1.PersistentVolumeClaim, error) {
	return cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(context.TODO(), pvc, metav1.CreateOptions{})
}

// CreatePV creates a PersistentVolume with the given config and returns its
// pointer and error status.
func CreatePV(cs clientset.Interface, pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	return cs.CoreV1().PersistentVolumes().Create(context.TODO(), pv, metav1.CreateOptions{})
}

// DeletePVC deletes the given PVC in the given namespace.
func DeletePVC(cs clientset.Interface, pvcName string, nsName string) error {
	return cs.CoreV1().PersistentVolumeClaims(nsName).Delete(context.TODO(), pvcName, *metav1.NewDeleteOptions(0))
}

// DeletePV deletes the given PV in the given namespace.
func DeletePV(cs clientset.Interface, pvName string) error {
	return cs.CoreV1().PersistentVolumes().Delete(context.TODO(), pvName, *metav1.NewDeleteOptions(0))
}

// RunPausePod creates a pod with "Pause" image and the given config and waits
// until it is scheduled. It returns its pointer and error status.
func RunPausePod(cs clientset.Interface, pod *v1.Pod) (*v1.Pod, error) {
	pod, err := cs.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create pause pod: %w", err)
	}
	if err = WaitForPodToSchedule(context.TODO(), cs, pod); err != nil {
		return pod, fmt.Errorf("pod %v/%v didn't schedule successfully. Error: %w", pod.Namespace, pod.Name, err)
	}
	if pod, err = cs.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{}); err != nil {
		return pod, fmt.Errorf("failed to get pod %v/%v info: %w", pod.Namespace, pod.Name, err)
	}
	return pod, nil
}

type PodWithContainersConfig struct {
	Name       string
	Namespace  string
	Containers []v1.Container
}

// InitPodWithContainers initializes a pod API object from the given config. This is used primarily for generating
// pods with containers each having a specific image.
func InitPodWithContainers(cs clientset.Interface, conf *PodWithContainersConfig) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      conf.Name,
			Namespace: conf.Namespace,
		},
		Spec: v1.PodSpec{
			Containers: conf.Containers,
		},
	}
	return pod
}

// RunPodWithContainers creates a pod with given config and containers and waits
// until it is scheduled. It returns its pointer and error status.
func RunPodWithContainers(cs clientset.Interface, pod *v1.Pod) (*v1.Pod, error) {
	pod, err := cs.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create pod-with-containers: %w", err)
	}
	if err = WaitForPodToSchedule(context.TODO(), cs, pod); err != nil {
		return pod, fmt.Errorf("pod %v didn't schedule successfully. Error: %w", pod.Name, err)
	}
	if pod, err = cs.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{}); err != nil {
		return pod, fmt.Errorf("failed to get pod %v info: %w", pod.Name, err)
	}
	return pod, nil
}

// PodScheduledIn returns true if a given pod is placed onto one of the expected nodes.
func PodScheduledIn(c clientset.Interface, podNamespace, podName string, nodeNames []string) wait.ConditionWithContextFunc {
	return func(ctx context.Context) (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		if pod.Spec.NodeName == "" {
			return false, nil
		}
		return slices.Contains(nodeNames, pod.Spec.NodeName), nil
	}
}

// PodUnschedulable returns a condition function that returns true if the given pod
// gets unschedulable status of reason 'Unschedulable'.
func PodUnschedulable(c clientset.Interface, podNamespace, podName string) wait.ConditionWithContextFunc {
	return func(ctx context.Context) (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
		return cond != nil && cond.Status == v1.ConditionFalse &&
			cond.Reason == v1.PodReasonUnschedulable && pod.Spec.NodeName == "", nil
	}
}

// PodSchedulingError returns a condition function that returns true if the given pod
// gets unschedulable status for reasons other than "Unschedulable". The scheduler
// records such reasons in case of error.
func PodSchedulingError(c clientset.Interface, podNamespace, podName string) wait.ConditionWithContextFunc {
	return func(ctx context.Context) (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
		return cond != nil && cond.Status == v1.ConditionFalse &&
			cond.Reason != v1.PodReasonUnschedulable, nil
	}
}

// PodSchedulingGated returns a condition function that returns true if the given pod
// gets unschedulable status of reason 'SchedulingGated'.
func PodSchedulingGated(ctx context.Context, c clientset.Interface, podNamespace, podName string) wait.ConditionWithContextFunc {
	return func(context.Context) (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
		return cond != nil && cond.Status == v1.ConditionFalse &&
			cond.Reason == v1.PodReasonSchedulingGated && pod.Spec.NodeName == "", nil
	}
}

// WaitForPodUnschedulableWithTimeout waits for a pod to fail scheduling and returns
// an error if it does not become unschedulable within the given timeout.
func WaitForPodUnschedulableWithTimeout(ctx context.Context, cs clientset.Interface, pod *v1.Pod, timeout time.Duration) error {
	return wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, timeout, false, PodUnschedulable(cs, pod.Namespace, pod.Name))
}

// WaitForPodUnschedulable waits for a pod to fail scheduling and returns
// an error if it does not become unschedulable within the timeout duration (30 seconds).
func WaitForPodUnschedulable(ctx context.Context, cs clientset.Interface, pod *v1.Pod) error {
	return WaitForPodUnschedulableWithTimeout(ctx, cs, pod, 30*time.Second)
}

// WaitForPodSchedulingGated waits for a pod to be in scheduling gated state
// and returns an error if it does not fall into this state within the given timeout.
func WaitForPodSchedulingGated(ctx context.Context, cs clientset.Interface, pod *v1.Pod, timeout time.Duration) error {
	return wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, timeout, false, PodSchedulingGated(ctx, cs, pod.Namespace, pod.Name))
}

// WaitForPDBsStable waits for PDBs to have "CurrentHealthy" status equal to
// the expected values.
func WaitForPDBsStable(testCtx *testutils.TestContext, pdbs []*policy.PodDisruptionBudget, pdbPodNum []int32) error {
	return wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, 60*time.Second, false, func(context.Context) (bool, error) {
		pdbList, err := testCtx.ClientSet.PolicyV1().PodDisruptionBudgets(testCtx.NS.Name).List(testCtx.Ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		if len(pdbList.Items) != len(pdbs) {
			return false, nil
		}
		for i, pdb := range pdbs {
			found := false
			for _, cpdb := range pdbList.Items {
				if pdb.Name == cpdb.Name && pdb.Namespace == cpdb.Namespace {
					found = true
					if cpdb.Status.CurrentHealthy != pdbPodNum[i] {
						return false, nil
					}
				}
			}
			if !found {
				return false, nil
			}
		}
		return true, nil
	})
}

// WaitCachedPodsStable waits until scheduler cache has the given pods.
func WaitCachedPodsStable(testCtx *testutils.TestContext, pods []*v1.Pod) error {
	return wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, 30*time.Second, false, func(context.Context) (bool, error) {
		cachedPods, err := testCtx.Scheduler.Cache.PodCount()
		if err != nil {
			return false, err
		}
		if len(pods) != cachedPods {
			return false, nil
		}
		for _, p := range pods {
			actualPod, err1 := testCtx.ClientSet.CoreV1().Pods(p.Namespace).Get(testCtx.Ctx, p.Name, metav1.GetOptions{})
			if err1 != nil {
				return false, err1
			}
			cachedPod, err2 := testCtx.Scheduler.Cache.GetPod(actualPod)
			if err2 != nil || cachedPod == nil {
				return false, err2
			}
		}
		return true, nil
	})
}

// DeletePod deletes the given pod in the given namespace.
func DeletePod(cs clientset.Interface, podName string, nsName string) error {
	return cs.CoreV1().Pods(nsName).Delete(context.TODO(), podName, *metav1.NewDeleteOptions(0))
}

func GetPod(cs clientset.Interface, podName string, podNamespace string) (*v1.Pod, error) {
	return cs.CoreV1().Pods(podNamespace).Get(context.TODO(), podName, metav1.GetOptions{})
}

func CreateNamespacesWithLabels(cs clientset.Interface, namespaces []string, labels map[string]string) error {
	for _, n := range namespaces {
		ns := v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: n, Labels: labels}}
		if _, err := cs.CoreV1().Namespaces().Create(context.TODO(), &ns, metav1.CreateOptions{}); err != nil {
			return err
		}
	}
	return nil
}

// timeout returns a timeout error if the given `f` function doesn't
// complete within `d` duration; otherwise it returns nil.
func timeout(ctx context.Context, d time.Duration, f func()) error {
	ctx, cancel := context.WithTimeout(ctx, d)
	defer cancel()

	done := make(chan struct{})
	go func() {
		f()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// NextEntityOrDie returns the next entity (either Pod or PodGroup) in the scheduler queue.
// The operation needs to be completed within 5 seconds; otherwise the test gets aborted.
func NextEntityOrDie(t utiltesting.TB, testCtx *testutils.TestContext) schedulerframework.QueuedEntityInfo {
	t.Helper()

	var entity schedulerframework.QueuedEntityInfo
	logger := klog.FromContext(testCtx.Ctx)
	// NextEntity() is a blocking operation. Wrap it in timeout() to avoid relying on
	// default go testing timeout (10m) to abort.
	if err := timeout(testCtx.Ctx, time.Second*5, func() {
		entity, _ = testCtx.Scheduler.NextEntity(logger)
	}); err != nil {
		t.Fatalf("Timed out waiting for the entity to be popped: %v", err)
	}
	return entity
}

func WaitForNominatedNodeNameWithTimeout(ctx context.Context, cs clientset.Interface, pod *v1.Pod, timeout time.Duration) error {
	if err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, timeout, false, func(ctx context.Context) (bool, error) {
		pod, err := cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if len(pod.Status.NominatedNodeName) > 0 {
			return true, nil
		}
		return false, err
	}); err != nil {
		return fmt.Errorf(".status.nominatedNodeName of Pod %v/%v did not get set: %w", pod.Namespace, pod.Name, err)
	}
	return nil
}

func WaitForNominatedNodeName(ctx context.Context, cs clientset.Interface, pod *v1.Pod) error {
	return WaitForNominatedNodeNameWithTimeout(ctx, cs, pod, wait.ForeverTestTimeout)
}
