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
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/scale"
	"k8s.io/kube-scheduler/config/v1beta3"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/disruption"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpreemption"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/pointer"
)

// initDisruptionController initializes and runs a Disruption Controller to properly
// update PodDisuptionBudget objects.
func initDisruptionController(t *testing.T, testCtx *testutils.TestContext) *disruption.DisruptionController {
	informers := informers.NewSharedInformerFactory(testCtx.ClientSet, 12*time.Hour)

	discoveryClient := cacheddiscovery.NewMemCacheClient(testCtx.ClientSet.Discovery())
	mapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)

	config := restclient.Config{Host: testCtx.HTTPServer.URL}
	scaleKindResolver := scale.NewDiscoveryScaleKindResolver(testCtx.ClientSet.Discovery())
	scaleClient, err := scale.NewForConfig(&config, mapper, dynamic.LegacyAPIPathResolverFunc, scaleKindResolver)
	if err != nil {
		t.Fatalf("Error in create scaleClient: %v", err)
	}

	dc := disruption.NewDisruptionController(
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
	go dc.Run(testCtx.Ctx)
	return dc
}

// initTest initializes a test environment and creates API server and scheduler with default
// configuration.
func initTest(t *testing.T, nsPrefix string, opts ...scheduler.Option) *testutils.TestContext {
	testCtx := testutils.InitTestSchedulerWithOptions(t, testutils.InitTestAPIServer(t, nsPrefix, nil), opts...)
	testutils.SyncInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)
	return testCtx
}

// initTestDisablePreemption initializes a test environment and creates API server and scheduler with default
// configuration but with pod preemption disabled.
func initTestDisablePreemption(t *testing.T, nsPrefix string) *testutils.TestContext {
	cfg := configtesting.V1beta3ToInternalWithDefaults(t, v1beta3.KubeSchedulerConfiguration{
		Profiles: []v1beta3.KubeSchedulerProfile{{
			SchedulerName: pointer.StringPtr(v1.DefaultSchedulerName),
			Plugins: &v1beta3.Plugins{
				PostFilter: v1beta3.PluginSet{
					Disabled: []v1beta3.Plugin{
						{Name: defaultpreemption.Name},
					},
				},
			},
		}},
	})
	testCtx := testutils.InitTestSchedulerWithOptions(
		t, testutils.InitTestAPIServer(t, nsPrefix, nil),
		scheduler.WithProfiles(cfg.Profiles...))
	testutils.SyncInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)
	return testCtx
}

// waitForReflection waits till the passFunc confirms that the object it expects
// to see is in the store. Used to observe reflected events.
func waitForReflection(t *testing.T, nodeLister corelisters.NodeLister, key string,
	passFunc func(n interface{}) bool) error {
	var nodes []*v1.Node
	err := wait.Poll(time.Millisecond*100, wait.ForeverTestTimeout, func() (bool, error) {
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

func updateNode(cs clientset.Interface, node *v1.Node) (*v1.Node, error) {
	return cs.CoreV1().Nodes().Update(context.TODO(), node, metav1.UpdateOptions{})
}

func createNode(cs clientset.Interface, node *v1.Node) (*v1.Node, error) {
	return cs.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
}

// createNodes creates `numNodes` nodes. The created node names will be in the
// form of "`prefix`-X" where X is an ordinal.
// DEPRECATED
// use createAndWaitForNodesInCache instead, which ensures the created nodes
// to be present in scheduler cache.
func createNodes(cs clientset.Interface, prefix string, wrapper *st.NodeWrapper, numNodes int) ([]*v1.Node, error) {
	nodes := make([]*v1.Node, numNodes)
	for i := 0; i < numNodes; i++ {
		nodeName := fmt.Sprintf("%v-%d", prefix, i)
		node, err := createNode(cs, wrapper.Name(nodeName).Obj())
		if err != nil {
			return nodes[:], err
		}
		nodes[i] = node
	}
	return nodes[:], nil
}

// createAndWaitForNodesInCache calls createNodes(), and wait for the created
// nodes to be present in scheduler cache.
func createAndWaitForNodesInCache(testCtx *testutils.TestContext, prefix string, wrapper *st.NodeWrapper, numNodes int) ([]*v1.Node, error) {
	existingNodes := testCtx.Scheduler.Cache.NodeCount()
	nodes, err := createNodes(testCtx.ClientSet, prefix, wrapper, numNodes)
	if err != nil {
		return nodes, fmt.Errorf("cannot create nodes: %v", err)
	}
	return nodes, waitForNodesInCache(testCtx.Scheduler, numNodes+existingNodes)
}

// waitForNodesInCache ensures at least <nodeCount> nodes are present in scheduler cache
// within 30 seconds; otherwise returns false.
func waitForNodesInCache(sched *scheduler.Scheduler, nodeCount int) error {
	err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		return sched.Cache.NodeCount() >= nodeCount, nil
	})
	if err != nil {
		return fmt.Errorf("cannot obtain available nodes in scheduler cache: %v", err)
	}
	return nil
}

type pausePodConfig struct {
	Name                              string
	Namespace                         string
	Affinity                          *v1.Affinity
	Annotations, Labels, NodeSelector map[string]string
	Resources                         *v1.ResourceRequirements
	Tolerations                       []v1.Toleration
	NodeName                          string
	SchedulerName                     string
	Priority                          *int32
	PreemptionPolicy                  *v1.PreemptionPolicy
	PriorityClassName                 string
}

// initPausePod initializes a pod API object from the given config. It is used
// mainly in pod creation process.
func initPausePod(conf *pausePodConfig) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        conf.Name,
			Namespace:   conf.Namespace,
			Labels:      conf.Labels,
			Annotations: conf.Annotations,
		},
		Spec: v1.PodSpec{
			NodeSelector: conf.NodeSelector,
			Affinity:     conf.Affinity,
			Containers: []v1.Container{
				{
					Name:  conf.Name,
					Image: imageutils.GetPauseImageName(),
				},
			},
			Tolerations:       conf.Tolerations,
			NodeName:          conf.NodeName,
			SchedulerName:     conf.SchedulerName,
			Priority:          conf.Priority,
			PreemptionPolicy:  conf.PreemptionPolicy,
			PriorityClassName: conf.PriorityClassName,
		},
	}
	if conf.Resources != nil {
		pod.Spec.Containers[0].Resources = *conf.Resources
	}
	return pod
}

// createPausePod creates a pod with "Pause" image and the given config and
// return its pointer and error status.
func createPausePod(cs clientset.Interface, p *v1.Pod) (*v1.Pod, error) {
	return cs.CoreV1().Pods(p.Namespace).Create(context.TODO(), p, metav1.CreateOptions{})
}

// createPausePodWithResource creates a pod with "Pause" image and the given
// resources and returns its pointer and error status. The resource list can be
// nil.
func createPausePodWithResource(cs clientset.Interface, podName string,
	nsName string, res *v1.ResourceList) (*v1.Pod, error) {
	var conf pausePodConfig
	if res == nil {
		conf = pausePodConfig{
			Name:      podName,
			Namespace: nsName,
		}
	} else {
		conf = pausePodConfig{
			Name:      podName,
			Namespace: nsName,
			Resources: &v1.ResourceRequirements{
				Requests: *res,
			},
		}
	}
	return createPausePod(cs, initPausePod(&conf))
}

// runPausePod creates a pod with "Pause" image and the given config and waits
// until it is scheduled. It returns its pointer and error status.
func runPausePod(cs clientset.Interface, pod *v1.Pod) (*v1.Pod, error) {
	pod, err := cs.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create pause pod: %v", err)
	}
	if err = testutils.WaitForPodToSchedule(cs, pod); err != nil {
		return pod, fmt.Errorf("Pod %v/%v didn't schedule successfully. Error: %v", pod.Namespace, pod.Name, err)
	}
	if pod, err = cs.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{}); err != nil {
		return pod, fmt.Errorf("failed to get pod %v/%v info: %v", pod.Namespace, pod.Name, err)
	}
	return pod, nil
}

type podWithContainersConfig struct {
	Name       string
	Namespace  string
	Containers []v1.Container
}

// initPodWithContainers initializes a pod API object from the given config. This is used primarily for generating
// pods with containers each having a specific image.
func initPodWithContainers(cs clientset.Interface, conf *podWithContainersConfig) *v1.Pod {
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

// runPodWithContainers creates a pod with given config and containers and waits
// until it is scheduled. It returns its pointer and error status.
func runPodWithContainers(cs clientset.Interface, pod *v1.Pod) (*v1.Pod, error) {
	pod, err := cs.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create pod-with-containers: %v", err)
	}
	if err = testutils.WaitForPodToSchedule(cs, pod); err != nil {
		return pod, fmt.Errorf("Pod %v didn't schedule successfully. Error: %v", pod.Name, err)
	}
	if pod, err = cs.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{}); err != nil {
		return pod, fmt.Errorf("failed to get pod %v info: %v", pod.Name, err)
	}
	return pod, nil
}

// podIsGettingEvicted returns true if the pod's deletion timestamp is set.
func podIsGettingEvicted(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if pod.DeletionTimestamp != nil {
			return true, nil
		}
		return false, nil
	}
}

// podScheduledIn returns true if a given pod is placed onto one of the expected nodes.
func podScheduledIn(c clientset.Interface, podNamespace, podName string, nodeNames []string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		if pod.Spec.NodeName == "" {
			return false, nil
		}
		for _, nodeName := range nodeNames {
			if pod.Spec.NodeName == nodeName {
				return true, nil
			}
		}
		return false, nil
	}
}

// podUnschedulable returns a condition function that returns true if the given pod
// gets unschedulable status.
func podUnschedulable(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
		return cond != nil && cond.Status == v1.ConditionFalse &&
			cond.Reason == v1.PodReasonUnschedulable && pod.Spec.NodeName == "", nil
	}
}

// podSchedulingError returns a condition function that returns true if the given pod
// gets unschedulable status for reasons other than "Unschedulable". The scheduler
// records such reasons in case of error.
func podSchedulingError(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
		return cond != nil && cond.Status == v1.ConditionFalse &&
			cond.Reason != v1.PodReasonUnschedulable, nil
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
	return waitForPodUnschedulableWithTimeout(cs, pod, 30*time.Second)
}

// waitForPodToScheduleWithTimeout waits for a pod to get scheduled and returns
// an error if it does not scheduled within the given timeout.
func waitForPodToScheduleWithTimeout(cs clientset.Interface, pod *v1.Pod, timeout time.Duration) error {
	return wait.Poll(100*time.Millisecond, timeout, podScheduled(cs, pod.Namespace, pod.Name))
}

// waitForPDBsStable waits for PDBs to have "CurrentHealthy" status equal to
// the expected values.
func waitForPDBsStable(testCtx *testutils.TestContext, pdbs []*policy.PodDisruptionBudget, pdbPodNum []int32) error {
	return wait.Poll(time.Second, 60*time.Second, func() (bool, error) {
		pdbList, err := testCtx.ClientSet.PolicyV1beta1().PodDisruptionBudgets(testCtx.NS.Name).List(context.TODO(), metav1.ListOptions{})
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

// waitCachedPodsStable waits until scheduler cache has the given pods.
func waitCachedPodsStable(testCtx *testutils.TestContext, pods []*v1.Pod) error {
	return wait.Poll(time.Second, 30*time.Second, func() (bool, error) {
		cachedPods, err := testCtx.Scheduler.Cache.PodCount()
		if err != nil {
			return false, err
		}
		if len(pods) != cachedPods {
			return false, nil
		}
		for _, p := range pods {
			actualPod, err1 := testCtx.ClientSet.CoreV1().Pods(p.Namespace).Get(context.TODO(), p.Name, metav1.GetOptions{})
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

// deletePod deletes the given pod in the given namespace.
func deletePod(cs clientset.Interface, podName string, nsName string) error {
	return cs.CoreV1().Pods(nsName).Delete(context.TODO(), podName, *metav1.NewDeleteOptions(0))
}

func getPod(cs clientset.Interface, podName string, podNamespace string) (*v1.Pod, error) {
	return cs.CoreV1().Pods(podNamespace).Get(context.TODO(), podName, metav1.GetOptions{})
}

// podScheduled returns true if a node is assigned to the given pod.
func podScheduled(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		return pod.Spec.NodeName != "", nil
	}
}

func createNamespacesWithLabels(cs clientset.Interface, namespaces []string, labels map[string]string) error {
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
		done <- struct{}{}
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// nextPodOrDie returns the next Pod in the scheduler queue.
// The operation needs to be completed within 5 seconds; otherwise the test gets aborted.
func nextPodOrDie(t *testing.T, testCtx *testutils.TestContext) *framework.QueuedPodInfo {
	t.Helper()

	var podInfo *framework.QueuedPodInfo
	// NextPod() is a blocking operation. Wrap it in timeout() to avoid relying on
	// default go testing timeout (10m) to abort.
	if err := timeout(testCtx.Ctx, time.Second*5, func() {
		podInfo = testCtx.Scheduler.NextPod()
	}); err != nil {
		t.Fatalf("Timed out waiting for the Pod to be popped: %v", err)
	}
	return podInfo
}

// nextPod returns the next Pod in the scheduler queue, with a 5 seconds timeout.
func nextPod(t *testing.T, testCtx *testutils.TestContext) *framework.QueuedPodInfo {
	t.Helper()

	var podInfo *framework.QueuedPodInfo
	// NextPod() is a blocking operation. Wrap it in timeout() to avoid relying on
	// default go testing timeout (10m) to abort.
	if err := timeout(testCtx.Ctx, time.Second*5, func() {
		podInfo = testCtx.Scheduler.NextPod()
	}); err != nil {
		return nil
	}
	return podInfo
}
