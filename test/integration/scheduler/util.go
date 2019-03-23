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
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	clientv1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/disruption"
	"k8s.io/kubernetes/pkg/scheduler"
	// Register defaults in pkg/scheduler/algorithmprovider.
	_ "k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/pkg/scheduler/factory"
	plugins "k8s.io/kubernetes/pkg/scheduler/plugins/v1alpha1"
	taintutils "k8s.io/kubernetes/pkg/util/taints"
	"k8s.io/kubernetes/test/integration/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

type testContext struct {
	closeFn                framework.CloseFunc
	httpServer             *httptest.Server
	ns                     *v1.Namespace
	clientSet              *clientset.Clientset
	informerFactory        informers.SharedInformerFactory
	schedulerConfigFactory factory.Configurator
	schedulerConfig        *factory.Config
	scheduler              *scheduler.Scheduler
	stopCh                 chan struct{}
}

// createConfiguratorWithPodInformer creates a configurator for scheduler.
func createConfiguratorWithPodInformer(
	schedulerName string,
	clientSet clientset.Interface,
	podInformer coreinformers.PodInformer,
	informerFactory informers.SharedInformerFactory,
	stopCh <-chan struct{},
) factory.Configurator {
	return factory.NewConfigFactory(&factory.ConfigFactoryArgs{
		SchedulerName:                  schedulerName,
		Client:                         clientSet,
		NodeInformer:                   informerFactory.Core().V1().Nodes(),
		PodInformer:                    podInformer,
		PvInformer:                     informerFactory.Core().V1().PersistentVolumes(),
		PvcInformer:                    informerFactory.Core().V1().PersistentVolumeClaims(),
		ReplicationControllerInformer:  informerFactory.Core().V1().ReplicationControllers(),
		ReplicaSetInformer:             informerFactory.Apps().V1().ReplicaSets(),
		StatefulSetInformer:            informerFactory.Apps().V1().StatefulSets(),
		ServiceInformer:                informerFactory.Core().V1().Services(),
		PdbInformer:                    informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		StorageClassInformer:           informerFactory.Storage().V1().StorageClasses(),
		HardPodAffinitySymmetricWeight: v1.DefaultHardPodAffinitySymmetricWeight,
		DisablePreemption:              false,
		PercentageOfNodesToScore:       schedulerapi.DefaultPercentageOfNodesToScore,
		BindTimeoutSeconds:             600,
		StopCh:                         stopCh,
	})
}

// initTestMasterAndScheduler initializes a test environment and creates a master with default
// configuration.
func initTestMaster(t *testing.T, nsPrefix string, admission admission.Interface) *testContext {
	context := testContext{
		stopCh: make(chan struct{}),
	}

	// 1. Create master
	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	masterConfig := framework.NewIntegrationTestMasterConfig()

	if admission != nil {
		masterConfig.GenericConfig.AdmissionControl = admission
	}

	_, context.httpServer, context.closeFn = framework.RunAMasterUsingServer(masterConfig, s, h)

	if nsPrefix != "default" {
		context.ns = framework.CreateTestingNamespace(nsPrefix+string(uuid.NewUUID()), s, t)
	} else {
		context.ns = framework.CreateTestingNamespace("default", s, t)
	}

	// 2. Create kubeclient
	context.clientSet = clientset.NewForConfigOrDie(
		&restclient.Config{
			QPS: -1, Host: s.URL,
			ContentConfig: restclient.ContentConfig{
				GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"},
			},
		},
	)
	return &context
}

// initTestScheduler initializes a test environment and creates a scheduler with default
// configuration.
func initTestScheduler(
	t *testing.T,
	context *testContext,
	setPodInformer bool,
	policy *schedulerapi.Policy,
) *testContext {
	// Pod preemption is enabled by default scheduler configuration, but preemption only happens when PodPriority
	// feature gate is enabled at the same time.
	return initTestSchedulerWithOptions(t, context, setPodInformer, policy, nil, false, time.Second)
}

// initTestSchedulerWithOptions initializes a test environment and creates a scheduler with default
// configuration and other options.
func initTestSchedulerWithOptions(
	t *testing.T,
	context *testContext,
	setPodInformer bool,
	policy *schedulerapi.Policy,
	pluginSet plugins.PluginSet,
	disablePreemption bool,
	resyncPeriod time.Duration,
) *testContext {
	// 1. Create scheduler
	context.informerFactory = informers.NewSharedInformerFactory(context.clientSet, resyncPeriod)

	var podInformer coreinformers.PodInformer

	// create independent pod informer if required
	if setPodInformer {
		podInformer = factory.NewPodInformer(context.clientSet, 12*time.Hour)
	} else {
		podInformer = context.informerFactory.Core().V1().Pods()
	}

	context.schedulerConfigFactory = createConfiguratorWithPodInformer(
		v1.DefaultSchedulerName, context.clientSet, podInformer, context.informerFactory, context.stopCh)

	var err error

	if policy != nil {
		context.schedulerConfig, err = context.schedulerConfigFactory.CreateFromConfig(*policy)
	} else {
		context.schedulerConfig, err = context.schedulerConfigFactory.Create()
	}

	if err != nil {
		t.Fatalf("Couldn't create scheduler config: %v", err)
	}

	// set DisablePreemption option
	context.schedulerConfig.DisablePreemption = disablePreemption

	context.scheduler = scheduler.NewFromConfig(context.schedulerConfig)

	scheduler.AddAllEventHandlers(context.scheduler,
		v1.DefaultSchedulerName,
		context.informerFactory.Core().V1().Nodes(),
		podInformer,
		context.informerFactory.Core().V1().PersistentVolumes(),
		context.informerFactory.Core().V1().PersistentVolumeClaims(),
		context.informerFactory.Core().V1().ReplicationControllers(),
		context.informerFactory.Apps().V1().ReplicaSets(),
		context.informerFactory.Apps().V1().StatefulSets(),
		context.informerFactory.Core().V1().Services(),
		context.informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		context.informerFactory.Storage().V1().StorageClasses(),
	)

	// set setPodInformer if provided.
	if setPodInformer {
		go podInformer.Informer().Run(context.schedulerConfig.StopEverything)
		controller.WaitForCacheSync("scheduler", context.schedulerConfig.StopEverything, podInformer.Informer().HasSynced)
	}

	// Set pluginSet if provided. DefaultPluginSet is used if this is not specified.
	if pluginSet != nil {
		context.schedulerConfig.PluginSet = pluginSet
	}

	eventBroadcaster := record.NewBroadcaster()
	context.schedulerConfig.Recorder = eventBroadcaster.NewRecorder(
		legacyscheme.Scheme,
		v1.EventSource{Component: v1.DefaultSchedulerName},
	)
	eventBroadcaster.StartRecordingToSink(&clientv1core.EventSinkImpl{
		Interface: context.clientSet.CoreV1().Events(""),
	})

	context.informerFactory.Start(context.schedulerConfig.StopEverything)
	context.informerFactory.WaitForCacheSync(context.schedulerConfig.StopEverything)

	context.scheduler.Run()
	return context
}

// initDisruptionController initializes and runs a Disruption Controller to properly
// update PodDisuptionBudget objects.
func initDisruptionController(context *testContext) *disruption.DisruptionController {
	informers := informers.NewSharedInformerFactory(context.clientSet, 12*time.Hour)

	dc := disruption.NewDisruptionController(
		informers.Core().V1().Pods(),
		informers.Policy().V1beta1().PodDisruptionBudgets(),
		informers.Core().V1().ReplicationControllers(),
		informers.Apps().V1().ReplicaSets(),
		informers.Apps().V1().Deployments(),
		informers.Apps().V1().StatefulSets(),
		context.clientSet)

	informers.Start(context.schedulerConfig.StopEverything)
	informers.WaitForCacheSync(context.schedulerConfig.StopEverything)
	go dc.Run(context.schedulerConfig.StopEverything)
	return dc
}

// initTest initializes a test environment and creates master and scheduler with default
// configuration.
func initTest(t *testing.T, nsPrefix string) *testContext {
	return initTestScheduler(t, initTestMaster(t, nsPrefix, nil), true, nil)
}

// initTestDisablePreemption initializes a test environment and creates master and scheduler with default
// configuration but with pod preemption disabled.
func initTestDisablePreemption(t *testing.T, nsPrefix string) *testContext {
	return initTestSchedulerWithOptions(
		t, initTestMaster(t, nsPrefix, nil), true, nil, nil, true, time.Second)
}

// cleanupTest deletes the scheduler and the test namespace. It should be called
// at the end of a test.
func cleanupTest(t *testing.T, context *testContext) {
	// Kill the scheduler.
	close(context.stopCh)
	// Cleanup nodes.
	context.clientSet.CoreV1().Nodes().DeleteCollection(nil, metav1.ListOptions{})
	framework.DeleteTestingNamespace(context.ns, context.httpServer, t)
	context.closeFn()
}

// waitForReflection waits till the passFunc confirms that the object it expects
// to see is in the store. Used to observe reflected events.
func waitForReflection(t *testing.T, nodeLister corelisters.NodeLister, key string,
	passFunc func(n interface{}) bool) error {
	nodes := []*v1.Node{}
	err := wait.Poll(time.Millisecond*100, wait.ForeverTestTimeout, func() (bool, error) {
		n, err := nodeLister.Get(key)

		switch {
		case err == nil && passFunc(n):
			return true, nil
		case errors.IsNotFound(err):
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

// nodeHasLabels returns a function that checks if a node has all the given labels.
func nodeHasLabels(cs clientset.Interface, nodeName string, labels map[string]string) wait.ConditionFunc {
	return func() (bool, error) {
		node, err := cs.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		for k, v := range labels {
			if node.Labels == nil || node.Labels[k] != v {
				return false, nil
			}
		}
		return true, nil
	}
}

// waitForNodeLabels waits for the given node to have all the given labels.
func waitForNodeLabels(cs clientset.Interface, nodeName string, labels map[string]string) error {
	return wait.Poll(time.Millisecond*100, wait.ForeverTestTimeout, nodeHasLabels(cs, nodeName, labels))
}

// initNode returns a node with the given resource list and images. If 'res' is nil, a predefined amount of
// resource will be used.
func initNode(name string, res *v1.ResourceList, images []v1.ContainerImage) *v1.Node {
	// if resource is nil, we use a default amount of resources for the node.
	if res == nil {
		res = &v1.ResourceList{
			v1.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
		}
	}

	n := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       v1.NodeSpec{Unschedulable: false},
		Status: v1.NodeStatus{
			Capacity: *res,
			Images:   images,
		},
	}
	return n
}

// createNode creates a node with the given resource list.
func createNode(cs clientset.Interface, name string, res *v1.ResourceList) (*v1.Node, error) {
	return cs.CoreV1().Nodes().Create(initNode(name, res, nil))
}

// createNodeWithImages creates a node with the given resource list and images.
func createNodeWithImages(cs clientset.Interface, name string, res *v1.ResourceList, images []v1.ContainerImage) (*v1.Node, error) {
	return cs.CoreV1().Nodes().Create(initNode(name, res, images))
}

// updateNodeStatus updates the status of node.
func updateNodeStatus(cs clientset.Interface, node *v1.Node) error {
	_, err := cs.CoreV1().Nodes().UpdateStatus(node)
	return err
}

// createNodes creates `numNodes` nodes. The created node names will be in the
// form of "`prefix`-X" where X is an ordinal.
func createNodes(cs clientset.Interface, prefix string, res *v1.ResourceList, numNodes int) ([]*v1.Node, error) {
	nodes := make([]*v1.Node, numNodes)
	for i := 0; i < numNodes; i++ {
		nodeName := fmt.Sprintf("%v-%d", prefix, i)
		node, err := createNode(cs, nodeName, res)
		if err != nil {
			return nodes[:], err
		}
		nodes[i] = node
	}
	return nodes[:], nil
}

// nodeTainted return a condition function that returns true if the given node contains
// the taints.
func nodeTainted(cs clientset.Interface, nodeName string, taints []v1.Taint) wait.ConditionFunc {
	return func() (bool, error) {
		node, err := cs.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		if len(taints) != len(node.Spec.Taints) {
			return false, nil
		}

		for _, taint := range taints {
			if !taintutils.TaintExists(node.Spec.Taints, &taint) {
				return false, nil
			}
		}

		return true, nil
	}
}

// waitForNodeTaints waits for a node to have the target taints and returns
// an error if it does not have taints within the given timeout.
func waitForNodeTaints(cs clientset.Interface, node *v1.Node, taints []v1.Taint) error {
	return wait.Poll(100*time.Millisecond, 30*time.Second, nodeTainted(cs, node.Name, taints))
}

// cleanupNodes deletes all nodes.
func cleanupNodes(cs clientset.Interface, t *testing.T) {
	err := cs.CoreV1().Nodes().DeleteCollection(
		metav1.NewDeleteOptions(0), metav1.ListOptions{})
	if err != nil {
		t.Errorf("error while deleting all nodes: %v", err)
	}
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
}

// initPausePod initializes a pod API object from the given config. It is used
// mainly in pod creation process.
func initPausePod(cs clientset.Interface, conf *pausePodConfig) *v1.Pod {
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
			Tolerations:   conf.Tolerations,
			NodeName:      conf.NodeName,
			SchedulerName: conf.SchedulerName,
			Priority:      conf.Priority,
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
	return cs.CoreV1().Pods(p.Namespace).Create(p)
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
	return createPausePod(cs, initPausePod(cs, &conf))
}

// runPausePod creates a pod with "Pause" image and the given config and waits
// until it is scheduled. It returns its pointer and error status.
func runPausePod(cs clientset.Interface, pod *v1.Pod) (*v1.Pod, error) {
	pod, err := cs.CoreV1().Pods(pod.Namespace).Create(pod)
	if err != nil {
		return nil, fmt.Errorf("Error creating pause pod: %v", err)
	}
	if err = waitForPodToSchedule(cs, pod); err != nil {
		return pod, fmt.Errorf("Pod %v/%v didn't schedule successfully. Error: %v", pod.Namespace, pod.Name, err)
	}
	if pod, err = cs.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{}); err != nil {
		return pod, fmt.Errorf("Error getting pod %v/%v info: %v", pod.Namespace, pod.Name, err)
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
	pod, err := cs.CoreV1().Pods(pod.Namespace).Create(pod)
	if err != nil {
		return nil, fmt.Errorf("Error creating pod-with-containers: %v", err)
	}
	if err = waitForPodToSchedule(cs, pod); err != nil {
		return pod, fmt.Errorf("Pod %v didn't schedule successfully. Error: %v", pod.Name, err)
	}
	if pod, err = cs.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{}); err != nil {
		return pod, fmt.Errorf("Error getting pod %v info: %v", pod.Name, err)
	}
	return pod, nil
}

// podDeleted returns true if a pod is not found in the given namespace.
func podDeleted(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(podName, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return true, nil
		}
		if pod.DeletionTimestamp != nil {
			return true, nil
		}
		return false, nil
	}
}

// podIsGettingEvicted returns true if the pod's deletion timestamp is set.
func podIsGettingEvicted(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if pod.DeletionTimestamp != nil {
			return true, nil
		}
		return false, nil
	}
}

// podScheduled returns true if a node is assigned to the given pod.
func podScheduled(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(podName, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return false, nil
		}
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

// podUnschedulable returns a condition function that returns true if the given pod
// gets unschedulable status.
func podSchedulableCondition(c clientset.Interface, podNamespace, podName string) (*v1.PodCondition, error) {
	pod, err := c.CoreV1().Pods(podNamespace).Get(podName, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
	return cond, nil
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

// podSchedulingError returns a condition function that returns true if the given pod
// gets unschedulable status for reasons other than "Unschedulable". The scheduler
// records such reasons in case of error.
func podSchedulingError(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
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
			cond.Reason != v1.PodReasonUnschedulable, nil
	}
}

// waitForPodToScheduleWithTimeout waits for a pod to get scheduled and returns
// an error if it does not scheduled within the given timeout.
func waitForPodToScheduleWithTimeout(cs clientset.Interface, pod *v1.Pod, timeout time.Duration) error {
	return wait.Poll(100*time.Millisecond, timeout, podScheduled(cs, pod.Namespace, pod.Name))
}

// waitForPodToSchedule waits for a pod to get scheduled and returns an error if
// it does not get scheduled within the timeout duration (30 seconds).
func waitForPodToSchedule(cs clientset.Interface, pod *v1.Pod) error {
	return waitForPodToScheduleWithTimeout(cs, pod, 30*time.Second)
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

// waitForPDBsStable waits for PDBs to have "CurrentHealthy" status equal to
// the expected values.
func waitForPDBsStable(context *testContext, pdbs []*policy.PodDisruptionBudget, pdbPodNum []int32) error {
	return wait.Poll(time.Second, 60*time.Second, func() (bool, error) {
		pdbList, err := context.clientSet.PolicyV1beta1().PodDisruptionBudgets(context.ns.Name).List(metav1.ListOptions{})
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
func waitCachedPodsStable(context *testContext, pods []*v1.Pod) error {
	return wait.Poll(time.Second, 30*time.Second, func() (bool, error) {
		cachedPods, err := context.scheduler.Config().SchedulerCache.List(labels.Everything())
		if err != nil {
			return false, err
		}
		if len(pods) != len(cachedPods) {
			return false, nil
		}
		for _, p := range pods {
			actualPod, err1 := context.clientSet.CoreV1().Pods(p.Namespace).Get(p.Name, metav1.GetOptions{})
			if err1 != nil {
				return false, err1
			}
			cachedPod, err2 := context.scheduler.Config().SchedulerCache.GetPod(actualPod)
			if err2 != nil || cachedPod == nil {
				return false, err2
			}
		}
		return true, nil
	})
}

// deletePod deletes the given pod in the given namespace.
func deletePod(cs clientset.Interface, podName string, nsName string) error {
	return cs.CoreV1().Pods(nsName).Delete(podName, metav1.NewDeleteOptions(0))
}

// cleanupPods deletes the given pods and waits for them to be actually deleted.
func cleanupPods(cs clientset.Interface, t *testing.T, pods []*v1.Pod) {
	for _, p := range pods {
		err := cs.CoreV1().Pods(p.Namespace).Delete(p.Name, metav1.NewDeleteOptions(0))
		if err != nil && !errors.IsNotFound(err) {
			t.Errorf("error while deleting pod %v/%v: %v", p.Namespace, p.Name, err)
		}
	}
	for _, p := range pods {
		if err := wait.Poll(time.Millisecond, wait.ForeverTestTimeout,
			podDeleted(cs, p.Namespace, p.Name)); err != nil {
			t.Errorf("error while waiting for pod  %v/%v to get deleted: %v", p.Namespace, p.Name, err)
		}
	}
}

// noPodsInNamespace returns true if no pods in the given namespace.
func noPodsInNamespace(c clientset.Interface, podNamespace string) wait.ConditionFunc {
	return func() (bool, error) {
		pods, err := c.CoreV1().Pods(podNamespace).List(metav1.ListOptions{})
		if err != nil {
			return false, err
		}

		return len(pods.Items) == 0, nil
	}
}

// cleanupPodsInNamespace deletes the pods in the given namespace and waits for them to
// be actually deleted.
func cleanupPodsInNamespace(cs clientset.Interface, t *testing.T, ns string) {
	if err := cs.CoreV1().Pods(ns).DeleteCollection(nil, metav1.ListOptions{}); err != nil {
		t.Errorf("error while listing pod in namespace %v: %v", ns, err)
		return
	}

	if err := wait.Poll(time.Second, wait.ForeverTestTimeout,
		noPodsInNamespace(cs, ns)); err != nil {
		t.Errorf("error while waiting for pods in namespace %v: %v", ns, err)
	}
}

func waitForSchedulerCacheCleanup(sched *scheduler.Scheduler, t *testing.T) {
	schedulerCacheIsEmpty := func() (bool, error) {
		snapshot := sched.Cache().Snapshot()

		return len(snapshot.Nodes) == 0 && len(snapshot.AssumedPods) == 0, nil
	}

	if err := wait.Poll(time.Second, wait.ForeverTestTimeout, schedulerCacheIsEmpty); err != nil {
		t.Errorf("Failed to wait for scheduler cache cleanup: %v", err)
	}
}
