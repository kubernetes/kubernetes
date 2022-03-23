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

package util

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	pvutil "k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	taintutils "k8s.io/kubernetes/pkg/util/taints"
	"k8s.io/kubernetes/test/integration/framework"
)

// ShutdownFunc represents the function handle to be called, typically in a defer handler, to shutdown a running module
type ShutdownFunc func()

// StartApiserver starts a local API server for testing and returns the handle to the URL and the shutdown function to stop it.
func StartApiserver() (string, ShutdownFunc) {
	h := &framework.APIServerHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	_, _, closeFn := framework.RunAnAPIServerUsingServer(framework.NewIntegrationTestControlPlaneConfig(), s, h)

	shutdownFunc := func() {
		klog.Infof("destroying API server")
		closeFn()
		s.Close()
		klog.Infof("destroyed API server")
	}
	return s.URL, shutdownFunc
}

// StartScheduler configures and starts a scheduler given a handle to the clientSet interface
// and event broadcaster. It returns the running scheduler, podInformer and the shutdown function to stop it.
func StartScheduler(clientSet clientset.Interface, kubeConfig *restclient.Config, cfg *kubeschedulerconfig.KubeSchedulerConfiguration) (*scheduler.Scheduler, coreinformers.PodInformer, ShutdownFunc) {
	ctx, cancel := context.WithCancel(context.Background())

	informerFactory := scheduler.NewInformerFactory(clientSet, 0)
	evtBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{
		Interface: clientSet.EventsV1()})

	evtBroadcaster.StartRecordingToSink(ctx.Done())

	sched, err := scheduler.New(
		clientSet,
		informerFactory,
		nil,
		profile.NewRecorderFactory(evtBroadcaster),
		ctx.Done(),
		scheduler.WithKubeConfig(kubeConfig),
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithPercentageOfNodesToScore(cfg.PercentageOfNodesToScore),
		scheduler.WithPodMaxBackoffSeconds(cfg.PodMaxBackoffSeconds),
		scheduler.WithPodInitialBackoffSeconds(cfg.PodInitialBackoffSeconds),
		scheduler.WithExtenders(cfg.Extenders...),
		scheduler.WithParallelism(cfg.Parallelism))
	if err != nil {
		klog.Fatalf("Error creating scheduler: %v", err)
	}

	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	go sched.Run(ctx)

	shutdownFunc := func() {
		klog.Infof("destroying scheduler")
		cancel()
		klog.Infof("destroyed scheduler")
	}
	return sched, informerFactory.Core().V1().Pods(), shutdownFunc
}

// StartFakePVController is a simplified pv controller logic that sets PVC VolumeName and annotation for each PV binding.
// TODO(mborsz): Use a real PV controller here.
func StartFakePVController(clientSet clientset.Interface) ShutdownFunc {
	ctx, cancel := context.WithCancel(context.Background())

	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)
	pvInformer := informerFactory.Core().V1().PersistentVolumes()

	syncPV := func(obj *v1.PersistentVolume) {
		ctx := context.Background()
		if obj.Spec.ClaimRef != nil {
			claimRef := obj.Spec.ClaimRef
			pvc, err := clientSet.CoreV1().PersistentVolumeClaims(claimRef.Namespace).Get(ctx, claimRef.Name, metav1.GetOptions{})
			if err != nil {
				klog.Errorf("error while getting %v/%v: %v", claimRef.Namespace, claimRef.Name, err)
				return
			}

			if pvc.Spec.VolumeName == "" {
				pvc.Spec.VolumeName = obj.Name
				metav1.SetMetaDataAnnotation(&pvc.ObjectMeta, pvutil.AnnBindCompleted, "yes")
				_, err := clientSet.CoreV1().PersistentVolumeClaims(claimRef.Namespace).Update(ctx, pvc, metav1.UpdateOptions{})
				if err != nil {
					klog.Errorf("error while updating %v/%v: %v", claimRef.Namespace, claimRef.Name, err)
					return
				}
			}
		}
	}

	pvInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			syncPV(obj.(*v1.PersistentVolume))
		},
		UpdateFunc: func(_, obj interface{}) {
			syncPV(obj.(*v1.PersistentVolume))
		},
	})

	informerFactory.Start(ctx.Done())
	return ShutdownFunc(cancel)
}

// TestContext store necessary context info
type TestContext struct {
	CloseFn            framework.CloseFunc
	HTTPServer         *httptest.Server
	NS                 *v1.Namespace
	ClientSet          clientset.Interface
	KubeConfig         *restclient.Config
	InformerFactory    informers.SharedInformerFactory
	DynInformerFactory dynamicinformer.DynamicSharedInformerFactory
	Scheduler          *scheduler.Scheduler
	Ctx                context.Context
	CancelFn           context.CancelFunc
}

// CleanupNodes cleans all nodes which were created during integration test
func CleanupNodes(cs clientset.Interface, t *testing.T) {
	err := cs.CoreV1().Nodes().DeleteCollection(context.TODO(), *metav1.NewDeleteOptions(0), metav1.ListOptions{})
	if err != nil {
		t.Errorf("error while deleting all nodes: %v", err)
	}
}

// PodDeleted returns true if a pod is not found in the given namespace.
func PodDeleted(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		if pod.DeletionTimestamp != nil {
			return true, nil
		}
		return false, nil
	}
}

// SyncInformerFactory starts informer and waits for caches to be synced
func SyncInformerFactory(testCtx *TestContext) {
	testCtx.InformerFactory.Start(testCtx.Ctx.Done())
	if testCtx.DynInformerFactory != nil {
		testCtx.DynInformerFactory.Start(testCtx.Ctx.Done())
	}
	testCtx.InformerFactory.WaitForCacheSync(testCtx.Ctx.Done())
	if testCtx.DynInformerFactory != nil {
		testCtx.DynInformerFactory.WaitForCacheSync(testCtx.Ctx.Done())
	}
}

// CleanupTest cleans related resources which were created during integration test
func CleanupTest(t *testing.T, testCtx *TestContext) {
	// Kill the scheduler.
	testCtx.CancelFn()
	// Cleanup nodes.
	testCtx.ClientSet.CoreV1().Nodes().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})
	framework.DeleteTestingNamespace(testCtx.NS, testCtx.HTTPServer, t)
	testCtx.CloseFn()
}

// CleanupPods deletes the given pods and waits for them to be actually deleted.
func CleanupPods(cs clientset.Interface, t *testing.T, pods []*v1.Pod) {
	for _, p := range pods {
		err := cs.CoreV1().Pods(p.Namespace).Delete(context.TODO(), p.Name, *metav1.NewDeleteOptions(0))
		if err != nil && !apierrors.IsNotFound(err) {
			t.Errorf("error while deleting pod %v/%v: %v", p.Namespace, p.Name, err)
		}
	}
	for _, p := range pods {
		if err := wait.Poll(time.Millisecond, wait.ForeverTestTimeout,
			PodDeleted(cs, p.Namespace, p.Name)); err != nil {
			t.Errorf("error while waiting for pod  %v/%v to get deleted: %v", p.Namespace, p.Name, err)
		}
	}
}

// AddTaintToNode add taints to specific node
func AddTaintToNode(cs clientset.Interface, nodeName string, taint v1.Taint) error {
	node, err := cs.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	node.Spec.Taints = append(node.Spec.Taints, taint)
	_, err = cs.CoreV1().Nodes().Update(context.TODO(), node, metav1.UpdateOptions{})
	return err
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

// WaitForNodeTaints waits for a node to have the target taints and returns
// an error if it does not have taints within the given timeout.
func WaitForNodeTaints(cs clientset.Interface, node *v1.Node, taints []v1.Taint) error {
	return wait.Poll(100*time.Millisecond, 30*time.Second, NodeTainted(cs, node.Name, taints))
}

// NodeTainted return a condition function that returns true if the given node contains
// the taints.
func NodeTainted(cs clientset.Interface, nodeName string, taints []v1.Taint) wait.ConditionFunc {
	return func() (bool, error) {
		node, err := cs.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		// node.Spec.Taints may have more taints
		if len(taints) > len(node.Spec.Taints) {
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

// NodeReadyStatus returns the status of first condition with type NodeReady.
// If none of the condition is of type NodeReady, returns an error.
func NodeReadyStatus(conditions []v1.NodeCondition) (v1.ConditionStatus, error) {
	for _, c := range conditions {
		if c.Type != v1.NodeReady {
			continue
		}
		// Just return the first condition with type NodeReady
		return c.Status, nil
	}
	return v1.ConditionFalse, errors.New("None of the conditions is of type NodeReady")
}

// GetTolerationSeconds gets the period of time the toleration
func GetTolerationSeconds(tolerations []v1.Toleration) (int64, error) {
	for _, t := range tolerations {
		if t.Key == v1.TaintNodeNotReady && t.Effect == v1.TaintEffectNoExecute && t.Operator == v1.TolerationOpExists {
			return *t.TolerationSeconds, nil
		}
	}
	return 0, fmt.Errorf("cannot find toleration")
}

// NodeCopyWithConditions duplicates the ode object with conditions
func NodeCopyWithConditions(node *v1.Node, conditions []v1.NodeCondition) *v1.Node {
	copy := node.DeepCopy()
	copy.ResourceVersion = "0"
	copy.Status.Conditions = conditions
	for i := range copy.Status.Conditions {
		copy.Status.Conditions[i].LastHeartbeatTime = metav1.Now()
	}
	return copy
}

// UpdateNodeStatus updates the status of node.
func UpdateNodeStatus(cs clientset.Interface, node *v1.Node) error {
	_, err := cs.CoreV1().Nodes().UpdateStatus(context.TODO(), node, metav1.UpdateOptions{})
	return err
}

// InitTestAPIServer initializes a test environment and creates an API server with default
// configuration.
func InitTestAPIServer(t *testing.T, nsPrefix string, admission admission.Interface) *TestContext {
	ctx, cancelFunc := context.WithCancel(context.Background())
	testCtx := TestContext{
		Ctx:      ctx,
		CancelFn: cancelFunc,
	}

	// 1. Create API server
	h := &framework.APIServerHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	controlPlaneConfig := framework.NewIntegrationTestControlPlaneConfig()

	if admission != nil {
		controlPlaneConfig.GenericConfig.AdmissionControl = admission
	}

	_, testCtx.HTTPServer, testCtx.CloseFn = framework.RunAnAPIServerUsingServer(controlPlaneConfig, s, h)

	if nsPrefix != "default" {
		testCtx.NS = framework.CreateTestingNamespace(nsPrefix+string(uuid.NewUUID()), s, t)
	} else {
		testCtx.NS = framework.CreateTestingNamespace("default", s, t)
	}

	// 2. Create kubeclient
	kubeConfig := &restclient.Config{
		QPS: -1, Host: s.URL,
		ContentConfig: restclient.ContentConfig{
			GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"},
		},
	}
	testCtx.KubeConfig = kubeConfig
	testCtx.ClientSet = clientset.NewForConfigOrDie(kubeConfig)
	return &testCtx
}

// WaitForSchedulerCacheCleanup waits for cleanup of scheduler's cache to complete
func WaitForSchedulerCacheCleanup(sched *scheduler.Scheduler, t *testing.T) {
	schedulerCacheIsEmpty := func() (bool, error) {
		dump := sched.Cache.Dump()

		return len(dump.Nodes) == 0 && len(dump.AssumedPods) == 0, nil
	}

	if err := wait.Poll(time.Second, wait.ForeverTestTimeout, schedulerCacheIsEmpty); err != nil {
		t.Errorf("Failed to wait for scheduler cache cleanup: %v", err)
	}
}

// InitTestScheduler initializes a test environment and creates a scheduler with default
// configuration.
func InitTestScheduler(
	t *testing.T,
	testCtx *TestContext,
) *TestContext {
	// Pod preemption is enabled by default scheduler configuration.
	return InitTestSchedulerWithOptions(t, testCtx)
}

// InitTestSchedulerWithOptions initializes a test environment and creates a scheduler with default
// configuration and other options.
func InitTestSchedulerWithOptions(
	t *testing.T,
	testCtx *TestContext,
	opts ...scheduler.Option,
) *TestContext {
	// 1. Create scheduler
	testCtx.InformerFactory = scheduler.NewInformerFactory(testCtx.ClientSet, 0)
	if testCtx.KubeConfig != nil {
		dynClient := dynamic.NewForConfigOrDie(testCtx.KubeConfig)
		testCtx.DynInformerFactory = dynamicinformer.NewFilteredDynamicSharedInformerFactory(dynClient, 0, v1.NamespaceAll, nil)
	}

	var err error
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{
		Interface: testCtx.ClientSet.EventsV1(),
	})

	opts = append(opts, scheduler.WithKubeConfig(testCtx.KubeConfig))
	testCtx.Scheduler, err = scheduler.New(
		testCtx.ClientSet,
		testCtx.InformerFactory,
		testCtx.DynInformerFactory,
		profile.NewRecorderFactory(eventBroadcaster),
		testCtx.Ctx.Done(),
		opts...,
	)

	if err != nil {
		t.Fatalf("Couldn't create scheduler: %v", err)
	}

	stopCh := make(chan struct{})
	eventBroadcaster.StartRecordingToSink(stopCh)

	return testCtx
}

// WaitForPodToScheduleWithTimeout waits for a pod to get scheduled and returns
// an error if it does not scheduled within the given timeout.
func WaitForPodToScheduleWithTimeout(cs clientset.Interface, pod *v1.Pod, timeout time.Duration) error {
	return wait.Poll(100*time.Millisecond, timeout, PodScheduled(cs, pod.Namespace, pod.Name))
}

// WaitForPodToSchedule waits for a pod to get scheduled and returns an error if
// it does not get scheduled within the timeout duration (30 seconds).
func WaitForPodToSchedule(cs clientset.Interface, pod *v1.Pod) error {
	return WaitForPodToScheduleWithTimeout(cs, pod, 30*time.Second)
}

// PodScheduled checks if the pod has been scheduled
func PodScheduled(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(context.TODO(), podName, metav1.GetOptions{})
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
