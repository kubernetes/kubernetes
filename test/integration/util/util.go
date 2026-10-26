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
	"encoding/json"
	"fmt"
	"net/http"
	"sync/atomic"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingapiv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/ktesting"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/metadata/metadatainformer"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/events"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/controller-manager/pkg/informerfactory"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/controller/resourceclaim"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	taintutils "k8s.io/kubernetes/pkg/util/taints"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// StartScheduler is a wrapper around StartSchedulerWithDone for backward compatibility.
func StartScheduler(tCtx ktesting.TContext, cfg *kubeschedulerconfig.KubeSchedulerConfiguration, outOfTreePluginRegistry frameworkruntime.Registry) (*scheduler.Scheduler, informers.SharedInformerFactory) {
	sched, informerFactory, _ := StartSchedulerWithDone(tCtx, cfg, outOfTreePluginRegistry)
	return sched, informerFactory
}

// StartSchedulerWithDone configures and starts a scheduler. Background goroutines
// will keep running until the context is canceled. It returns the running scheduler,
// the informer factory, and a channel that is closed when the scheduler goroutine
// actually exits. Callers can use this channel to ensure the scheduler has fully
// stopped before performing operations that might race with it (e.g., resetting
// global metrics).
func StartSchedulerWithDone(tCtx ktesting.TContext, cfg *kubeschedulerconfig.KubeSchedulerConfiguration, outOfTreePluginRegistry frameworkruntime.Registry) (*scheduler.Scheduler, informers.SharedInformerFactory, <-chan struct{}) {
	clientSet := tCtx.Client()
	informerFactory := scheduler.NewInformerFactory(clientSet, 0, nil)
	evtBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{
		Interface: clientSet.EventsV1()})
	go func() {
		<-tCtx.Done()
		evtBroadcaster.Shutdown()
	}()

	evtBroadcaster.StartRecordingToSink(tCtx.Done())

	logger := tCtx.Logger()

	sched, err := scheduler.New(
		tCtx,
		clientSet,
		informerFactory,
		nil,
		profile.NewRecorderFactory(evtBroadcaster),
		scheduler.WithKubeConfig(tCtx.RESTConfig()),
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithPercentageOfNodesToScore(cfg.PercentageOfNodesToScore),
		scheduler.WithPercentageOfPlacementsToScore(cfg.PercentageOfPlacementsToScore),
		scheduler.WithPodMaxBackoffSeconds(cfg.PodMaxBackoffSeconds),
		scheduler.WithPodInitialBackoffSeconds(cfg.PodInitialBackoffSeconds),
		scheduler.WithExtenders(cfg.Extenders...),
		scheduler.WithParallelism(cfg.Parallelism),
		scheduler.WithFrameworkOutOfTreeRegistry(outOfTreePluginRegistry),
	)
	tCtx.ExpectNoError(err, "creating scheduler")

	informerFactory.Start(tCtx.Done())
	informerFactory.WaitForCacheSync(tCtx.Done())
	err = sched.WaitForHandlersSync(tCtx)
	tCtx.ExpectNoError(err, "waiting for handlers to sync")
	logger.V(3).Info("Handlers synced")
	done := make(chan struct{})
	go func() {
		sched.Run(tCtx)
		close(done)
	}()

	return sched, informerFactory, done
}

// CreateResourceClaimController creates a ResourceClaim controller and returns a blocking run function.
// The caller is responsible for the management of the goroutine where that method is invoked.
func CreateResourceClaimController(ctx context.Context, tb testutils.TB, clientSet clientset.Interface, informerFactory informers.SharedInformerFactory) func() {
	podInformer := informerFactory.Core().V1().Pods()
	podGroupInformer := informerFactory.Scheduling().V1beta1().PodGroups()
	claimInformer := informerFactory.Resource().V1().ResourceClaims()
	claimTemplateInformer := informerFactory.Resource().V1().ResourceClaimTemplates()
	claimController, err := resourceclaim.NewController(klog.FromContext(ctx), clientSet, podInformer, podGroupInformer, claimInformer, claimTemplateInformer)
	if err != nil {
		tb.Fatalf("Error creating claim controller: %v", err)
	}
	return func() {
		claimController.Run(ctx, 5 /* workers */)
	}
}

// CreateGCController creates a garbage controller and returns a run function
// for it. The informer factory needs to be started before invoking that
// function.
func CreateGCController(ctx context.Context, tb testutils.TB, restConfig restclient.Config, informerSet informers.SharedInformerFactory) func() {
	restclient.AddUserAgent(&restConfig, "gc-controller")
	clientSet := clientset.NewForConfigOrDie(&restConfig)
	metadataClient, err := metadata.NewForConfig(&restConfig)
	if err != nil {
		tb.Fatalf("Failed to create metadataClient: %v", err)
	}
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cacheddiscovery.NewMemCacheClient(clientSet.Discovery()))
	restMapper.Reset()
	metadataInformers := metadatainformer.NewSharedInformerFactory(metadataClient, 0)
	alwaysStarted := make(chan struct{})
	close(alwaysStarted)
	gc, err := garbagecollector.NewGarbageCollector(
		ctx,
		clientSet,
		metadataClient,
		restMapper,
		garbagecollector.DefaultIgnoredResources(),
		informerfactory.NewInformerFactory(informerSet, metadataInformers),
		alwaysStarted,
	)
	if err != nil {
		tb.Fatalf("Failed creating garbage collector")
	}
	startGC := func() {
		syncPeriod := 5 * time.Second
		go wait.Until(func() {
			restMapper.Reset()
		}, syncPeriod, ctx.Done())
		go gc.Run(ctx, 1, syncPeriod)
		go gc.Sync(ctx, clientSet.Discovery(), syncPeriod)
	}
	return startGC
}

// TestContext store necessary context info.
// It also contains some optional parameters for InitTestScheduler.
type TestContext struct {
	// DisableEventSink, if set to true before calling InitTestScheduler,
	// will skip the eventBroadcaster.StartRecordingToSink and thus
	// some extra goroutines which are tricky to get rid of after
	// a test.
	DisableEventSink bool

	NS                 *v1.Namespace
	ClientSet          clientset.Interface
	KubeConfig         *restclient.Config
	InformerFactory    informers.SharedInformerFactory
	DynInformerFactory dynamicinformer.DynamicSharedInformerFactory
	Scheduler          *scheduler.Scheduler
	// This is the top context when initializing the test environment.
	// Note that this will be canceled at the time that cleanup callbacks
	// run! Use [TContext.CleanupCtx] to schedule callbacks which have
	// their own fresh context.
	Ctx context.Context
	// CloseFn will stop the apiserver and clean up the resources
	// after itself, including shutting down its storage layer.
	CloseFn framework.TearDownFunc
	// This is the context when initializing scheduler.
	SchedulerCtx context.Context
	// SchedulerCloseFn will tear down the resources in creating scheduler,
	// including the scheduler itself.
	SchedulerCloseFn framework.TearDownFunc

	// RoundTrip, if set, will be called for every HTTP request going to the apiserver.
	// It can be used for error injection.
	RoundTrip atomic.Pointer[RoundTripWrapper]
}

type RoundTripWrapper func(http.RoundTripper, *http.Request) (*http.Response, error)

type roundTripWrapper struct {
	tc        *TestContext
	transport http.RoundTripper
}

func (r roundTripWrapper) RoundTrip(req *http.Request) (*http.Response, error) {
	wrapper := r.tc.RoundTrip.Load()
	if wrapper != nil {
		return (*wrapper)(r.transport, req)
	}
	return r.transport.RoundTrip(req)
}

var _ http.RoundTripper = roundTripWrapper{}

// CleanupNodes cleans all nodes which were created during integration test
func CleanupNodes(cs clientset.Interface, t testutils.TB) {
	err := cs.CoreV1().Nodes().DeleteCollection(context.TODO(), *metav1.NewDeleteOptions(0), metav1.ListOptions{})
	if err != nil {
		t.Errorf("error while deleting all nodes: %v", err)
	}
}

// PodDeleted returns true if a pod is not found in the given namespace.
func PodDeleted(ctx context.Context, c clientset.Interface, podNamespace, podName string) wait.ConditionWithContextFunc {
	return func(context.Context) (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(ctx, podName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		if pod.DeletionTimestamp != nil {
			return true, nil
		}
		return false, nil
	}
}

// SyncSchedulerInformerFactory starts informer and waits for caches to be synced
func SyncSchedulerInformerFactory(testCtx *TestContext) {
	testCtx.InformerFactory.Start(testCtx.SchedulerCtx.Done())
	if testCtx.DynInformerFactory != nil {
		testCtx.DynInformerFactory.Start(testCtx.SchedulerCtx.Done())
	}
	testCtx.InformerFactory.WaitForCacheSync(testCtx.SchedulerCtx.Done())
	if testCtx.DynInformerFactory != nil {
		testCtx.DynInformerFactory.WaitForCacheSync(testCtx.SchedulerCtx.Done())
	}
}

// CleanupTest cleans related resources which were created during integration test.
// May only be called once and with a tCtx which is not already canceled.
func CleanupTest(tCtx ktesting.TContext, testCtx *TestContext) {
	// Cleanup nodes and namespaces.
	if err := testCtx.ClientSet.CoreV1().Nodes().DeleteCollection(tCtx, *metav1.NewDeleteOptions(0), metav1.ListOptions{}); err != nil {
		tCtx.Errorf("error while cleaning up nodes, error: %v", err)
	}
	framework.DeleteNamespaceOrDie(testCtx.ClientSet, testCtx.NS, tCtx)
	// Terminate the scheduler and apiserver.
	testCtx.CloseFn()
}

func RemovePodFinalizersInNamespace(ctx context.Context, cs clientset.Interface, t testutils.TB, ns string) {
	t.Helper()
	pods, err := cs.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed obtaining list of pods: %v", err)
	}
	RemovePodFinalizers(ctx, cs, t, pods.Items...)
}

// RemovePodFinalizers removes pod finalizers for the pods
func RemovePodFinalizers(ctx context.Context, cs clientset.Interface, t testutils.TB, pods ...v1.Pod) {
	t.Helper()
	for _, p := range pods {
		pod, err := cs.CoreV1().Pods(p.Namespace).Get(ctx, p.Name, metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			t.Errorf("error while removing pod finalizers for %v: %v", klog.KObj(&p), err)
		} else if pod != nil && len(pod.Finalizers) > 0 {
			// Use Patch to remove finalizer, instead of Update, to avoid transient
			// conflicts.
			patchBytes, _ := json.Marshal(map[string]interface{}{
				"metadata": map[string]interface{}{
					"$deleteFromPrimitiveList/finalizers": pod.Finalizers,
				},
			})
			_, err = cs.CoreV1().Pods(pod.Namespace).Patch(ctx, pod.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
			if err != nil {
				t.Errorf("error while updating pod status for %v: %v", klog.KObj(&p), err)
			}
		}
	}
}

// CleanupPods deletes the given pods and waits for them to be actually deleted.
func CleanupPods(ctx context.Context, cs clientset.Interface, t testutils.TB, pods []*v1.Pod) {
	for _, p := range pods {
		err := cs.CoreV1().Pods(p.Namespace).Delete(ctx, p.Name, *metav1.NewDeleteOptions(0))
		if err != nil && !apierrors.IsNotFound(err) {
			t.Errorf("error while deleting pod %v/%v: %v", p.Namespace, p.Name, err)
		}
	}
	for _, p := range pods {
		if err := wait.PollUntilContextTimeout(ctx, time.Duration(time.Microsecond.Seconds()), wait.ForeverTestTimeout, true,
			PodDeleted(ctx, cs, p.Namespace, p.Name)); err != nil {
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

// WaitForNodeTaints waits for a node to have the target taints and returns
// an error if it does not have taints within the given timeout.
func WaitForNodeTaints(ctx context.Context, cs clientset.Interface, node *v1.Node, taints []v1.Taint) error {
	return wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 30*time.Second, false, NodeTainted(ctx, cs, node.Name, taints))
}

// NodeTainted return a condition function that returns true if the given node contains
// the taints.
func NodeTainted(ctx context.Context, cs clientset.Interface, nodeName string, taints []v1.Taint) wait.ConditionWithContextFunc {
	return func(context.Context) (bool, error) {
		node, err := cs.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
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

// GetTolerationSeconds gets the period of time the toleration
func GetTolerationSeconds(tolerations []v1.Toleration) (int64, error) {
	for _, t := range tolerations {
		if t.Key == v1.TaintNodeNotReady && t.Effect == v1.TaintEffectNoExecute && t.Operator == v1.TolerationOpExists {
			return *t.TolerationSeconds, nil
		}
	}
	return 0, fmt.Errorf("cannot find toleration")
}

// InitTestAPIServer initializes a test environment and creates an API server with default
// configuration.
// It registers cleanup functions to t.Cleanup(), they will be called when the test completes,
// no need to do this again.
func InitTestAPIServer(t ktesting.TB, nsPrefix string, admission admission.Interface) *TestContext {
	tCtx := ktesting.Init(t)
	// We need to intercept context cancellation, otherwise
	// CleanupTest cannot succeed. CleanupTest calls
	// testCtx.CloseFn, which then explicitly shuts down
	// the server and cancels the context. The latter is
	// important because the caller might start more goroutines
	// which use the context.
	//
	// Cancellation gets propagated to the test server only during the startup
	// phase.
	parentCtx := tCtx
	tCtx = tCtx.WithoutCancel()
	stopCtxPropagation := context.AfterFunc(parentCtx, func() {
		tCtx.CancelBecause(context.Cause(parentCtx))
	})
	testCtx := &TestContext{Ctx: parentCtx}

	testCtx.ClientSet, testCtx.KubeConfig, testCtx.CloseFn = framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(options *options.ServerRunOptions) {
			options.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount", "TaintNodesByCondition", "Priority", "StorageObjectInUseProtection"}
			if options.APIEnablement.RuntimeConfig == nil {
				options.APIEnablement.RuntimeConfig = cliflag.ConfigurationMap{}
			}
			options.APIEnablement.RuntimeConfig[resourceapi.SchemeGroupVersion.String()] = "true"
			if utilfeature.DefaultMutableFeatureGate.EmulationVersion().LessThan(version.MustParse("v1.34.0")) {
				// Cannot enable the resourceapi.SchemeGroupVersion when emulating < 1.34 unless
				// we enable --runtime-config-emulation-forward-compatible.
				options.GenericServerRunOptions.RuntimeConfigEmulationForwardCompatible = true
			}
			if utilfeature.DefaultFeatureGate.Enabled(features.GenericWorkload) {
				options.APIEnablement.RuntimeConfig[schedulingapiv1beta1.SchemeGroupVersion.String()] = "true"
			}
		},
		ModifyServerConfig: func(config *controlplane.Config) {
			if admission != nil {
				config.ControlPlane.Generic.AdmissionControl = admission
			}
		},
	})

	// Support wrapping HTTP requests.
	testCtx.KubeConfig.Wrap(func(transport http.RoundTripper) http.RoundTripper {
		return roundTripWrapper{tc: testCtx, transport: transport}
	})
	var err error
	testCtx.ClientSet, err = clientset.NewForConfig(testCtx.KubeConfig)
	if err != nil {
		t.Fatal(err)
	}

	oldCloseFn := testCtx.CloseFn
	testCtx.CloseFn = func() {
		// Shutdown of additional goroutines and of the apiserver run in parallel.
		// parentCtx (== testCtx.Ctx) is the context that callers and their
		// goroutines were given, so it must be canceled here, not the
		// detached tCtx which only governs the apiserver's own shutdown.
		parentCtx.Cancel("must shut down, apiserver is being torn down")
		oldCloseFn()
	}

	if nsPrefix != "default" {
		testCtx.NS = framework.CreateNamespaceOrDie(testCtx.ClientSet, nsPrefix+string(uuid.NewUUID()), t)
	} else {
		testCtx.NS = framework.CreateNamespaceOrDie(testCtx.ClientSet, "default", t)
	}

	stopCtxPropagation()
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		CleanupTest(tCtx, testCtx)
	})

	return testCtx
}

// InitTestSchedulerWithOptions initializes a test environment and creates a scheduler with default
// configuration and other options.
func InitTestSchedulerWithOptions(
	t testutils.TB,
	testCtx *TestContext,
	resyncPeriod time.Duration,
	opts ...scheduler.Option,
) *TestContext {
	ctx, cancel := context.WithCancel(testCtx.Ctx)
	testCtx.SchedulerCtx = ctx

	// 1. Create scheduler
	testCtx.InformerFactory = scheduler.NewInformerFactory(testCtx.ClientSet, resyncPeriod, nil)
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
		ctx,
		testCtx.ClientSet,
		testCtx.InformerFactory,
		testCtx.DynInformerFactory,
		profile.NewRecorderFactory(eventBroadcaster),
		opts...,
	)

	if err != nil {
		t.Fatalf("Couldn't create scheduler: %v", err)
	}

	if !testCtx.DisableEventSink {
		eventBroadcaster.StartRecordingToSink(ctx.Done())
	}

	oldCloseFn := testCtx.CloseFn
	testCtx.CloseFn = func() {
		oldCloseFn()
		eventBroadcaster.Shutdown()
	}

	testCtx.SchedulerCloseFn = func() {
		cancel()
		eventBroadcaster.Shutdown()
	}

	return testCtx
}

type PausePodConfig struct {
	Name                              string
	Namespace                         string
	PodGroupName                      string
	Affinity                          *v1.Affinity
	Annotations, Labels, NodeSelector map[string]string
	Resources                         *v1.ResourceRequirements
	Tolerations                       []v1.Toleration
	NodeName                          string
	SchedulerName                     string
	Priority                          *int32
	PreemptionPolicy                  *v1.PreemptionPolicy
	PriorityClassName                 string
	Volumes                           []v1.Volume
	ContainerPorts                    []v1.ContainerPort
	RestartableInitContainerPorts     []v1.ContainerPort
	NonRestartableInitContainerPorts  []v1.ContainerPort
}

// InitPausePod initializes a pod API object from the given config. It is used
// mainly in pod creation process.
func InitPausePod(conf *PausePodConfig) *v1.Pod {
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
					Ports: conf.ContainerPorts,
				},
			},
			Tolerations:       conf.Tolerations,
			NodeName:          conf.NodeName,
			SchedulerName:     conf.SchedulerName,
			Priority:          conf.Priority,
			PreemptionPolicy:  conf.PreemptionPolicy,
			PriorityClassName: conf.PriorityClassName,
			Volumes:           conf.Volumes,
		},
	}
	if conf.Resources != nil {
		pod.Spec.Containers[0].Resources = *conf.Resources
	}
	if conf.RestartableInitContainerPorts != nil {
		var containerRestartPolicyAlways = v1.ContainerRestartPolicyAlways
		pod.Spec.InitContainers = []v1.Container{
			{
				Name:          conf.Name + "-init",
				Image:         imageutils.GetPauseImageName(),
				RestartPolicy: &containerRestartPolicyAlways,
				Ports:         conf.RestartableInitContainerPorts,
			},
		}
	}
	if conf.NonRestartableInitContainerPorts != nil {
		pod.Spec.InitContainers = []v1.Container{
			{
				Name:  conf.Name + "-init",
				Image: imageutils.GetPauseImageName(),
				Ports: conf.NonRestartableInitContainerPorts,
			},
		}
	}
	if conf.PodGroupName != "" {
		pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: new(conf.PodGroupName)}
	}
	return pod
}

// CreatePausePod creates a pod with "Pause" image and the given config and
// return its pointer and error status.
func CreatePausePod(cs clientset.Interface, p *v1.Pod) (*v1.Pod, error) {
	return cs.CoreV1().Pods(p.Namespace).Create(context.TODO(), p, metav1.CreateOptions{})
}

// PodIsGettingEvicted returns true if the pod's deletion timestamp is set.
func PodIsGettingEvicted(c clientset.Interface, podNamespace, podName string) wait.ConditionWithContextFunc {
	return func(ctx context.Context) (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if pod.DeletionTimestamp != nil {
			return true, nil
		}
		return false, nil
	}
}
