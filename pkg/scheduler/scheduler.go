/*
Copyright 2014 The Kubernetes Authors.

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
	"io/ioutil"
	"os"
	"reflect"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	appsinformers "k8s.io/client-go/informers/apps/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	policyinformers "k8s.io/client-go/informers/policy/v1beta1"
	storageinformers "k8s.io/client-go/informers/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	latestschedulerapi "k8s.io/kubernetes/pkg/scheduler/api/latest"
	"k8s.io/kubernetes/pkg/scheduler/api/validation"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/core"
	"k8s.io/kubernetes/pkg/scheduler/factory"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	schedulerinternalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	cachedebugger "k8s.io/kubernetes/pkg/scheduler/internal/cache/debugger"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"
)

const (
	// BindTimeoutSeconds defines the default bind timeout
	BindTimeoutSeconds = 100
	// SchedulerError is the reason recorded for events when an error occurs during scheduling a pod.
	SchedulerError = "SchedulerError"
)

// Scheduler watches for new unscheduled pods. It attempts to find
// nodes that they fit on and writes bindings back to the api server.
type Scheduler struct {
	// It is expected that changes made via SchedulerCache will be observed
	// by NodeLister and Algorithm.
	SchedulerCache internalcache.Cache

	NodeLister algorithm.NodeLister
	Algorithm  core.ScheduleAlgorithm
	GetBinder  func(pod *v1.Pod) Binder
	// PodConditionUpdater is used only in case of scheduling errors. If we succeed
	// with scheduling, PodScheduled condition will be updated in apiserver in /bind
	// handler so that binding and setting PodCondition it is atomic.
	PodConditionUpdater PodConditionUpdater
	// PodPreemptor is used to evict pods and update 'NominatedNode' field of
	// the preemptor pod.
	PodPreemptor PodPreemptor
	// Framework runs scheduler plugins at configured extension points.
	Framework framework.Framework

	// NextPod should be a function that blocks until the next pod
	// is available. We don't use a channel for this, because scheduling
	// a pod may take some amount of time and we don't want pods to get
	// stale while they sit in a channel.
	NextPod func() *v1.Pod

	// WaitForCacheSync waits for scheduler cache to populate.
	// It returns true if it was successful, false if the controller should shutdown.
	WaitForCacheSync func() bool

	// Error is called if there is an error. It is passed the pod in
	// question, and the error
	Error func(*v1.Pod, error)

	// Recorder is the EventRecorder to use
	Recorder record.EventRecorder

	// Close this to shut down the scheduler.
	StopEverything <-chan struct{}

	// VolumeBinder handles PVC/PV binding for the pod.
	VolumeBinder *volumebinder.VolumeBinder

	// Disable pod preemption or not.
	DisablePreemption bool

	// SchedulingQueue holds pods to be scheduled
	SchedulingQueue internalqueue.SchedulingQueue
}

// Cache returns the cache in scheduler for test to check the data in scheduler.

func (sched *Scheduler) Cache() internalcache.Cache {
	return sched.SchedulerCache
}

type schedulerOptions struct {
	schedulerName                  string
	hardPodAffinitySymmetricWeight int32
	disablePreemption              bool
	percentageOfNodesToScore       int32
	bindTimeoutSeconds             int64
}

// Option configures a Scheduler
type Option func(*schedulerOptions)

// WithName sets schedulerName for Scheduler, the default schedulerName is default-scheduler
func WithName(schedulerName string) Option {
	return func(o *schedulerOptions) {
		o.schedulerName = schedulerName
	}
}

// WithHardPodAffinitySymmetricWeight sets hardPodAffinitySymmetricWeight for Scheduler, the default value is 1
func WithHardPodAffinitySymmetricWeight(hardPodAffinitySymmetricWeight int32) Option {
	return func(o *schedulerOptions) {
		o.hardPodAffinitySymmetricWeight = hardPodAffinitySymmetricWeight
	}
}

// WithPreemptionDisabled sets disablePreemption for Scheduler, the default value is false
func WithPreemptionDisabled(disablePreemption bool) Option {
	return func(o *schedulerOptions) {
		o.disablePreemption = disablePreemption
	}
}

// WithPercentageOfNodesToScore sets percentageOfNodesToScore for Scheduler, the default value is 50
func WithPercentageOfNodesToScore(percentageOfNodesToScore int32) Option {
	return func(o *schedulerOptions) {
		o.percentageOfNodesToScore = percentageOfNodesToScore
	}
}

// WithBindTimeoutSeconds sets bindTimeoutSeconds for Scheduler, the default value is 100
func WithBindTimeoutSeconds(bindTimeoutSeconds int64) Option {
	return func(o *schedulerOptions) {
		o.bindTimeoutSeconds = bindTimeoutSeconds
	}
}

var defaultSchedulerOptions = schedulerOptions{
	schedulerName:                  v1.DefaultSchedulerName,
	hardPodAffinitySymmetricWeight: v1.DefaultHardPodAffinitySymmetricWeight,
	disablePreemption:              false,
	percentageOfNodesToScore:       schedulerapi.DefaultPercentageOfNodesToScore,
	bindTimeoutSeconds:             BindTimeoutSeconds,
}

// New returns a Scheduler
func New(client clientset.Interface,
	nodeInformer coreinformers.NodeInformer,
	podInformer coreinformers.PodInformer,
	pvInformer coreinformers.PersistentVolumeInformer,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	replicationControllerInformer coreinformers.ReplicationControllerInformer,
	replicaSetInformer appsinformers.ReplicaSetInformer,
	statefulSetInformer appsinformers.StatefulSetInformer,
	serviceInformer coreinformers.ServiceInformer,
	pdbInformer policyinformers.PodDisruptionBudgetInformer,
	storageClassInformer storageinformers.StorageClassInformer,
	recorder record.EventRecorder,
	schedulerAlgorithmSource kubeschedulerconfig.SchedulerAlgorithmSource,
	stopCh <-chan struct{},
	registry framework.Registry,
	opts ...func(o *schedulerOptions)) (*Scheduler, error) {

	options := defaultSchedulerOptions
	for _, opt := range opts {
		opt(&options)
	}
	stopEverything := stopCh
	if stopEverything == nil {
		stopEverything = wait.NeverStop
	}
	schedulerCache := internalcache.New(30*time.Second, stopEverything)

	// TODO(bsalamat): config files should be passed to the framework.
	framework, err := framework.NewFramework(registry, nil)
	if err != nil {
		klog.Fatalf("error initializing the scheduling framework: %v", err)
	}
	// storageClassInformer is only enabled through VolumeScheduling feature gate
	var storageClassLister storagelisters.StorageClassLister
	if storageClassInformer != nil {
		storageClassLister = storageClassInformer.Lister()
	}

	podQueue := internalqueue.NewSchedulingQueue(stopEverything)
	// Setup volume binder
	volumeBinder := volumebinder.NewVolumeBinder(client, nodeInformer, pvcInformer, pvInformer, storageClassInformer, time.Duration(options.bindTimeoutSeconds)*time.Second)
	scheduledPodsHasSynced := podInformer.Informer().HasSynced
	// ScheduledPodLister is something we provide to plug-in functions that
	// they may need to call.
	scheduledPodLister := assignedPodLister{podInformer.Lister()}
	// Setup cache debugger
	debugger := cachedebugger.New(
		nodeInformer.Lister(),
		podInformer.Lister(),
		schedulerCache,
		podQueue,
	)
	debugger.ListenForSignal(stopEverything)

	go func() {
		<-stopEverything
		podQueue.Close()
	}()

	// initiate pluginFactoryArgs
	pluginArgs := &factory.PluginFactoryArgs{
		PodLister:                      schedulerCache,
		ServiceLister:                  serviceInformer.Lister(),
		ControllerLister:               replicationControllerInformer.Lister(),
		ReplicaSetLister:               replicaSetInformer.Lister(),
		StatefulSetLister:              statefulSetInformer.Lister(),
		NodeLister:                     &nodeLister{nodeInformer.Lister()},
		PDBLister:                      pdbInformer.Lister(),
		NodeInfo:                       &predicates.CachedNodeInfo{NodeLister: nodeInformer.Lister()},
		PVInfo:                         &predicates.CachedPersistentVolumeInfo{PersistentVolumeLister: pvInformer.Lister()},
		PVCInfo:                        &predicates.CachedPersistentVolumeClaimInfo{PersistentVolumeClaimLister: pvcInformer.Lister()},
		StorageClassInfo:               &predicates.CachedStorageClassInfo{StorageClassLister: storageClassLister},
		VolumeBinder:                   volumeBinder,
		HardPodAffinitySymmetricWeight: options.hardPodAffinitySymmetricWeight,
	}
	source := schedulerAlgorithmSource
	predicateKeys := sets.NewString()
	priorityKeys := sets.NewString()
	var extenders []algorithm.SchedulerExtender
	var alwaysCheckAllPredicates bool
	switch {
	case source.Provider != nil:
		// Create the config from a named algorithm provider.
		providerName := *source.Provider
		klog.V(2).Infof("Creating scheduler from algorithm provider '%v'", providerName)
		provider, err := factory.GetAlgorithmProvider(providerName)
		if err != nil {
			return nil, err
		}
		// sc, err := CreateFromKeys(provider.FitPredicateKeys, provider.PriorityFunctionKeys, []algorithm.SchedulerExtender{})
		// if err != nil {
		// 	return nil, fmt.Errorf("couldn't create scheduler using provider %q: %v", *source.Provider, err)
		// }
		// config = sc
	case source.Policy != nil:
		// Create the config from a user specified policy source.
		policy := &schedulerapi.Policy{}
		switch {
		case source.Policy.File != nil:
			if err := initPolicyFromFile(source.Policy.File.Path, policy); err != nil {
				return nil, err
			}
		case source.Policy.ConfigMap != nil:
			if err := initPolicyFromConfigMap(client, source.Policy.ConfigMap, policy); err != nil {
				return nil, err
			}
		}
		klog.V(2).Infof("Creating scheduler from configuration: %v", policy)

		// validate the policy configuration
		if err := validation.ValidatePolicy(*policy); err != nil {
			return nil, err
		}
		if policy.Predicates == nil {
			klog.V(2).Infof("Using predicates from algorithm provider '%v'", factory.DefaultProvider)
			provider, err := factory.GetAlgorithmProvider(factory.DefaultProvider)
			if err != nil {
				return nil, err
			}
			predicateKeys = provider.FitPredicateKeys
		} else {
			for _, predicate := range policy.Predicates {
				klog.V(2).Infof("Registering predicate: %s", predicate.Name)
				predicateKeys.Insert(factory.RegisterCustomFitPredicate(predicate))
			}
		}

		if policy.Priorities == nil {
			klog.V(2).Infof("Using priorities from algorithm provider '%v'", factory.DefaultProvider)
			provider, err := factory.GetAlgorithmProvider(factory.DefaultProvider)
			if err != nil {
				return nil, err
			}
			priorityKeys = provider.PriorityFunctionKeys
		} else {
			for _, priority := range policy.Priorities {
				klog.V(2).Infof("Registering priority: %s", priority.Name)
				priorityKeys.Insert(factory.RegisterCustomPriorityFunction(priority))
			}
		}
		if len(policy.ExtenderConfigs) != 0 {
			ignoredExtendedResources := sets.NewString()
			for ii := range policy.ExtenderConfigs {
				klog.V(2).Infof("Creating extender with config %+v", policy.ExtenderConfigs[ii])
				extender, err := core.NewHTTPExtender(&policy.ExtenderConfigs[ii])
				if err != nil {
					return nil, err
				}
				extenders = append(extenders, extender)
				for _, r := range policy.ExtenderConfigs[ii].ManagedResources {
					if r.IgnoredByScheduler {
						ignoredExtendedResources.Insert(string(r.Name))
					}
				}
			}
			predicates.RegisterPredicateMetadataProducerWithExtendedResourceOptions(ignoredExtendedResources)
		}
		// Providing HardPodAffinitySymmetricWeight in the policy config is the new and preferred way of providing the value.
		// Give it higher precedence than scheduler CLI configuration when it is provided.
		if policy.HardPodAffinitySymmetricWeight != 0 {
			hardPodAffinitySymmetricWeight := policy.HardPodAffinitySymmetricWeight
		}
		// When AlwaysCheckAllPredicates is set to true, scheduler checks all the configured
		// predicates even after one or more of them fails.
		if policy.AlwaysCheckAllPredicates {
			alwaysCheckAllPredicates = policy.AlwaysCheckAllPredicates
		}
		// sc, err := configurator.CreateFromConfig(*policy)
		// if err != nil {
		// 	return nil, fmt.Errorf("couldn't create scheduler from policy: %v", err)
		// }
		// config = sc
	default:
		return nil, fmt.Errorf("unsupported algorithm source: %v", source)
	}

	if options.hardPodAffinitySymmetricWeight < 1 || options.hardPodAffinitySymmetricWeight > 100 {
		return nil, fmt.Errorf("invalid hardPodAffinitySymmetricWeight: %d, must be in the range 1-100", options.hardPodAffinitySymmetricWeight)
	}

	predicateFuncs, err := factory.GetFitPredicateFunctions(predicateKeys, *pluginArgs)
	if err != nil {
		return nil, err
	}

	priorityConfigs, err := factory.GetPriorityFunctionConfigs(priorityKeys, *pluginArgs)
	if err != nil {
		return nil, err
	}

	priorityMetaProducer, err := factory.GetPriorityMetadataProducer(*pluginArgs)
	if err != nil {
		return nil, err
	}

	predicateMetaProducer, err := factory.GetPredicateMetadataProducer(*pluginArgs)
	if err != nil {
		return nil, err
	}

	// TODO(bsalamat): the default registrar should be able to process config files.
	pluginSet := plugins.NewDefaultPluginSet(pluginsv1alpha1.NewPluginContext(), &schedulerCache)

	algo := core.NewGenericScheduler(
		schedulerCache,
		podQueue,
		predicateFuncs,
		predicateMetaProducer,
		priorityConfigs,
		priorityMetaProducer,
		pluginSet,
		extenders,
		volumeBinder,
		pvcInformer.Lister(),
		pdbInformer.Lister(),
		alwaysCheckAllPredicates,
		options.disablePreemption,
		options.percentageOfNodesToScore,
	)

	podBackoff := util.CreateDefaultPodBackoff()
	// Additional tweaks to the config produced by the configurator.
	config.Recorder = recorder
	config.DisablePreemption = options.disablePreemption
	config.StopEverything = stopCh

	// Create the scheduler.
	sched := &Scheduler{
		SchedulerCache: schedulerCache,
		// The scheduler only needs to consider schedulable nodes.
		NodeLister:          &nodeLister{nodeInformer.Lister()},
		Algorithm:           algo,
		GetBinder:           getBinderFunc(client, extenders),
		PodConditionUpdater: &podConditionUpdater{client},
		PodPreemptor:        &podPreemptor{client},
		PluginSet:           pluginSet,
		WaitForCacheSync: func() bool {
			return cache.WaitForCacheSync(stopEverything, scheduledPodsHasSynced)
		},
		NextPod:           internalqueue.MakeNextPodFunc(podQueue),
		Error:             MakeDefaultErrorFunc(podBackoff, podQueue),
		StopEverything:    stopEverything,
		VolumeBinder:      volumeBinder,
		SchedulingQueue:   podQueue,
		Recorder:          recorder,
		DisablePreemption: options.disablePreemption,
	}
	AddAllEventHandlers(sched, options.schedulerName, nodeInformer, podInformer, pvInformer, pvcInformer, replicationControllerInformer, replicaSetInformer, statefulSetInformer, serviceInformer, pdbInformer, storageClassInformer)
	return sched, nil

}

// assignedPodLister filters the pods returned from a PodLister to
// only include those that have a node name set.
type assignedPodLister struct {
	corelisters.PodLister
}

// List lists all Pods in the indexer for a given namespace.
func (l assignedPodLister) List(selector labels.Selector) ([]*v1.Pod, error) {
	list, err := l.PodLister.List(selector)
	if err != nil {
		return nil, err
	}
	filtered := make([]*v1.Pod, 0, len(list))
	for _, pod := range list {
		if len(pod.Spec.NodeName) > 0 {
			filtered = append(filtered, pod)
		}
	}
	return filtered, nil
}

// assignedPodNamespaceLister filters the pods returned from a PodNamespaceLister to
// only include those that have a node name set.
type assignedPodNamespaceLister struct {
	corelisters.PodNamespaceLister
}

// List lists all Pods in the indexer for a given namespace.
func (l assignedPodNamespaceLister) List(selector labels.Selector) (ret []*v1.Pod, err error) {
	list, err := l.PodNamespaceLister.List(selector)
	if err != nil {
		return nil, err
	}
	filtered := make([]*v1.Pod, 0, len(list))
	for _, pod := range list {
		if len(pod.Spec.NodeName) > 0 {
			filtered = append(filtered, pod)
		}
	}
	return filtered, nil
}

// Get retrieves the Pod from the indexer for a given namespace and name.
func (l assignedPodNamespaceLister) Get(name string) (*v1.Pod, error) {
	pod, err := l.PodNamespaceLister.Get(name)
	if err != nil {
		return nil, err
	}
	if len(pod.Spec.NodeName) > 0 {
		return pod, nil
	}
	return nil, apierrors.NewNotFound(schema.GroupResource{Resource: string(v1.ResourcePods)}, name)
}

// List lists all Pods in the indexer for a given namespace.
func (l assignedPodLister) Pods(namespace string) corelisters.PodNamespaceLister {
	return assignedPodNamespaceLister{l.PodLister.Pods(namespace)}
}

type nodeLister struct {
	corelisters.NodeLister
}

func (n *nodeLister) List() ([]*v1.Node, error) {
	return n.NodeLister.List(labels.Everything())
}

type podConditionUpdater struct {
	Client clientset.Interface
}

func (p *podConditionUpdater) Update(pod *v1.Pod, condition *v1.PodCondition) error {
	klog.V(3).Infof("Updating pod condition for %s/%s to (%s==%s, Reason=%s)", pod.Namespace, pod.Name, condition.Type, condition.Status, condition.Reason)
	if podutil.UpdatePodCondition(&pod.Status, condition) {
		_, err := p.Client.CoreV1().Pods(pod.Namespace).UpdateStatus(pod)
		return err
	}
	return nil
}

type podPreemptor struct {
	Client clientset.Interface
}

func (p *podPreemptor) GetUpdatedPod(pod *v1.Pod) (*v1.Pod, error) {
	return p.Client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
}

func (p *podPreemptor) DeletePod(pod *v1.Pod) error {
	return p.Client.CoreV1().Pods(pod.Namespace).Delete(pod.Name, &metav1.DeleteOptions{})
}

func (p *podPreemptor) SetNominatedNodeName(pod *v1.Pod, nominatedNodeName string) error {
	podCopy := pod.DeepCopy()
	podCopy.Status.NominatedNodeName = nominatedNodeName
	_, err := p.Client.CoreV1().Pods(pod.Namespace).UpdateStatus(podCopy)
	return err
}

func (p *podPreemptor) RemoveNominatedNodeName(pod *v1.Pod) error {
	if len(pod.Status.NominatedNodeName) == 0 {
		return nil
	}
	return p.SetNominatedNodeName(pod, "")
}

// skipPodUpdate checks whether the specified pod update should be ignored.
// This function will return true if
//   - The pod has already been assumed, AND
//   - The pod has only its ResourceVersion, Spec.NodeName and/or Annotations
//     updated.
func skipPodUpdate(pod *v1.Pod, schedulerCache schedulerinternalcache.Cache) bool {
	// Non-assumed pods should never be skipped.
	isAssumed, err := schedulerCache.IsAssumedPod(pod)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to check whether pod %s/%s is assumed: %v", pod.Namespace, pod.Name, err))
		return false
	}
	if !isAssumed {
		return false
	}

	// Gets the assumed pod from the cache.
	assumedPod, err := schedulerCache.GetPod(pod)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to get assumed pod %s/%s from cache: %v", pod.Namespace, pod.Name, err))
		return false
	}

	// Compares the assumed pod in the cache with the pod update. If they are
	// equal (with certain fields excluded), this pod update will be skipped.
	f := func(pod *v1.Pod) *v1.Pod {
		p := pod.DeepCopy()
		// ResourceVersion must be excluded because each object update will
		// have a new resource version.
		p.ResourceVersion = ""
		// Spec.NodeName must be excluded because the pod assumed in the cache
		// is expected to have a node assigned while the pod update may nor may
		// not have this field set.
		p.Spec.NodeName = ""
		// Annotations must be excluded for the reasons described in
		// https://github.com/kubernetes/kubernetes/issues/52914.
		p.Annotations = nil
		return p
	}
	assumedPodCopy, podCopy := f(assumedPod), f(pod)
	if !reflect.DeepEqual(assumedPodCopy, podCopy) {
		return false
	}
	klog.V(3).Infof("Skipping pod %s/%s update", pod.Namespace, pod.Name)
	return true
}

// initPolicyFromFile initialize policy from file
func initPolicyFromFile(policyFile string, policy *schedulerapi.Policy) error {
	// Use a policy serialized in a file.
	_, err := os.Stat(policyFile)
	if err != nil {
		return fmt.Errorf("missing policy config file %s", policyFile)
	}
	data, err := ioutil.ReadFile(policyFile)
	if err != nil {
		return fmt.Errorf("couldn't read policy config: %v", err)
	}
	err = runtime.DecodeInto(latestschedulerapi.Codec, []byte(data), policy)
	if err != nil {
		return fmt.Errorf("invalid policy: %v", err)
	}
	return nil
}

// initPolicyFromConfigMap initialize policy from configMap
func initPolicyFromConfigMap(client clientset.Interface, policyRef *kubeschedulerconfig.SchedulerPolicyConfigMapSource, policy *schedulerapi.Policy) error {
	// Use a policy serialized in a config map value.
	policyConfigMap, err := client.CoreV1().ConfigMaps(policyRef.Namespace).Get(policyRef.Name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("couldn't get policy config map %s/%s: %v", policyRef.Namespace, policyRef.Name, err)
	}
	data, found := policyConfigMap.Data[kubeschedulerconfig.SchedulerPolicyConfigMapKey]
	if !found {
		return fmt.Errorf("missing policy config map value at key %q", kubeschedulerconfig.SchedulerPolicyConfigMapKey)
	}
	err = runtime.DecodeInto(latestschedulerapi.Codec, []byte(data), policy)
	if err != nil {
		return fmt.Errorf("invalid policy: %v", err)
	}
	return nil
}

// // NewFromConfig returns a new scheduler using the provided Config.
// func NewFromConfig(config *factory.Config) *Scheduler {
// 	metrics.Register()
// 	return &Scheduler{
// 		config: config,
// 	}
// }

// Run begins watching and scheduling. It waits for cache to be synced, then starts a goroutine and returns immediately.
func (sched *Scheduler) Run() {
	if !sched.WaitForCacheSync() {
		return
	}

	go wait.Until(sched.scheduleOne, 0, sched.StopEverything)
}

// Config returns scheduler's config pointer. It is exposed for testing purposes.
// func (sched *Scheduler) Config() *factory.Config {
// 	return sched.config
// }

// recordFailedSchedulingEvent records an event for the pod that indicates the
// pod has failed to schedule.
// NOTE: This function modifies "pod". "pod" should be copied before being passed.
func (sched *Scheduler) recordSchedulingFailure(pod *v1.Pod, err error, reason string, message string) {
	sched.Error(pod, err)
	sched.Recorder.Event(pod, v1.EventTypeWarning, "FailedScheduling", message)
	sched.PodConditionUpdater.Update(pod, &v1.PodCondition{
		Type:    v1.PodScheduled,
		Status:  v1.ConditionFalse,
		Reason:  reason,
		Message: err.Error(),
	})
}

// schedule implements the scheduling algorithm and returns the suggested result(host,
// evaluated nodes number,feasible nodes number).
func (sched *Scheduler) schedule(pod *v1.Pod) (core.ScheduleResult, error) {
	result, err := sched.Algorithm.Schedule(pod, sched.NodeLister)
	if err != nil {
		pod = pod.DeepCopy()
		sched.recordSchedulingFailure(pod, err, v1.PodReasonUnschedulable, err.Error())
		return core.ScheduleResult{}, err
	}
	return result, err
}

// preempt tries to create room for a pod that has failed to schedule, by preempting lower priority pods if possible.
// If it succeeds, it adds the name of the node where preemption has happened to the pod spec.
// It returns the node name and an error if any.
func (sched *Scheduler) preempt(preemptor *v1.Pod, scheduleErr error) (string, error) {
	preemptor, err := sched.PodPreemptor.GetUpdatedPod(preemptor)
	if err != nil {
		klog.Errorf("Error getting the updated preemptor pod object: %v", err)
		return "", err
	}

	node, victims, nominatedPodsToClear, err := sched.Algorithm.Preempt(preemptor, sched.NodeLister, scheduleErr)
	if err != nil {
		klog.Errorf("Error preempting victims to make room for %v/%v.", preemptor.Namespace, preemptor.Name)
		return "", err
	}
	var nodeName = ""
	if node != nil {
		nodeName = node.Name
		// Update the scheduling queue with the nominated pod information. Without
		// this, there would be a race condition between the next scheduling cycle
		// and the time the scheduler receives a Pod Update for the nominated pod.
		sched.SchedulingQueue.UpdateNominatedPodForNode(preemptor, nodeName)

		// Make a call to update nominated node name of the pod on the API server.
		err = sched.PodPreemptor.SetNominatedNodeName(preemptor, nodeName)
		if err != nil {
			klog.Errorf("Error in preemption process. Cannot update pod %v/%v annotations: %v", preemptor.Namespace, preemptor.Name, err)
			sched.SchedulingQueue.DeleteNominatedPodIfExists(preemptor)
			return "", err
		}

		for _, victim := range victims {
			if err := sched.PodPreemptor.DeletePod(victim); err != nil {
				klog.Errorf("Error preempting pod %v/%v: %v", victim.Namespace, victim.Name, err)
				return "", err
			}
			sched.Recorder.Eventf(victim, v1.EventTypeNormal, "Preempted", "by %v/%v on node %v", preemptor.Namespace, preemptor.Name, nodeName)
		}
		metrics.PreemptionVictims.Set(float64(len(victims)))
	}
	// Clearing nominated pods should happen outside of "if node != nil". Node could
	// be nil when a pod with nominated node name is eligible to preempt again,
	// but preemption logic does not find any node for it. In that case Preempt()
	// function of generic_scheduler.go returns the pod itself for removal of
	// the 'NominatedPod' field.
	for _, p := range nominatedPodsToClear {
		rErr := sched.PodPreemptor.RemoveNominatedNodeName(p)
		if rErr != nil {
			klog.Errorf("Cannot remove 'NominatedPod' field of pod: %v", rErr)
			// We do not return as this error is not critical.
		}
	}
	return nodeName, err
}

// assumeVolumes will update the volume cache with the chosen bindings
//
// This function modifies assumed if volume binding is required.
func (sched *Scheduler) assumeVolumes(assumed *v1.Pod, host string) (allBound bool, err error) {
	allBound, err = sched.VolumeBinder.Binder.AssumePodVolumes(assumed, host)
	if err != nil {
		sched.recordSchedulingFailure(assumed, err, SchedulerError,
			fmt.Sprintf("AssumePodVolumes failed: %v", err))
	}
	return
}

// bindVolumes will make the API update with the assumed bindings and wait until
// the PV controller has completely finished the binding operation.
//
// If binding errors, times out or gets undone, then an error will be returned to
// retry scheduling.
func (sched *Scheduler) bindVolumes(assumed *v1.Pod) error {
	klog.V(5).Infof("Trying to bind volumes for pod \"%v/%v\"", assumed.Namespace, assumed.Name)
	err := sched.VolumeBinder.Binder.BindPodVolumes(assumed)
	if err != nil {
		klog.V(1).Infof("Failed to bind volumes for pod \"%v/%v\": %v", assumed.Namespace, assumed.Name, err)

		// Unassume the Pod and retry scheduling
		if forgetErr := sched.SchedulerCache.ForgetPod(assumed); forgetErr != nil {
			klog.Errorf("scheduler cache ForgetPod failed: %v", forgetErr)
		}

		sched.recordSchedulingFailure(assumed, err, "VolumeBindingFailed", err.Error())
		return err
	}

	klog.V(5).Infof("Success binding volumes for pod \"%v/%v\"", assumed.Namespace, assumed.Name)
	return nil
}

// assume signals to the cache that a pod is already in the cache, so that binding can be asynchronous.
// assume modifies `assumed`.
func (sched *Scheduler) assume(assumed *v1.Pod, host string) error {
	// Optimistically assume that the binding will succeed and send it to apiserver
	// in the background.
	// If the binding fails, scheduler will release resources allocated to assumed pod
	// immediately.
	assumed.Spec.NodeName = host

	if err := sched.SchedulerCache.AssumePod(assumed); err != nil {
		klog.Errorf("scheduler cache AssumePod failed: %v", err)

		// This is most probably result of a BUG in retrying logic.
		// We report an error here so that pod scheduling can be retried.
		// This relies on the fact that Error will check if the pod has been bound
		// to a node and if so will not add it back to the unscheduled pods queue
		// (otherwise this would cause an infinite loop).
		sched.recordSchedulingFailure(assumed, err, SchedulerError,
			fmt.Sprintf("AssumePod failed: %v", err))
		return err
	}
	// if "assumed" is a nominated pod, we should remove it from internal cache
	if sched.SchedulingQueue != nil {
		sched.SchedulingQueue.DeleteNominatedPodIfExists(assumed)
	}

	return nil
}

// bind binds a pod to a given node defined in a binding object.  We expect this to run asynchronously, so we
// handle binding metrics internally.
func (sched *Scheduler) bind(assumed *v1.Pod, b *v1.Binding) error {
	bindingStart := time.Now()
	// If binding succeeded then PodScheduled condition will be updated in apiserver so that
	// it's atomic with setting host.
	err := sched.GetBinder(assumed).Bind(b)
	if finErr := sched.SchedulerCache.FinishBinding(assumed); finErr != nil {
		klog.Errorf("scheduler cache FinishBinding failed: %v", finErr)
	}
	if err != nil {
		klog.V(1).Infof("Failed to bind pod: %v/%v", assumed.Namespace, assumed.Name)
		if err := sched.SchedulerCache.ForgetPod(assumed); err != nil {
			klog.Errorf("scheduler cache ForgetPod failed: %v", err)
		}
		sched.recordSchedulingFailure(assumed, err, SchedulerError,
			fmt.Sprintf("Binding rejected: %v", err))
		return err
	}

	metrics.BindingLatency.Observe(metrics.SinceInSeconds(bindingStart))
	metrics.DeprecatedBindingLatency.Observe(metrics.SinceInMicroseconds(bindingStart))
	metrics.SchedulingLatency.WithLabelValues(metrics.Binding).Observe(metrics.SinceInSeconds(bindingStart))
	metrics.DeprecatedSchedulingLatency.WithLabelValues(metrics.Binding).Observe(metrics.SinceInSeconds(bindingStart))
	sched.Recorder.Eventf(assumed, v1.EventTypeNormal, "Scheduled", "Successfully assigned %v/%v to %v", assumed.Namespace, assumed.Name, b.Target.Name)
	return nil
}

// scheduleOne does the entire scheduling workflow for a single pod.  It is serialized on the scheduling algorithm's host fitting.
func (sched *Scheduler) scheduleOne() {
<<<<<<< HEAD
	fwk := sched.config.Framework
=======
	plugins := sched.PluginSet
	// Remove all plugin context data at the beginning of a scheduling cycle.
	if plugins.Data().Ctx != nil {
		plugins.Data().Ctx.Reset()
	}
>>>>>>> flatten Scheduler struct

	pod := sched.NextPod()
	// pod could be nil when schedulerQueue is closed
	if pod == nil {
		return
	}
	if pod.DeletionTimestamp != nil {
		sched.Recorder.Eventf(pod, v1.EventTypeWarning, "FailedScheduling", "skip schedule deleting pod: %v/%v", pod.Namespace, pod.Name)
		klog.V(3).Infof("Skip schedule deleting pod: %v/%v", pod.Namespace, pod.Name)
		return
	}

	klog.V(3).Infof("Attempting to schedule pod: %v/%v", pod.Namespace, pod.Name)

	// Synchronously attempt to find a fit for the pod.
	start := time.Now()
	pluginContext := framework.NewPluginContext()
	scheduleResult, err := sched.schedule(pod)
	if err != nil {
		// schedule() may have failed because the pod would not fit on any host, so we try to
		// preempt, with the expectation that the next time the pod is tried for scheduling it
		// will fit due to the preemption. It is also possible that a different pod will schedule
		// into the resources that were preempted, but this is harmless.
		if fitError, ok := err.(*core.FitError); ok {
			if !util.PodPriorityEnabled() || sched.DisablePreemption {
				klog.V(3).Infof("Pod priority feature is not enabled or preemption is disabled by scheduler configuration." +
					" No preemption is performed.")
			} else {
				preemptionStartTime := time.Now()
				sched.preempt(pod, fitError)
				metrics.PreemptionAttempts.Inc()
				metrics.SchedulingAlgorithmPremptionEvaluationDuration.Observe(metrics.SinceInSeconds(preemptionStartTime))
				metrics.DeprecatedSchedulingAlgorithmPremptionEvaluationDuration.Observe(metrics.SinceInMicroseconds(preemptionStartTime))
				metrics.SchedulingLatency.WithLabelValues(metrics.PreemptionEvaluation).Observe(metrics.SinceInSeconds(preemptionStartTime))
				metrics.DeprecatedSchedulingLatency.WithLabelValues(metrics.PreemptionEvaluation).Observe(metrics.SinceInSeconds(preemptionStartTime))
			}
			// Pod did not fit anywhere, so it is counted as a failure. If preemption
			// succeeds, the pod should get counted as a success the next time we try to
			// schedule it. (hopefully)
			metrics.PodScheduleFailures.Inc()
		} else {
			klog.Errorf("error selecting node for pod: %v", err)
			metrics.PodScheduleErrors.Inc()
		}
		return
	}
	metrics.SchedulingAlgorithmLatency.Observe(metrics.SinceInSeconds(start))
	metrics.DeprecatedSchedulingAlgorithmLatency.Observe(metrics.SinceInMicroseconds(start))
	// Tell the cache to assume that a pod now is running on a given node, even though it hasn't been bound yet.
	// This allows us to keep scheduling without waiting on binding to occur.
	assumedPod := pod.DeepCopy()

	// Assume volumes first before assuming the pod.
	//
	// If all volumes are completely bound, then allBound is true and binding will be skipped.
	//
	// Otherwise, binding of volumes is started after the pod is assumed, but before pod binding.
	//
	// This function modifies 'assumedPod' if volume binding is required.
	allBound, err := sched.assumeVolumes(assumedPod, scheduleResult.SuggestedHost)
	if err != nil {
		klog.Errorf("error assuming volumes: %v", err)
		metrics.PodScheduleErrors.Inc()
		return
	}

	// Run "reserve" plugins.
	if sts := fwk.RunReservePlugins(pluginContext, assumedPod, scheduleResult.SuggestedHost); !sts.IsSuccess() {
		sched.recordSchedulingFailure(assumedPod, sts.AsError(), SchedulerError, sts.Message())
		metrics.PodScheduleErrors.Inc()
		return
	}

	// assume modifies `assumedPod` by setting NodeName=scheduleResult.SuggestedHost
	err = sched.assume(assumedPod, scheduleResult.SuggestedHost)
	if err != nil {
		klog.Errorf("error assuming pod: %v", err)
		metrics.PodScheduleErrors.Inc()
		return
	}
	// bind the pod to its host asynchronously (we can do this b/c of the assumption step above).
	go func() {
		// Bind volumes first before Pod
		if !allBound {
			err := sched.bindVolumes(assumedPod)
			if err != nil {
				klog.Errorf("error binding volumes: %v", err)
				metrics.PodScheduleErrors.Inc()
				return
			}
		}

		// Run "prebind" plugins.
		prebindStatus := fwk.RunPrebindPlugins(pluginContext, assumedPod, scheduleResult.SuggestedHost)
		if !prebindStatus.IsSuccess() {
			var reason string
			if prebindStatus.Code() == framework.Unschedulable {
				reason = v1.PodReasonUnschedulable
			} else {
				metrics.PodScheduleErrors.Inc()
				reason = SchedulerError
			}
			if forgetErr := sched.Cache().ForgetPod(assumedPod); forgetErr != nil {
				klog.Errorf("scheduler cache ForgetPod failed: %v", forgetErr)
			}
			sched.recordSchedulingFailure(assumedPod, prebindStatus.AsError(), reason, prebindStatus.Message())
			return
		}

		err := sched.bind(assumedPod, &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Namespace: assumedPod.Namespace, Name: assumedPod.Name, UID: assumedPod.UID},
			Target: v1.ObjectReference{
				Kind: "Node",
				Name: scheduleResult.SuggestedHost,
			},
		})
		metrics.E2eSchedulingLatency.Observe(metrics.SinceInSeconds(start))
		metrics.DeprecatedE2eSchedulingLatency.Observe(metrics.SinceInMicroseconds(start))
		if err != nil {
			klog.Errorf("error binding pod: %v", err)
			metrics.PodScheduleErrors.Inc()
		} else {
			klog.V(2).Infof("pod %v/%v is bound successfully on node %v, %d nodes evaluated, %d nodes were found feasible", assumedPod.Namespace, assumedPod.Name, scheduleResult.SuggestedHost, scheduleResult.EvaluatedNodes, scheduleResult.FeasibleNodes)
			metrics.PodScheduleSuccesses.Inc()
		}
	}()
}
