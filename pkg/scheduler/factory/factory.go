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

// Package factory can set up a scheduler. This code is here instead of
// cmd/scheduler for both testability and reuse.
package factory

import (
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	appsinformers "k8s.io/client-go/informers/apps/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	policyinformers "k8s.io/client-go/informers/policy/v1beta1"
	storageinformers "k8s.io/client-go/informers/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	policylisters "k8s.io/client-go/listers/policy/v1beta1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/pkg/scheduler/api/validation"
	"k8s.io/kubernetes/pkg/scheduler/core"
	schedulerinternalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	cachedebugger "k8s.io/kubernetes/pkg/scheduler/internal/cache/debugger"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/plugins"
	pluginsv1alpha1 "k8s.io/kubernetes/pkg/scheduler/plugins/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"
)

const (
	initialGetBackoff = 100 * time.Millisecond
	maximalGetBackoff = time.Minute
)

// Binder knows how to write a binding.
type Binder interface {
	Bind(binding *v1.Binding) error
}

// PodConditionUpdater updates the condition of a pod based on the passed
// PodCondition
type PodConditionUpdater interface {
	Update(pod *v1.Pod, podCondition *v1.PodCondition) error
}

// Config is an implementation of the Scheduler's configured input data.
// TODO over time we should make this struct a hidden implementation detail of the scheduler.
type Config struct {
	// It is expected that changes made via SchedulerCache will be observed
	// by NodeLister and Algorithm.
	SchedulerCache schedulerinternalcache.Cache

	NodeLister algorithm.NodeLister
	Algorithm  core.ScheduleAlgorithm
	GetBinder  func(pod *v1.Pod) Binder
	// PodConditionUpdater is used only in case of scheduling errors. If we succeed
	// with scheduling, PodScheduled condition will be updated in apiserver in /bind
	// handler so that binding and setting PodCondition it is atomic.
	PodConditionUpdater PodConditionUpdater
	// PodPreemptor is used to evict pods and update pod annotations.
	PodPreemptor PodPreemptor
	// PlugingSet has a set of plugins and data used to run them.
	PluginSet pluginsv1alpha1.PluginSet

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

// PodPreemptor has methods needed to delete a pod and to update
// annotations of the preemptor pod.
type PodPreemptor interface {
	GetUpdatedPod(pod *v1.Pod) (*v1.Pod, error)
	DeletePod(pod *v1.Pod) error
	SetNominatedNodeName(pod *v1.Pod, nominatedNode string) error
	RemoveNominatedNodeName(pod *v1.Pod) error
}

// Configurator defines I/O, caching, and other functionality needed to
// construct a new scheduler. An implementation of this can be seen in
// factory.go.
type Configurator interface {
	// Exposed for testing
	GetHardPodAffinitySymmetricWeight() int32

	// Predicate related accessors to be exposed for use by k8s.io/autoscaler/cluster-autoscaler
	GetPredicateMetadataProducer() (predicates.PredicateMetadataProducer, error)
	GetPredicates(predicateKeys sets.String) (map[string]predicates.FitPredicate, error)

	// Needs to be exposed for things like integration tests where we want to make fake nodes.
	GetNodeLister() corelisters.NodeLister
	// Exposed for testing
	GetClient() clientset.Interface
	// Exposed for testing
	GetScheduledPodLister() corelisters.PodLister

	Create() (*Config, error)
	CreateFromProvider(providerName string) (*Config, error)
	CreateFromConfig(policy schedulerapi.Policy) (*Config, error)
	CreateFromKeys(predicateKeys, priorityKeys sets.String, extenders []algorithm.SchedulerExtender) (*Config, error)
}

// configFactory is the default implementation of the scheduler.Configurator interface.
type configFactory struct {
	client clientset.Interface
	// a means to list all known scheduled pods.
	scheduledPodLister corelisters.PodLister
	// a means to list all known scheduled pods and pods assumed to have been scheduled.
	podLister algorithm.PodLister
	// a means to list all nodes
	nodeLister corelisters.NodeLister
	// a means to list all PersistentVolumes
	pVLister corelisters.PersistentVolumeLister
	// a means to list all PersistentVolumeClaims
	pVCLister corelisters.PersistentVolumeClaimLister
	// a means to list all services
	serviceLister corelisters.ServiceLister
	// a means to list all controllers
	controllerLister corelisters.ReplicationControllerLister
	// a means to list all replicasets
	replicaSetLister appslisters.ReplicaSetLister
	// a means to list all statefulsets
	statefulSetLister appslisters.StatefulSetLister
	// a means to list all PodDisruptionBudgets
	pdbLister policylisters.PodDisruptionBudgetLister
	// a means to list all StorageClasses
	storageClassLister storagelisters.StorageClassLister
	// pluginRunner has a set of plugins and the context used for running them.
	pluginSet pluginsv1alpha1.PluginSet

	// Close this to stop all reflectors
	StopEverything <-chan struct{}

	scheduledPodsHasSynced cache.InformerSynced

	schedulerCache schedulerinternalcache.Cache

	// SchedulerName of a scheduler is used to select which pods will be
	// processed by this scheduler, based on pods's "spec.schedulerName".
	schedulerName string

	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 0-100.
	hardPodAffinitySymmetricWeight int32

	// Handles volume binding decisions
	volumeBinder *volumebinder.VolumeBinder

	// Always check all predicates even if the middle of one predicate fails.
	alwaysCheckAllPredicates bool

	// Disable pod preemption or not.
	disablePreemption bool

	// percentageOfNodesToScore specifies percentage of all nodes to score in each scheduling cycle.
	percentageOfNodesToScore int32

	bindTimeoutSeconds int64
	// queue for pods that need scheduling
	podQueue internalqueue.SchedulingQueue
}

// ConfigFactoryArgs is a set arguments passed to NewConfigFactory.
type ConfigFactoryArgs struct {
	SchedulerName                  string
	Client                         clientset.Interface
	NodeInformer                   coreinformers.NodeInformer
	PodInformer                    coreinformers.PodInformer
	PvInformer                     coreinformers.PersistentVolumeInformer
	PvcInformer                    coreinformers.PersistentVolumeClaimInformer
	ReplicationControllerInformer  coreinformers.ReplicationControllerInformer
	ReplicaSetInformer             appsinformers.ReplicaSetInformer
	StatefulSetInformer            appsinformers.StatefulSetInformer
	ServiceInformer                coreinformers.ServiceInformer
	PdbInformer                    policyinformers.PodDisruptionBudgetInformer
	StorageClassInformer           storageinformers.StorageClassInformer
	HardPodAffinitySymmetricWeight int32
	DisablePreemption              bool
	PercentageOfNodesToScore       int32
	BindTimeoutSeconds             int64
	StopCh                         <-chan struct{}
}

// NewConfigFactory initializes the default implementation of a Configurator. To encourage eventual privatization of the struct type, we only
// return the interface.
func NewConfigFactory(args *ConfigFactoryArgs) Configurator {
	stopEverything := args.StopCh
	if stopEverything == nil {
		stopEverything = wait.NeverStop
	}
	schedulerCache := schedulerinternalcache.New(30*time.Second, stopEverything)

	// storageClassInformer is only enabled through VolumeScheduling feature gate
	var storageClassLister storagelisters.StorageClassLister
	if args.StorageClassInformer != nil {
		storageClassLister = args.StorageClassInformer.Lister()
	}
	c := &configFactory{
		client:                         args.Client,
		podLister:                      schedulerCache,
		podQueue:                       internalqueue.NewSchedulingQueue(stopEverything),
		nodeLister:                     args.NodeInformer.Lister(),
		pVLister:                       args.PvInformer.Lister(),
		pVCLister:                      args.PvcInformer.Lister(),
		serviceLister:                  args.ServiceInformer.Lister(),
		controllerLister:               args.ReplicationControllerInformer.Lister(),
		replicaSetLister:               args.ReplicaSetInformer.Lister(),
		statefulSetLister:              args.StatefulSetInformer.Lister(),
		pdbLister:                      args.PdbInformer.Lister(),
		storageClassLister:             storageClassLister,
		schedulerCache:                 schedulerCache,
		StopEverything:                 stopEverything,
		schedulerName:                  args.SchedulerName,
		hardPodAffinitySymmetricWeight: args.HardPodAffinitySymmetricWeight,
		disablePreemption:              args.DisablePreemption,
		percentageOfNodesToScore:       args.PercentageOfNodesToScore,
		bindTimeoutSeconds:             args.BindTimeoutSeconds,
	}
	// Setup volume binder
	c.volumeBinder = volumebinder.NewVolumeBinder(args.Client, args.NodeInformer, args.PvcInformer, args.PvInformer, args.StorageClassInformer, time.Duration(args.BindTimeoutSeconds)*time.Second)
	c.scheduledPodsHasSynced = args.PodInformer.Informer().HasSynced
	// ScheduledPodLister is something we provide to plug-in functions that
	// they may need to call.
	c.scheduledPodLister = assignedPodLister{args.PodInformer.Lister()}

	// Setup cache debugger
	debugger := cachedebugger.New(
		args.NodeInformer.Lister(),
		args.PodInformer.Lister(),
		c.schedulerCache,
		c.podQueue,
	)
	debugger.ListenForSignal(c.StopEverything)

	go func() {
		<-c.StopEverything
		c.podQueue.Close()
	}()
	return c
}

// GetNodeStore provides the cache to the nodes, mostly internal use, but may also be called by mock-tests.
func (c *configFactory) GetNodeLister() corelisters.NodeLister {
	return c.nodeLister
}

func (c *configFactory) GetHardPodAffinitySymmetricWeight() int32 {
	return c.hardPodAffinitySymmetricWeight
}

func (c *configFactory) GetSchedulerName() string {
	return c.schedulerName
}

// GetClient provides a kubernetes Client, mostly internal use, but may also be called by mock-tests.
func (c *configFactory) GetClient() clientset.Interface {
	return c.client
}

// GetScheduledPodLister provides a pod lister, mostly internal use, but may also be called by mock-tests.
func (c *configFactory) GetScheduledPodLister() corelisters.PodLister {
	return c.scheduledPodLister
}

// Create creates a scheduler with the default algorithm provider.
func (c *configFactory) Create() (*Config, error) {
	return c.CreateFromProvider(DefaultProvider)
}

// Creates a scheduler from the name of a registered algorithm provider.
func (c *configFactory) CreateFromProvider(providerName string) (*Config, error) {
	klog.V(2).Infof("Creating scheduler from algorithm provider '%v'", providerName)
	provider, err := GetAlgorithmProvider(providerName)
	if err != nil {
		return nil, err
	}
	return c.CreateFromKeys(provider.FitPredicateKeys, provider.PriorityFunctionKeys, []algorithm.SchedulerExtender{})
}

// Creates a scheduler from the configuration file
func (c *configFactory) CreateFromConfig(policy schedulerapi.Policy) (*Config, error) {
	klog.V(2).Infof("Creating scheduler from configuration: %v", policy)

	// validate the policy configuration
	if err := validation.ValidatePolicy(policy); err != nil {
		return nil, err
	}

	predicateKeys := sets.NewString()
	if policy.Predicates == nil {
		klog.V(2).Infof("Using predicates from algorithm provider '%v'", DefaultProvider)
		provider, err := GetAlgorithmProvider(DefaultProvider)
		if err != nil {
			return nil, err
		}
		predicateKeys = provider.FitPredicateKeys
	} else {
		for _, predicate := range policy.Predicates {
			klog.V(2).Infof("Registering predicate: %s", predicate.Name)
			predicateKeys.Insert(RegisterCustomFitPredicate(predicate))
		}
	}

	priorityKeys := sets.NewString()
	if policy.Priorities == nil {
		klog.V(2).Infof("Using priorities from algorithm provider '%v'", DefaultProvider)
		provider, err := GetAlgorithmProvider(DefaultProvider)
		if err != nil {
			return nil, err
		}
		priorityKeys = provider.PriorityFunctionKeys
	} else {
		for _, priority := range policy.Priorities {
			klog.V(2).Infof("Registering priority: %s", priority.Name)
			priorityKeys.Insert(RegisterCustomPriorityFunction(priority))
		}
	}

	var extenders []algorithm.SchedulerExtender
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
		c.hardPodAffinitySymmetricWeight = policy.HardPodAffinitySymmetricWeight
	}
	// When AlwaysCheckAllPredicates is set to true, scheduler checks all the configured
	// predicates even after one or more of them fails.
	if policy.AlwaysCheckAllPredicates {
		c.alwaysCheckAllPredicates = policy.AlwaysCheckAllPredicates
	}

	return c.CreateFromKeys(predicateKeys, priorityKeys, extenders)
}

// Creates a scheduler from a set of registered fit predicate keys and priority keys.
func (c *configFactory) CreateFromKeys(predicateKeys, priorityKeys sets.String, extenders []algorithm.SchedulerExtender) (*Config, error) {
	klog.V(2).Infof("Creating scheduler with fit predicates '%v' and priority functions '%v'", predicateKeys, priorityKeys)

	if c.GetHardPodAffinitySymmetricWeight() < 1 || c.GetHardPodAffinitySymmetricWeight() > 100 {
		return nil, fmt.Errorf("invalid hardPodAffinitySymmetricWeight: %d, must be in the range 1-100", c.GetHardPodAffinitySymmetricWeight())
	}

	predicateFuncs, err := c.GetPredicates(predicateKeys)
	if err != nil {
		return nil, err
	}

	priorityConfigs, err := c.GetPriorityFunctionConfigs(priorityKeys)
	if err != nil {
		return nil, err
	}

	priorityMetaProducer, err := c.GetPriorityMetadataProducer()
	if err != nil {
		return nil, err
	}

	predicateMetaProducer, err := c.GetPredicateMetadataProducer()
	if err != nil {
		return nil, err
	}

	// TODO(bsalamat): the default registrar should be able to process config files.
	c.pluginSet = plugins.NewDefaultPluginSet(pluginsv1alpha1.NewPluginContext(), &c.schedulerCache)

	algo := core.NewGenericScheduler(
		c.schedulerCache,
		c.podQueue,
		predicateFuncs,
		predicateMetaProducer,
		priorityConfigs,
		priorityMetaProducer,
		c.pluginSet,
		extenders,
		c.volumeBinder,
		c.pVCLister,
		c.pdbLister,
		c.alwaysCheckAllPredicates,
		c.disablePreemption,
		c.percentageOfNodesToScore,
	)

	podBackoff := util.CreateDefaultPodBackoff()
	return &Config{
		SchedulerCache: c.schedulerCache,
		// The scheduler only needs to consider schedulable nodes.
		NodeLister:          &nodeLister{c.nodeLister},
		Algorithm:           algo,
		GetBinder:           getBinderFunc(c.client, extenders),
		PodConditionUpdater: &podConditionUpdater{c.client},
		PodPreemptor:        &podPreemptor{c.client},
		PluginSet:           c.pluginSet,
		WaitForCacheSync: func() bool {
			return cache.WaitForCacheSync(c.StopEverything, c.scheduledPodsHasSynced)
		},
		NextPod:         internalqueue.MakeNextPodFunc(c.podQueue),
		Error:           MakeDefaultErrorFunc(c.client, podBackoff, c.podQueue, c.schedulerCache, c.StopEverything),
		StopEverything:  c.StopEverything,
		VolumeBinder:    c.volumeBinder,
		SchedulingQueue: c.podQueue,
	}, nil
}

// getBinderFunc returns a func which returns an extender that supports bind or a default binder based on the given pod.
func getBinderFunc(client clientset.Interface, extenders []algorithm.SchedulerExtender) func(pod *v1.Pod) Binder {
	var extenderBinder algorithm.SchedulerExtender
	for i := range extenders {
		if extenders[i].IsBinder() {
			extenderBinder = extenders[i]
			break
		}
	}
	defaultBinder := &binder{client}
	return func(pod *v1.Pod) Binder {
		if extenderBinder != nil && extenderBinder.IsInterested(pod) {
			return extenderBinder
		}
		return defaultBinder
	}
}

type nodeLister struct {
	corelisters.NodeLister
}

func (n *nodeLister) List() ([]*v1.Node, error) {
	return n.NodeLister.List(labels.Everything())
}

func (c *configFactory) GetPriorityFunctionConfigs(priorityKeys sets.String) ([]priorities.PriorityConfig, error) {
	pluginArgs, err := c.getPluginArgs()
	if err != nil {
		return nil, err
	}

	return getPriorityFunctionConfigs(priorityKeys, *pluginArgs)
}

func (c *configFactory) GetPriorityMetadataProducer() (priorities.PriorityMetadataProducer, error) {
	pluginArgs, err := c.getPluginArgs()
	if err != nil {
		return nil, err
	}

	return getPriorityMetadataProducer(*pluginArgs)
}

func (c *configFactory) GetPredicateMetadataProducer() (predicates.PredicateMetadataProducer, error) {
	pluginArgs, err := c.getPluginArgs()
	if err != nil {
		return nil, err
	}
	return getPredicateMetadataProducer(*pluginArgs)
}

func (c *configFactory) GetPredicates(predicateKeys sets.String) (map[string]predicates.FitPredicate, error) {
	pluginArgs, err := c.getPluginArgs()
	if err != nil {
		return nil, err
	}

	return getFitPredicateFunctions(predicateKeys, *pluginArgs)
}

func (c *configFactory) getPluginArgs() (*PluginFactoryArgs, error) {
	return &PluginFactoryArgs{
		PodLister:                      c.podLister,
		ServiceLister:                  c.serviceLister,
		ControllerLister:               c.controllerLister,
		ReplicaSetLister:               c.replicaSetLister,
		StatefulSetLister:              c.statefulSetLister,
		NodeLister:                     &nodeLister{c.nodeLister},
		PDBLister:                      c.pdbLister,
		NodeInfo:                       &predicates.CachedNodeInfo{NodeLister: c.nodeLister},
		PVInfo:                         &predicates.CachedPersistentVolumeInfo{PersistentVolumeLister: c.pVLister},
		PVCInfo:                        &predicates.CachedPersistentVolumeClaimInfo{PersistentVolumeClaimLister: c.pVCLister},
		StorageClassInfo:               &predicates.CachedStorageClassInfo{StorageClassLister: c.storageClassLister},
		VolumeBinder:                   c.volumeBinder,
		HardPodAffinitySymmetricWeight: c.hardPodAffinitySymmetricWeight,
	}, nil
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

// List lists all Pods in the indexer for a given namespace.
func (l assignedPodLister) Pods(namespace string) corelisters.PodNamespaceLister {
	return assignedPodNamespaceLister{l.PodLister.Pods(namespace)}
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
	return nil, errors.NewNotFound(schema.GroupResource{Resource: string(v1.ResourcePods)}, name)
}

type podInformer struct {
	informer cache.SharedIndexInformer
}

func (i *podInformer) Informer() cache.SharedIndexInformer {
	return i.informer
}

func (i *podInformer) Lister() corelisters.PodLister {
	return corelisters.NewPodLister(i.informer.GetIndexer())
}

// NewPodInformer creates a shared index informer that returns only non-terminal pods.
func NewPodInformer(client clientset.Interface, resyncPeriod time.Duration) coreinformers.PodInformer {
	selector := fields.ParseSelectorOrDie(
		"status.phase!=" + string(v1.PodSucceeded) +
			",status.phase!=" + string(v1.PodFailed))
	lw := cache.NewListWatchFromClient(client.CoreV1().RESTClient(), string(v1.ResourcePods), metav1.NamespaceAll, selector)
	return &podInformer{
		informer: cache.NewSharedIndexInformer(lw, &v1.Pod{}, resyncPeriod, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}),
	}
}

// MakeDefaultErrorFunc construct a function to handle pod scheduler error
func MakeDefaultErrorFunc(client clientset.Interface, backoff *util.PodBackoff, podQueue internalqueue.SchedulingQueue, schedulerCache schedulerinternalcache.Cache, stopEverything <-chan struct{}) func(pod *v1.Pod, err error) {
	return func(pod *v1.Pod, err error) {
		if err == core.ErrNoNodesAvailable {
			klog.V(4).Infof("Unable to schedule %v/%v: no nodes are registered to the cluster; waiting", pod.Namespace, pod.Name)
		} else {
			if _, ok := err.(*core.FitError); ok {
				klog.V(4).Infof("Unable to schedule %v/%v: no fit: %v; waiting", pod.Namespace, pod.Name, err)
			} else if errors.IsNotFound(err) {
				if errStatus, ok := err.(errors.APIStatus); ok && errStatus.Status().Details.Kind == "node" {
					nodeName := errStatus.Status().Details.Name
					// when node is not found, We do not remove the node right away. Trying again to get
					// the node and if the node is still not found, then remove it from the scheduler cache.
					_, err := client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
					if err != nil && errors.IsNotFound(err) {
						node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}
						schedulerCache.RemoveNode(&node)
					}
				}
			} else {
				klog.Errorf("Error scheduling %v/%v: %v; retrying", pod.Namespace, pod.Name, err)
			}
		}

		backoff.Gc()
		podSchedulingCycle := podQueue.SchedulingCycle()
		// Retry asynchronously.
		// Note that this is extremely rudimentary and we need a more real error handling path.
		go func() {
			defer runtime.HandleCrash()
			podID := types.NamespacedName{
				Namespace: pod.Namespace,
				Name:      pod.Name,
			}

			// When pod priority is enabled, we would like to place an unschedulable
			// pod in the unschedulable queue. This ensures that if the pod is nominated
			// to run on a node, scheduler takes the pod into account when running
			// predicates for the node.
			if !util.PodPriorityEnabled() {
				if !backoff.TryBackoffAndWait(podID, stopEverything) {
					klog.Warningf("Request for pod %v already in flight, abandoning", podID)
					return
				}
			}
			// Get the pod again; it may have changed/been scheduled already.
			getBackoff := initialGetBackoff
			for {
				pod, err := client.CoreV1().Pods(podID.Namespace).Get(podID.Name, metav1.GetOptions{})
				if err == nil {
					if len(pod.Spec.NodeName) == 0 {
						podQueue.AddUnschedulableIfNotPresent(pod, podSchedulingCycle)
					}
					break
				}
				if errors.IsNotFound(err) {
					klog.Warningf("A pod %v no longer exists", podID)
					return
				}
				klog.Errorf("Error getting pod %v for retry: %v; retrying...", podID, err)
				if getBackoff = getBackoff * 2; getBackoff > maximalGetBackoff {
					getBackoff = maximalGetBackoff
				}
				time.Sleep(getBackoff)
			}
		}()
	}
}

// nodeEnumerator allows a cache.Poller to enumerate items in a v1.NodeList
type nodeEnumerator struct {
	*v1.NodeList
}

// Len returns the number of items in the node list.
func (ne *nodeEnumerator) Len() int {
	if ne.NodeList == nil {
		return 0
	}
	return len(ne.Items)
}

// Get returns the item (and ID) with the particular index.
func (ne *nodeEnumerator) Get(index int) interface{} {
	return &ne.Items[index]
}

type binder struct {
	Client clientset.Interface
}

// Bind just does a POST binding RPC.
func (b *binder) Bind(binding *v1.Binding) error {
	klog.V(3).Infof("Attempting to bind %v to %v", binding.Name, binding.Target.Name)
	return b.Client.CoreV1().Pods(binding.Namespace).Bind(binding)
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
