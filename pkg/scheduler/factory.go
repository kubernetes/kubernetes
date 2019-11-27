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
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	policylisters "k8s.io/client-go/listers/policy/v1beta1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/core"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	cachedebugger "k8s.io/kubernetes/pkg/scheduler/internal/cache/debugger"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	nodeinfosnapshot "k8s.io/kubernetes/pkg/scheduler/nodeinfo/snapshot"
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

// Configurator defines I/O, caching, and other functionality needed to
// construct a new scheduler.
type Configurator struct {
	client clientset.Interface

	informerFactory informers.SharedInformerFactory

	podInformer coreinformers.PodInformer

	// Close this to stop all reflectors
	StopEverything <-chan struct{}

	schedulerCache internalcache.Cache

	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range [0-100].
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

	podInitialBackoffSeconds int64

	podMaxBackoffSeconds int64

	enableNonPreempting bool

	// framework configuration arguments.
	registry                     framework.Registry
	plugins                      *schedulerapi.Plugins
	pluginConfig                 []schedulerapi.PluginConfig
	pluginConfigProducerRegistry *plugins.ConfigProducerRegistry
	nodeInfoSnapshot             *nodeinfosnapshot.Snapshot

	algorithmFactoryArgs AlgorithmFactoryArgs
	configProducerArgs   *plugins.ConfigProducerArgs
}

// GetHardPodAffinitySymmetricWeight is exposed for testing.
func (c *Configurator) GetHardPodAffinitySymmetricWeight() int32 {
	return c.hardPodAffinitySymmetricWeight
}

// Create creates a scheduler with the default algorithm provider.
func (c *Configurator) Create() (*Scheduler, error) {
	return c.CreateFromProvider(DefaultProvider)
}

// CreateFromProvider creates a scheduler from the name of a registered algorithm provider.
func (c *Configurator) CreateFromProvider(providerName string) (*Scheduler, error) {
	klog.V(2).Infof("Creating scheduler from algorithm provider '%v'", providerName)
	provider, err := GetAlgorithmProvider(providerName)
	if err != nil {
		return nil, err
	}
	return c.CreateFromKeys(provider.FitPredicateKeys, provider.PriorityFunctionKeys, []algorithm.SchedulerExtender{})
}

// CreateFromConfig creates a scheduler from the configuration file
func (c *Configurator) CreateFromConfig(policy schedulerapi.Policy) (*Scheduler, error) {
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
			predicateKeys.Insert(RegisterCustomFitPredicate(predicate, c.configProducerArgs))
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
			if priority.Name == priorities.EqualPriority {
				klog.V(2).Infof("Skip registering priority: %s", priority.Name)
				continue
			}
			klog.V(2).Infof("Registering priority: %s", priority.Name)
			priorityKeys.Insert(RegisterCustomPriorityFunction(priority, c.configProducerArgs))
		}
	}

	var extenders []algorithm.SchedulerExtender
	if len(policy.Extenders) != 0 {
		ignoredExtendedResources := sets.NewString()
		var ignorableExtenders []algorithm.SchedulerExtender
		for ii := range policy.Extenders {
			klog.V(2).Infof("Creating extender with config %+v", policy.Extenders[ii])
			extender, err := core.NewHTTPExtender(&policy.Extenders[ii])
			if err != nil {
				return nil, err
			}
			if !extender.IsIgnorable() {
				extenders = append(extenders, extender)
			} else {
				ignorableExtenders = append(ignorableExtenders, extender)
			}
			for _, r := range policy.Extenders[ii].ManagedResources {
				if r.IgnoredByScheduler {
					ignoredExtendedResources.Insert(string(r.Name))
				}
			}
		}
		// place ignorable extenders to the tail of extenders
		extenders = append(extenders, ignorableExtenders...)
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

// CreateFromKeys creates a scheduler from a set of registered fit predicate keys and priority keys.
func (c *Configurator) CreateFromKeys(predicateKeys, priorityKeys sets.String, extenders []algorithm.SchedulerExtender) (*Scheduler, error) {
	klog.V(2).Infof("Creating scheduler with fit predicates '%v' and priority functions '%v'", predicateKeys, priorityKeys)

	if c.GetHardPodAffinitySymmetricWeight() < 1 || c.GetHardPodAffinitySymmetricWeight() > 100 {
		return nil, fmt.Errorf("invalid hardPodAffinitySymmetricWeight: %d, must be in the range 1-100", c.GetHardPodAffinitySymmetricWeight())
	}

	predicateFuncs, pluginsForPredicates, pluginConfigForPredicates, err := c.getPredicateConfigs(predicateKeys)
	if err != nil {
		return nil, err
	}

	priorityConfigs, pluginsForPriorities, pluginConfigForPriorities, err := c.getPriorityConfigs(priorityKeys)
	if err != nil {
		return nil, err
	}

	priorityMetaProducer, err := getPriorityMetadataProducer(c.algorithmFactoryArgs)
	if err != nil {
		return nil, err
	}

	predicateMetaProducer, err := getPredicateMetadataProducer(c.algorithmFactoryArgs)
	if err != nil {
		return nil, err
	}

	// Combine all framework configurations. If this results in any duplication, framework
	// instantiation should fail.
	var plugins schedulerapi.Plugins
	plugins.Append(pluginsForPredicates)
	plugins.Append(pluginsForPriorities)
	plugins.Append(c.plugins)
	var pluginConfig []schedulerapi.PluginConfig
	pluginConfig = append(pluginConfig, pluginConfigForPredicates...)
	pluginConfig = append(pluginConfig, pluginConfigForPriorities...)
	pluginConfig = append(pluginConfig, c.pluginConfig...)

	framework, err := framework.NewFramework(
		c.registry,
		&plugins,
		pluginConfig,
		framework.WithClientSet(c.client),
		framework.WithInformerFactory(c.informerFactory),
		framework.WithSnapshotSharedLister(c.nodeInfoSnapshot),
	)
	if err != nil {
		klog.Fatalf("error initializing the scheduling framework: %v", err)
	}

	podQueue := internalqueue.NewSchedulingQueue(
		c.StopEverything,
		framework,
		internalqueue.WithPodInitialBackoffDuration(time.Duration(c.podInitialBackoffSeconds)*time.Second),
		internalqueue.WithPodMaxBackoffDuration(time.Duration(c.podMaxBackoffSeconds)*time.Second),
	)

	// Setup cache debugger.
	debugger := cachedebugger.New(
		c.informerFactory.Core().V1().Nodes().Lister(),
		c.podInformer.Lister(),
		c.schedulerCache,
		podQueue,
	)
	debugger.ListenForSignal(c.StopEverything)

	go func() {
		<-c.StopEverything
		podQueue.Close()
	}()

	algo := core.NewGenericScheduler(
		c.schedulerCache,
		podQueue,
		predicateFuncs,
		predicateMetaProducer,
		priorityConfigs,
		priorityMetaProducer,
		c.nodeInfoSnapshot,
		framework,
		extenders,
		c.volumeBinder,
		c.informerFactory.Core().V1().PersistentVolumeClaims().Lister(),
		GetPodDisruptionBudgetLister(c.informerFactory),
		c.alwaysCheckAllPredicates,
		c.disablePreemption,
		c.percentageOfNodesToScore,
		c.enableNonPreempting,
	)

	return &Scheduler{
		SchedulerCache:  c.schedulerCache,
		Algorithm:       algo,
		GetBinder:       getBinderFunc(c.client, extenders),
		Framework:       framework,
		NextPod:         internalqueue.MakeNextPodFunc(podQueue),
		Error:           MakeDefaultErrorFunc(c.client, podQueue, c.schedulerCache),
		StopEverything:  c.StopEverything,
		VolumeBinder:    c.volumeBinder,
		SchedulingQueue: podQueue,
		Plugins:         plugins,
		PluginConfig:    pluginConfig,
	}, nil
}

// getBinderFunc returns a func which returns an extender that supports bind or a default binder based on the given pod.
func getBinderFunc(client clientset.Interface, extenders []algorithm.SchedulerExtender) func(pod *v1.Pod) Binder {
	defaultBinder := &binder{client}
	return func(pod *v1.Pod) Binder {
		for _, extender := range extenders {
			if extender.IsBinder() && extender.IsInterested(pod) {
				return extender
			}
		}
		return defaultBinder
	}
}

// getPriorityConfigs returns priorities configuration: ones that will run as priorities and ones that will run
// as framework plugins. Specifically, a priority will run as a framework plugin if a plugin config producer was
// registered for that priority.
func (c *Configurator) getPriorityConfigs(priorityKeys sets.String) ([]priorities.PriorityConfig, *schedulerapi.Plugins, []schedulerapi.PluginConfig, error) {
	allPriorityConfigs, err := getPriorityFunctionConfigs(priorityKeys, c.algorithmFactoryArgs)
	if err != nil {
		return nil, nil, nil, err
	}

	if c.pluginConfigProducerRegistry == nil {
		return allPriorityConfigs, nil, nil, nil
	}

	var priorityConfigs []priorities.PriorityConfig
	var plugins schedulerapi.Plugins
	var pluginConfig []schedulerapi.PluginConfig
	frameworkConfigProducers := c.pluginConfigProducerRegistry.PriorityToConfigProducer
	for _, p := range allPriorityConfigs {
		if producer, exist := frameworkConfigProducers[p.Name]; exist {
			args := *c.configProducerArgs
			args.Weight = int32(p.Weight)
			pl, pc := producer(args)
			plugins.Append(&pl)
			pluginConfig = append(pluginConfig, pc...)
		} else {
			priorityConfigs = append(priorityConfigs, p)
		}
	}
	return priorityConfigs, &plugins, pluginConfig, nil
}

// getPredicateConfigs returns predicates configuration: ones that will run as fitPredicates and ones that will run
// as framework plugins. Specifically, a predicate will run as a framework plugin if a plugin config producer was
// registered for that predicate.
// Note that the framework executes plugins according to their order in the Plugins list, and so predicates run as plugins
// are added to the Plugins list according to the order specified in predicates.Ordering().
func (c *Configurator) getPredicateConfigs(predicateKeys sets.String) (map[string]predicates.FitPredicate, *schedulerapi.Plugins, []schedulerapi.PluginConfig, error) {
	allFitPredicates, err := getFitPredicateFunctions(predicateKeys, c.algorithmFactoryArgs)
	if err != nil {
		return nil, nil, nil, err
	}

	if c.pluginConfigProducerRegistry == nil {
		return allFitPredicates, nil, nil, nil
	}

	asPlugins := sets.NewString()
	asFitPredicates := make(map[string]predicates.FitPredicate)
	frameworkConfigProducers := c.pluginConfigProducerRegistry.PredicateToConfigProducer

	// First, identify the predicates that will run as actual fit predicates, and ones
	// that will run as framework plugins.
	for predicateKey := range allFitPredicates {
		if _, exist := frameworkConfigProducers[predicateKey]; exist {
			asPlugins.Insert(predicateKey)
		} else {
			asFitPredicates[predicateKey] = allFitPredicates[predicateKey]
		}
	}

	// Second, create the framework plugin configurations, and place them in the order
	// that the corresponding predicates were supposed to run.
	var plugins schedulerapi.Plugins
	var pluginConfig []schedulerapi.PluginConfig

	for _, predicateKey := range predicates.Ordering() {
		if asPlugins.Has(predicateKey) {
			producer := frameworkConfigProducers[predicateKey]
			p, pc := producer(*c.configProducerArgs)
			plugins.Append(&p)
			pluginConfig = append(pluginConfig, pc...)
			asPlugins.Delete(predicateKey)
		}
	}

	// Third, add the rest in no specific order.
	for predicateKey := range asPlugins {
		producer := frameworkConfigProducers[predicateKey]
		p, pc := producer(*c.configProducerArgs)
		plugins.Append(&p)
		pluginConfig = append(pluginConfig, pc...)
	}

	return asFitPredicates, &plugins, pluginConfig, nil
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
func MakeDefaultErrorFunc(client clientset.Interface, podQueue internalqueue.SchedulingQueue, schedulerCache internalcache.Cache) func(*framework.PodInfo, error) {
	return func(podInfo *framework.PodInfo, err error) {
		pod := podInfo.Pod
		if err == core.ErrNoNodesAvailable {
			klog.V(2).Infof("Unable to schedule %v/%v: no nodes are registered to the cluster; waiting", pod.Namespace, pod.Name)
		} else {
			if _, ok := err.(*core.FitError); ok {
				klog.V(2).Infof("Unable to schedule %v/%v: no fit: %v; waiting", pod.Namespace, pod.Name, err)
			} else if errors.IsNotFound(err) {
				klog.V(2).Infof("Unable to schedule %v/%v: possibly due to node not found: %v; waiting", pod.Namespace, pod.Name, err)
				if errStatus, ok := err.(errors.APIStatus); ok && errStatus.Status().Details.Kind == "node" {
					nodeName := errStatus.Status().Details.Name
					// when node is not found, We do not remove the node right away. Trying again to get
					// the node and if the node is still not found, then remove it from the scheduler cache.
					_, err := client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
					if err != nil && errors.IsNotFound(err) {
						node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}
						if err := schedulerCache.RemoveNode(&node); err != nil {
							klog.V(4).Infof("Node %q is not found; failed to remove it from the cache.", node.Name)
						}
					}
				}
			} else {
				klog.Errorf("Error scheduling %v/%v: %v; retrying", pod.Namespace, pod.Name, err)
			}
		}

		podSchedulingCycle := podQueue.SchedulingCycle()
		// Retry asynchronously.
		// Note that this is extremely rudimentary and we need a more real error handling path.
		go func() {
			defer runtime.HandleCrash()
			podID := types.NamespacedName{
				Namespace: pod.Namespace,
				Name:      pod.Name,
			}

			// An unschedulable pod will be placed in the unschedulable queue.
			// This ensures that if the pod is nominated to run on a node,
			// scheduler takes the pod into account when running predicates for the node.
			// Get the pod again; it may have changed/been scheduled already.
			getBackoff := initialGetBackoff
			for {
				pod, err := client.CoreV1().Pods(podID.Namespace).Get(podID.Name, metav1.GetOptions{})
				if err == nil {
					if len(pod.Spec.NodeName) == 0 {
						podInfo.Pod = pod
						if err := podQueue.AddUnschedulableIfNotPresent(podInfo, podSchedulingCycle); err != nil {
							klog.Error(err)
						}
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

type binder struct {
	Client clientset.Interface
}

// Bind just does a POST binding RPC.
func (b *binder) Bind(binding *v1.Binding) error {
	klog.V(3).Infof("Attempting to bind %v to %v", binding.Name, binding.Target.Name)
	return b.Client.CoreV1().Pods(binding.Namespace).Bind(binding)
}

// GetPodDisruptionBudgetLister returns pdb lister from the given informer factory. Returns nil if PodDisruptionBudget feature is disabled.
func GetPodDisruptionBudgetLister(informerFactory informers.SharedInformerFactory) policylisters.PodDisruptionBudgetLister {
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.PodDisruptionBudget) {
		return informerFactory.Policy().V1beta1().PodDisruptionBudgets().Lister()
	}
	return nil
}

// GetCSINodeLister returns CSINode lister from the given informer factory. Returns nil if CSINodeInfo feature is disabled.
func GetCSINodeLister(informerFactory informers.SharedInformerFactory) storagelisters.CSINodeLister {
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.CSINodeInfo) {
		return informerFactory.Storage().V1().CSINodes().Lister()
	}
	return nil
}
