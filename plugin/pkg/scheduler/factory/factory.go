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
// plugin/cmd/scheduler for both testability and reuse.
package factory

import (
	"fmt"
	"reflect"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	appsinformers "k8s.io/client-go/informers/apps/v1beta1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	extensionsinformers "k8s.io/client-go/informers/extensions/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	appslisters "k8s.io/client-go/listers/apps/v1beta1"
	corelisters "k8s.io/client-go/listers/core/v1"
	extensionslisters "k8s.io/client-go/listers/extensions/v1beta1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/helper"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/api/validation"
	"k8s.io/kubernetes/plugin/pkg/scheduler/core"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
	"k8s.io/kubernetes/plugin/pkg/scheduler/util"

	"github.com/golang/glog"
)

const (
	initialGetBackoff = 100 * time.Millisecond
	maximalGetBackoff = time.Minute
)

var (
	serviceAffinitySet           = sets.NewString("ServiceAffinity")
	maxPDVolumeCountPredicateSet = sets.NewString("MaxPDVolumeCountPredicate")
	matchInterPodAffinitySet     = sets.NewString("MatchInterPodAffinity")
	generalPredicatesSets        = sets.NewString("GeneralPredicates")
	noDiskConflictSet            = sets.NewString("NoDiskConflict")
)

// ConfigFactory is the default implementation of the scheduler.Configurator interface.
// TODO make this private if possible, so that only its interface is externally used.
type ConfigFactory struct {
	client clientset.Interface
	// queue for pods that need scheduling
	podQueue *cache.FIFO
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
	replicaSetLister extensionslisters.ReplicaSetLister
	// a means to list all statefulsets
	statefulSetLister appslisters.StatefulSetLister

	// Close this to stop all reflectors
	StopEverything chan struct{}

	scheduledPodsHasSynced cache.InformerSynced

	schedulerCache schedulercache.Cache

	// SchedulerName of a scheduler is used to select which pods will be
	// processed by this scheduler, based on pods's "spec.SchedulerName".
	schedulerName string

	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 0-100.
	hardPodAffinitySymmetricWeight int

	// Equivalence class cache
	equivalencePodCache *core.EquivalenceCache

	// Enable equivalence class cache
	enableEquivalenceClassCache bool
}

// NewConfigFactory initializes the default implementation of a Configurator To encourage eventual privatization of the struct type, we only
// return the interface.
func NewConfigFactory(
	schedulerName string,
	client clientset.Interface,
	nodeInformer coreinformers.NodeInformer,
	podInformer coreinformers.PodInformer,
	pvInformer coreinformers.PersistentVolumeInformer,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	replicationControllerInformer coreinformers.ReplicationControllerInformer,
	replicaSetInformer extensionsinformers.ReplicaSetInformer,
	statefulSetInformer appsinformers.StatefulSetInformer,
	serviceInformer coreinformers.ServiceInformer,
	hardPodAffinitySymmetricWeight int,
	enableEquivalenceClassCache bool,
) scheduler.Configurator {
	stopEverything := make(chan struct{})
	schedulerCache := schedulercache.New(30*time.Second, stopEverything)

	c := &ConfigFactory{
		client:                         client,
		podLister:                      schedulerCache,
		podQueue:                       cache.NewFIFO(cache.MetaNamespaceKeyFunc),
		pVLister:                       pvInformer.Lister(),
		pVCLister:                      pvcInformer.Lister(),
		serviceLister:                  serviceInformer.Lister(),
		controllerLister:               replicationControllerInformer.Lister(),
		replicaSetLister:               replicaSetInformer.Lister(),
		statefulSetLister:              statefulSetInformer.Lister(),
		schedulerCache:                 schedulerCache,
		StopEverything:                 stopEverything,
		schedulerName:                  schedulerName,
		hardPodAffinitySymmetricWeight: hardPodAffinitySymmetricWeight,
		enableEquivalenceClassCache:    enableEquivalenceClassCache,
	}

	c.scheduledPodsHasSynced = podInformer.Informer().HasSynced
	// scheduled pod cache
	podInformer.Informer().AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Pod:
					return assignedNonTerminatedPod(t)
				default:
					runtime.HandleError(fmt.Errorf("unable to handle object in %T: %T", c, obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    c.addPodToCache,
				UpdateFunc: c.updatePodInCache,
				DeleteFunc: c.deletePodFromCache,
			},
		},
	)
	// unscheduled pod queue
	podInformer.Informer().AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Pod:
					return unassignedNonTerminatedPod(t)
				default:
					runtime.HandleError(fmt.Errorf("unable to handle object in %T: %T", c, obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc: func(obj interface{}) {
					if err := c.podQueue.Add(obj); err != nil {
						runtime.HandleError(fmt.Errorf("unable to queue %T: %v", obj, err))
					}
				},
				UpdateFunc: func(oldObj, newObj interface{}) {
					if err := c.podQueue.Update(newObj); err != nil {
						runtime.HandleError(fmt.Errorf("unable to update %T: %v", newObj, err))
					}
				},
				DeleteFunc: func(obj interface{}) {
					if err := c.podQueue.Delete(obj); err != nil {
						runtime.HandleError(fmt.Errorf("unable to dequeue %T: %v", obj, err))
					}
				},
			},
		},
	)
	// ScheduledPodLister is something we provide to plug-in functions that
	// they may need to call.
	c.scheduledPodLister = assignedPodLister{podInformer.Lister()}

	// Only nodes in the "Ready" condition with status == "True" are schedulable
	nodeInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.addNodeToCache,
			UpdateFunc: c.updateNodeInCache,
			DeleteFunc: c.deleteNodeFromCache,
		},
		0,
	)
	c.nodeLister = nodeInformer.Lister()

	// On add and delete of PVs, it will affect equivalence cache items
	// related to persistent volume
	pvInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			// MaxPDVolumeCountPredicate: since it relies on the counts of PV.
			AddFunc:    c.onPvAdd,
			DeleteFunc: c.onPvDelete,
		},
		0,
	)
	c.pVLister = pvInformer.Lister()

	// This is for MaxPDVolumeCountPredicate: add/delete PVC will affect counts of PV when it is bound.
	pvcInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.onPvcAdd,
			DeleteFunc: c.onPvcDelete,
		},
		0,
	)
	c.pVCLister = pvcInformer.Lister()

	// This is for ServiceAffinity: affected by the selector of the service is updated.
	// Also, if new service is added, equivalence cache will also become invalid since
	// existing pods may be "captured" by this service and change this predicate result.
	serviceInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.onServiceAdd,
			UpdateFunc: c.onServiceUpdate,
			DeleteFunc: c.onServiceDelete,
		},
		0,
	)
	c.serviceLister = serviceInformer.Lister()

	// Existing equivalence cache should not be affected by add/delete RC/Deployment etc,
	// it only make sense when pod is scheduled or deleted

	return c
}

func (c *ConfigFactory) onPvAdd(obj interface{}) {
	if c.enableEquivalenceClassCache {
		pv, ok := obj.(*v1.PersistentVolume)
		if !ok {
			glog.Errorf("cannot convert to *v1.PersistentVolume: %v", obj)
			return
		}
		c.invalidatePredicatesForPv(pv)
	}
}

func (c *ConfigFactory) onPvDelete(obj interface{}) {
	if c.enableEquivalenceClassCache {
		var pv *v1.PersistentVolume
		switch t := obj.(type) {
		case *v1.PersistentVolume:
			pv = t
		case cache.DeletedFinalStateUnknown:
			var ok bool
			pv, ok = t.Obj.(*v1.PersistentVolume)
			if !ok {
				glog.Errorf("cannot convert to *v1.PersistentVolume: %v", t.Obj)
				return
			}
		default:
			glog.Errorf("cannot convert to *v1.PersistentVolume: %v", t)
			return
		}
		c.invalidatePredicatesForPv(pv)
	}
}

func (c *ConfigFactory) invalidatePredicatesForPv(pv *v1.PersistentVolume) {
	invalidPredicates := sets.NewString("MaxPDVolumeCountPredicate")
	if pv.Spec.AWSElasticBlockStore != nil {
		invalidPredicates.Insert("MaxEBSVolumeCount")
	}
	if pv.Spec.GCEPersistentDisk != nil {
		invalidPredicates.Insert("MaxGCEPDVolumeCount")
	}
	if pv.Spec.AzureDisk != nil {
		invalidPredicates.Insert("MaxAzureDiskVolumeCount")
	}
	c.equivalencePodCache.InvalidateCachedPredicateItemOfAllNodes(invalidPredicates)
}

func (c *ConfigFactory) onPvcAdd(obj interface{}) {
	if c.enableEquivalenceClassCache {
		pvc, ok := obj.(*v1.PersistentVolumeClaim)
		if !ok {
			glog.Errorf("cannot convert to *v1.PersistentVolumeClaim: %v", obj)
			return
		}
		c.invalidatePredicatesForPvc(pvc)
	}
}

func (c *ConfigFactory) onPvcDelete(obj interface{}) {
	if c.enableEquivalenceClassCache {
		var pvc *v1.PersistentVolumeClaim
		switch t := obj.(type) {
		case *v1.PersistentVolumeClaim:
			pvc = t
		case cache.DeletedFinalStateUnknown:
			var ok bool
			pvc, ok = t.Obj.(*v1.PersistentVolumeClaim)
			if !ok {
				glog.Errorf("cannot convert to *v1.PersistentVolumeClaim: %v", t.Obj)
				return
			}
		default:
			glog.Errorf("cannot convert to *v1.PersistentVolumeClaim: %v", t)
			return
		}
		c.invalidatePredicatesForPvc(pvc)
	}
}

func (c *ConfigFactory) invalidatePredicatesForPvc(pvc *v1.PersistentVolumeClaim) {
	if pvc.Spec.VolumeName != "" {
		c.equivalencePodCache.InvalidateCachedPredicateItemOfAllNodes(maxPDVolumeCountPredicateSet)
	}
}

func (c *ConfigFactory) onServiceAdd(obj interface{}) {
	if c.enableEquivalenceClassCache {
		c.equivalencePodCache.InvalidateCachedPredicateItemOfAllNodes(serviceAffinitySet)
	}
}

func (c *ConfigFactory) onServiceUpdate(oldObj interface{}, newObj interface{}) {
	if c.enableEquivalenceClassCache {
		// TODO(resouer) We may need to invalidate this for specified group of pods only
		oldService := oldObj.(*v1.Service)
		newService := newObj.(*v1.Service)
		if !reflect.DeepEqual(oldService.Spec.Selector, newService.Spec.Selector) {
			c.equivalencePodCache.InvalidateCachedPredicateItemOfAllNodes(serviceAffinitySet)
		}
	}
}

func (c *ConfigFactory) onServiceDelete(obj interface{}) {
	if c.enableEquivalenceClassCache {
		c.equivalencePodCache.InvalidateCachedPredicateItemOfAllNodes(serviceAffinitySet)
	}
}

// GetNodeStore provides the cache to the nodes, mostly internal use, but may also be called by mock-tests.
func (c *ConfigFactory) GetNodeLister() corelisters.NodeLister {
	return c.nodeLister
}

func (c *ConfigFactory) GetHardPodAffinitySymmetricWeight() int {
	return c.hardPodAffinitySymmetricWeight
}

func (f *ConfigFactory) GetSchedulerName() string {
	return f.schedulerName
}

// GetClient provides a kubernetes client, mostly internal use, but may also be called by mock-tests.
func (f *ConfigFactory) GetClient() clientset.Interface {
	return f.client
}

// GetScheduledPodListerIndexer provides a pod lister, mostly internal use, but may also be called by mock-tests.
func (c *ConfigFactory) GetScheduledPodLister() corelisters.PodLister {
	return c.scheduledPodLister
}

func (c *ConfigFactory) addPodToCache(obj interface{}) {
	pod, ok := obj.(*v1.Pod)
	if !ok {
		glog.Errorf("cannot convert to *v1.Pod: %v", obj)
		return
	}

	if err := c.schedulerCache.AddPod(pod); err != nil {
		glog.Errorf("scheduler cache AddPod failed: %v", err)
	}
	// NOTE: Updating equivalence cache of addPodToCache has been
	// handled optimistically in InvalidateCachedPredicateItemForPodAdd.
}

func (c *ConfigFactory) updatePodInCache(oldObj, newObj interface{}) {
	oldPod, ok := oldObj.(*v1.Pod)
	if !ok {
		glog.Errorf("cannot convert oldObj to *v1.Pod: %v", oldObj)
		return
	}
	newPod, ok := newObj.(*v1.Pod)
	if !ok {
		glog.Errorf("cannot convert newObj to *v1.Pod: %v", newObj)
		return
	}

	if err := c.schedulerCache.UpdatePod(oldPod, newPod); err != nil {
		glog.Errorf("scheduler cache UpdatePod failed: %v", err)
	}

	c.invalidateCachedPredicatesOnUpdatePod(newPod, oldPod)
}

func (c *ConfigFactory) invalidateCachedPredicatesOnUpdatePod(newPod *v1.Pod, oldPod *v1.Pod) {
	if c.enableEquivalenceClassCache {
		// if the pod does not have binded node, updating equivalence cache is meaningless;
		// if pod's binded node has been changed, that case should be handled by pod add & delete.
		if len(newPod.Spec.NodeName) != 0 && newPod.Spec.NodeName == oldPod.Spec.NodeName {
			if !reflect.DeepEqual(oldPod.GetLabels(), newPod.GetLabels()) {
				// MatchInterPodAffinity need to be reconsidered for this node,
				// as well as all nodes in its same failure domain.
				c.equivalencePodCache.InvalidateCachedPredicateItemOfAllNodes(
					matchInterPodAffinitySet)
			}
			// if requested container resource changed, invalidate GeneralPredicates of this node
			if !reflect.DeepEqual(predicates.GetResourceRequest(newPod),
				predicates.GetResourceRequest(oldPod)) {
				c.equivalencePodCache.InvalidateCachedPredicateItem(
					newPod.Spec.NodeName, generalPredicatesSets)
			}
		}
	}
}

func (c *ConfigFactory) deletePodFromCache(obj interface{}) {
	var pod *v1.Pod
	switch t := obj.(type) {
	case *v1.Pod:
		pod = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		pod, ok = t.Obj.(*v1.Pod)
		if !ok {
			glog.Errorf("cannot convert to *v1.Pod: %v", t.Obj)
			return
		}
	default:
		glog.Errorf("cannot convert to *v1.Pod: %v", t)
		return
	}
	if err := c.schedulerCache.RemovePod(pod); err != nil {
		glog.Errorf("scheduler cache RemovePod failed: %v", err)
	}

	c.invalidateCachedPredicatesOnDeletePod(pod)
}

func (c *ConfigFactory) invalidateCachedPredicatesOnDeletePod(pod *v1.Pod) {
	if c.enableEquivalenceClassCache {
		// part of this case is the same as pod add.
		c.equivalencePodCache.InvalidateCachedPredicateItemForPodAdd(pod, pod.Spec.NodeName)
		// MatchInterPodAffinity need to be reconsidered for this node,
		// as well as all nodes in its same failure domain.
		// TODO(resouer) can we just do this for nodes in the same failure domain
		c.equivalencePodCache.InvalidateCachedPredicateItemOfAllNodes(
			matchInterPodAffinitySet)

		// if this pod have these PV, cached result of disk conflict will become invalid.
		for _, volume := range pod.Spec.Volumes {
			if volume.GCEPersistentDisk != nil || volume.AWSElasticBlockStore != nil ||
				volume.RBD != nil || volume.ISCSI != nil {
				c.equivalencePodCache.InvalidateCachedPredicateItem(
					pod.Spec.NodeName, noDiskConflictSet)
			}
		}
	}
}

func (c *ConfigFactory) addNodeToCache(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if !ok {
		glog.Errorf("cannot convert to *v1.Node: %v", obj)
		return
	}

	if err := c.schedulerCache.AddNode(node); err != nil {
		glog.Errorf("scheduler cache AddNode failed: %v", err)
	}

	// NOTE: add a new node does not affect existing predicates in equivalence cache
}

func (c *ConfigFactory) updateNodeInCache(oldObj, newObj interface{}) {
	oldNode, ok := oldObj.(*v1.Node)
	if !ok {
		glog.Errorf("cannot convert oldObj to *v1.Node: %v", oldObj)
		return
	}
	newNode, ok := newObj.(*v1.Node)
	if !ok {
		glog.Errorf("cannot convert newObj to *v1.Node: %v", newObj)
		return
	}

	if err := c.schedulerCache.UpdateNode(oldNode, newNode); err != nil {
		glog.Errorf("scheduler cache UpdateNode failed: %v", err)
	}

	c.invalidateCachedPredicatesOnNodeUpdate(newNode, oldNode)
}

func (c *ConfigFactory) invalidateCachedPredicatesOnNodeUpdate(newNode *v1.Node, oldNode *v1.Node) {
	if c.enableEquivalenceClassCache {
		// Begin to update equivalence cache based on node update
		// TODO(resouer): think about lazily initialize this set
		invalidPredicates := sets.NewString()

		oldTaints, oldErr := helper.GetTaintsFromNodeAnnotations(oldNode.GetAnnotations())
		if oldErr != nil {
			glog.Errorf("Failed to get taints from old node annotation for equivalence cache")
		}
		newTaints, newErr := helper.GetTaintsFromNodeAnnotations(newNode.GetAnnotations())
		if newErr != nil {
			glog.Errorf("Failed to get taints from new node annotation for equivalence cache")
		}

		if !reflect.DeepEqual(oldNode.Status.Allocatable, newNode.Status.Allocatable) {
			invalidPredicates.Insert("GeneralPredicates") // "PodFitsResources"
		}
		if !reflect.DeepEqual(oldNode.GetLabels(), newNode.GetLabels()) {
			invalidPredicates.Insert("GeneralPredicates", "ServiceAffinity") // "PodSelectorMatches"
			for k, v := range oldNode.GetLabels() {
				// any label can be topology key of pod, we have to invalidate in all cases
				if v != newNode.GetLabels()[k] {
					invalidPredicates.Insert("MatchInterPodAffinity")
				}
				// NoVolumeZoneConflict will only be affected by zone related label change
				if k == kubeletapis.LabelZoneFailureDomain || k == kubeletapis.LabelZoneRegion {
					if v != newNode.GetLabels()[k] {
						invalidPredicates.Insert("NoVolumeZoneConflict")
					}
				}
			}
		}
		if !reflect.DeepEqual(oldTaints, newTaints) {
			invalidPredicates.Insert("PodToleratesNodeTaints")
		}
		if !reflect.DeepEqual(oldNode.Status.Conditions, newNode.Status.Conditions) {
			oldConditions := make(map[v1.NodeConditionType]v1.ConditionStatus)
			newConditions := make(map[v1.NodeConditionType]v1.ConditionStatus)
			for _, cond := range oldNode.Status.Conditions {
				oldConditions[cond.Type] = cond.Status
			}
			for _, cond := range newNode.Status.Conditions {
				newConditions[cond.Type] = cond.Status
			}
			if oldConditions[v1.NodeMemoryPressure] != newConditions[v1.NodeMemoryPressure] {
				invalidPredicates.Insert("CheckNodeMemoryPressure")
			}
			if oldConditions[v1.NodeDiskPressure] != newConditions[v1.NodeDiskPressure] {
				invalidPredicates.Insert("CheckNodeDiskPressure")
			}
			if oldConditions[v1.NodeReady] != newConditions[v1.NodeReady] ||
				oldConditions[v1.NodeOutOfDisk] != newConditions[v1.NodeOutOfDisk] ||
				oldConditions[v1.NodeNetworkUnavailable] != newConditions[v1.NodeNetworkUnavailable] ||
				newNode.Spec.Unschedulable != oldNode.Spec.Unschedulable {
				invalidPredicates.Insert("CheckNodeCondition")
			}
		}
		c.equivalencePodCache.InvalidateCachedPredicateItem(newNode.GetName(), invalidPredicates)
	}
}

func (c *ConfigFactory) deleteNodeFromCache(obj interface{}) {
	var node *v1.Node
	switch t := obj.(type) {
	case *v1.Node:
		node = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		node, ok = t.Obj.(*v1.Node)
		if !ok {
			glog.Errorf("cannot convert to *v1.Node: %v", t.Obj)
			return
		}
	default:
		glog.Errorf("cannot convert to *v1.Node: %v", t)
		return
	}
	if err := c.schedulerCache.RemoveNode(node); err != nil {
		glog.Errorf("scheduler cache RemoveNode failed: %v", err)
	}
	if c.enableEquivalenceClassCache {
		c.equivalencePodCache.InvalidateAllCachedPredicateItemOfNode(node.GetName())
	}
}

// Create creates a scheduler with the default algorithm provider.
func (f *ConfigFactory) Create() (*scheduler.Config, error) {
	return f.CreateFromProvider(DefaultProvider)
}

// Creates a scheduler from the name of a registered algorithm provider.
func (f *ConfigFactory) CreateFromProvider(providerName string) (*scheduler.Config, error) {
	glog.V(2).Infof("Creating scheduler from algorithm provider '%v'", providerName)
	provider, err := GetAlgorithmProvider(providerName)
	if err != nil {
		return nil, err
	}

	return f.CreateFromKeys(provider.FitPredicateKeys, provider.PriorityFunctionKeys, []algorithm.SchedulerExtender{})
}

// Creates a scheduler from the configuration file
func (f *ConfigFactory) CreateFromConfig(policy schedulerapi.Policy) (*scheduler.Config, error) {
	glog.V(2).Infof("Creating scheduler from configuration: %v", policy)

	// validate the policy configuration
	if err := validation.ValidatePolicy(policy); err != nil {
		return nil, err
	}

	predicateKeys := sets.NewString()
	for _, predicate := range policy.Predicates {
		glog.V(2).Infof("Registering predicate: %s", predicate.Name)
		predicateKeys.Insert(RegisterCustomFitPredicate(predicate))
	}

	priorityKeys := sets.NewString()
	for _, priority := range policy.Priorities {
		glog.V(2).Infof("Registering priority: %s", priority.Name)
		priorityKeys.Insert(RegisterCustomPriorityFunction(priority))
	}

	extenders := make([]algorithm.SchedulerExtender, 0)
	if len(policy.ExtenderConfigs) != 0 {
		for ii := range policy.ExtenderConfigs {
			glog.V(2).Infof("Creating extender with config %+v", policy.ExtenderConfigs[ii])
			if extender, err := core.NewHTTPExtender(&policy.ExtenderConfigs[ii]); err != nil {
				return nil, err
			} else {
				extenders = append(extenders, extender)
			}
		}
	}
	// Providing HardPodAffinitySymmetricWeight in the policy config is the new and preferred way of providing the value.
	// Give it higher precedence than scheduler CLI configuration when it is provided.
	if policy.HardPodAffinitySymmetricWeight != 0 {
		f.hardPodAffinitySymmetricWeight = policy.HardPodAffinitySymmetricWeight
	}
	return f.CreateFromKeys(predicateKeys, priorityKeys, extenders)
}

// getBinder returns an extender that supports bind or a default binder.
func (f *ConfigFactory) getBinder(extenders []algorithm.SchedulerExtender) scheduler.Binder {
	for i := range extenders {
		if extenders[i].IsBinder() {
			return extenders[i]
		}
	}
	return &binder{f.client}
}

// Creates a scheduler from a set of registered fit predicate keys and priority keys.
func (f *ConfigFactory) CreateFromKeys(predicateKeys, priorityKeys sets.String, extenders []algorithm.SchedulerExtender) (*scheduler.Config, error) {
	glog.V(2).Infof("Creating scheduler with fit predicates '%v' and priority functions '%v'", predicateKeys, priorityKeys)

	if f.GetHardPodAffinitySymmetricWeight() < 1 || f.GetHardPodAffinitySymmetricWeight() > 100 {
		return nil, fmt.Errorf("invalid hardPodAffinitySymmetricWeight: %d, must be in the range 1-100", f.GetHardPodAffinitySymmetricWeight())
	}

	predicateFuncs, err := f.GetPredicates(predicateKeys)
	if err != nil {
		return nil, err
	}

	priorityConfigs, err := f.GetPriorityFunctionConfigs(priorityKeys)
	if err != nil {
		return nil, err
	}

	priorityMetaProducer, err := f.GetPriorityMetadataProducer()
	if err != nil {
		return nil, err
	}

	predicateMetaProducer, err := f.GetPredicateMetadataProducer()
	if err != nil {
		return nil, err
	}

	// Init equivalence class cache
	if f.enableEquivalenceClassCache && getEquivalencePodFunc != nil {
		f.equivalencePodCache = core.NewEquivalenceCache(getEquivalencePodFunc)
		glog.Info("Created equivalence class cache")
	}
	algo := core.NewGenericScheduler(f.schedulerCache, f.equivalencePodCache, predicateFuncs, predicateMetaProducer, priorityConfigs, priorityMetaProducer, extenders)

	podBackoff := util.CreateDefaultPodBackoff()
	return &scheduler.Config{
		SchedulerCache: f.schedulerCache,
		Ecache:         f.equivalencePodCache,
		// The scheduler only needs to consider schedulable nodes.
		NodeLister:          &nodeLister{f.nodeLister},
		Algorithm:           algo,
		Binder:              f.getBinder(extenders),
		PodConditionUpdater: &podConditionUpdater{f.client},
		PodPreemptor:        &podPreemptor{f.client},
		WaitForCacheSync: func() bool {
			return cache.WaitForCacheSync(f.StopEverything, f.scheduledPodsHasSynced)
		},
		NextPod: func() *v1.Pod {
			return f.getNextPod()
		},
		Error:          f.MakeDefaultErrorFunc(podBackoff, f.podQueue),
		StopEverything: f.StopEverything,
	}, nil
}

type nodeLister struct {
	corelisters.NodeLister
}

func (n *nodeLister) List() ([]*v1.Node, error) {
	return n.NodeLister.List(labels.Everything())
}

func (f *ConfigFactory) GetPriorityFunctionConfigs(priorityKeys sets.String) ([]algorithm.PriorityConfig, error) {
	pluginArgs, err := f.getPluginArgs()
	if err != nil {
		return nil, err
	}

	return getPriorityFunctionConfigs(priorityKeys, *pluginArgs)
}

func (f *ConfigFactory) GetPriorityMetadataProducer() (algorithm.MetadataProducer, error) {
	pluginArgs, err := f.getPluginArgs()
	if err != nil {
		return nil, err
	}

	return getPriorityMetadataProducer(*pluginArgs)
}

func (f *ConfigFactory) GetPredicateMetadataProducer() (algorithm.PredicateMetadataProducer, error) {
	pluginArgs, err := f.getPluginArgs()
	if err != nil {
		return nil, err
	}
	return getPredicateMetadataProducer(*pluginArgs)
}

func (f *ConfigFactory) GetPredicates(predicateKeys sets.String) (map[string]algorithm.FitPredicate, error) {
	pluginArgs, err := f.getPluginArgs()
	if err != nil {
		return nil, err
	}

	return getFitPredicateFunctions(predicateKeys, *pluginArgs)
}

func (f *ConfigFactory) getPluginArgs() (*PluginFactoryArgs, error) {
	return &PluginFactoryArgs{
		PodLister:         f.podLister,
		ServiceLister:     f.serviceLister,
		ControllerLister:  f.controllerLister,
		ReplicaSetLister:  f.replicaSetLister,
		StatefulSetLister: f.statefulSetLister,
		NodeLister:        &nodeLister{f.nodeLister},
		NodeInfo:          &predicates.CachedNodeInfo{NodeLister: f.nodeLister},
		PVInfo:            &predicates.CachedPersistentVolumeInfo{PersistentVolumeLister: f.pVLister},
		PVCInfo:           &predicates.CachedPersistentVolumeClaimInfo{PersistentVolumeClaimLister: f.pVCLister},
		HardPodAffinitySymmetricWeight: f.hardPodAffinitySymmetricWeight,
	}, nil
}

func (f *ConfigFactory) getNextPod() *v1.Pod {
	for {
		pod := cache.Pop(f.podQueue).(*v1.Pod)
		if f.ResponsibleForPod(pod) {
			glog.V(4).Infof("About to try and schedule pod %v", pod.Name)
			return pod
		}
	}
}

func (f *ConfigFactory) ResponsibleForPod(pod *v1.Pod) bool {
	return f.schedulerName == pod.Spec.SchedulerName
}

// unassignedNonTerminatedPod selects pods that are unassigned and non-terminal.
func unassignedNonTerminatedPod(pod *v1.Pod) bool {
	if len(pod.Spec.NodeName) != 0 {
		return false
	}
	if pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
		return false
	}
	return true
}

// assignedNonTerminatedPod selects pods that are assigned and non-terminal (scheduled and running).
func assignedNonTerminatedPod(pod *v1.Pod) bool {
	if len(pod.Spec.NodeName) == 0 {
		return false
	}
	if pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
		return false
	}
	return true
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
	selector := fields.ParseSelectorOrDie("status.phase!=" + string(v1.PodSucceeded) + ",status.phase!=" + string(v1.PodFailed))
	lw := cache.NewListWatchFromClient(client.CoreV1().RESTClient(), string(v1.ResourcePods), metav1.NamespaceAll, selector)
	return &podInformer{
		informer: cache.NewSharedIndexInformer(lw, &v1.Pod{}, resyncPeriod, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}),
	}
}

func (factory *ConfigFactory) MakeDefaultErrorFunc(backoff *util.PodBackoff, podQueue *cache.FIFO) func(pod *v1.Pod, err error) {
	return func(pod *v1.Pod, err error) {
		if err == core.ErrNoNodesAvailable {
			glog.V(4).Infof("Unable to schedule %v %v: no nodes are registered to the cluster; waiting", pod.Namespace, pod.Name)
		} else {
			if _, ok := err.(*core.FitError); ok {
				glog.V(4).Infof("Unable to schedule %v %v: no fit: %v; waiting", pod.Namespace, pod.Name, err)
			} else {
				glog.Errorf("Error scheduling %v %v: %v; retrying", pod.Namespace, pod.Name, err)
			}
		}
		backoff.Gc()
		// Retry asynchronously.
		// Note that this is extremely rudimentary and we need a more real error handling path.
		go func() {
			defer runtime.HandleCrash()
			podID := types.NamespacedName{
				Namespace: pod.Namespace,
				Name:      pod.Name,
			}

			entry := backoff.GetEntry(podID)
			if !entry.TryWait(backoff.MaxDuration()) {
				glog.Warningf("Request for pod %v already in flight, abandoning", podID)
				return
			}
			// Get the pod again; it may have changed/been scheduled already.
			getBackoff := initialGetBackoff
			for {
				pod, err := factory.client.CoreV1().Pods(podID.Namespace).Get(podID.Name, metav1.GetOptions{})
				if err == nil {
					if len(pod.Spec.NodeName) == 0 {
						podQueue.AddIfNotPresent(pod)
					}
					break
				}
				if errors.IsNotFound(err) {
					glog.Warningf("A pod %v no longer exists", podID)
					return
				}
				glog.Errorf("Error getting pod %v for retry: %v; retrying...", podID, err)
				if getBackoff = getBackoff * 2; getBackoff > maximalGetBackoff {
					getBackoff = maximalGetBackoff
				}
				time.Sleep(getBackoff)
			}
		}()
	}
}

// nodeEnumerator allows a cache.Poller to enumerate items in an v1.NodeList
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
	glog.V(3).Infof("Attempting to bind %v to %v", binding.Name, binding.Target.Name)
	return b.Client.CoreV1().Pods(binding.Namespace).Bind(binding)
}

type podConditionUpdater struct {
	Client clientset.Interface
}

func (p *podConditionUpdater) Update(pod *v1.Pod, condition *v1.PodCondition) error {
	glog.V(2).Infof("Updating pod condition for %s/%s to (%s==%s)", pod.Namespace, pod.Name, condition.Type, condition.Status)
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

//TODO(bsalamat): change this to patch PodStatus to avoid overwriting potential pending status updates.
func (p *podPreemptor) UpdatePodAnnotations(pod *v1.Pod, annotations map[string]string) error {
	podCopy := pod.DeepCopy()
	if podCopy.Annotations == nil {
		podCopy.Annotations = map[string]string{}
	}
	for k, v := range annotations {
		podCopy.Annotations[k] = v
	}
	_, err := p.Client.CoreV1().Pods(podCopy.Namespace).UpdateStatus(podCopy)
	return err
}
