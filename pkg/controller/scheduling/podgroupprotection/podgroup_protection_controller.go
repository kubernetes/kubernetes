/*
Copyright The Kubernetes Authors.

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

package podgroupprotection

import (
	"context"
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	schedulinginformersv1alpha3 "k8s.io/client-go/informers/scheduling/v1alpha3"
	schedulinginformersv1beta1 "k8s.io/client-go/informers/scheduling/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	schedulinglistersv1alpha3 "k8s.io/client-go/listers/scheduling/v1alpha3"
	schedulinglistersv1beta1 "k8s.io/client-go/listers/scheduling/v1beta1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/controller/util/protectionutil"
	"k8s.io/kubernetes/pkg/util/slice"
)

const (
	// The index name for looking up active pods by their
	// schedulingGroup.podGroupName field.
	activePodSchedulingGroupIndex = "activePodSchedulingGroup"

	// The index name for looking up child PodGroups by their
	// parentCompositePodGroupName field.
	childPodGroupParentCompositeGroupIndex = "childPodGroupParentCompositeGroup"

	// The index name for looking up child CompositePodGroups by their
	// parentCompositePodGroupName field.
	childCompositePodGroupParentCompositeGroupIndex = "childCompositePodGroupParentCompositeGroup"
)

// Controller manages the PodGroupProtectionFinalizer on PodGroup objects and
// CompositePodGroupProtectionFinalizer on CompositePodGroup objects.
// Finalizers are stamped at creation time by the PodGroupProtection admission
// plugin; this controller removes them when objects are being deleted and no
// child resources still reference them.
type Controller struct {
	kubeClient clientset.Interface

	podGroupLister  schedulinglistersv1beta1.PodGroupLister
	podGroupSynced  cache.InformerSynced
	podGroupIndexer cache.Indexer

	isCompositePodGroupEnabled bool
	compositePodGroupLister    schedulinglistersv1alpha3.CompositePodGroupLister
	compositePodGroupSynced    cache.InformerSynced
	compositePodGroupIndexer   cache.Indexer

	podSynced cache.InformerSynced

	// podIndexer has the common Pod indexer installed to
	// limit iteration over pods to those of interest.
	podIndexer cache.Indexer

	queue workqueue.TypedRateLimitingInterface[fwk.EntityKey]
}

// NewPodGroupProtectionController returns a new instance of the PodGroup protection controller.
func NewPodGroupProtectionController(
	logger klog.Logger,
	podGroupInformer schedulinginformersv1beta1.PodGroupInformer,
	compositePodGroupInformer schedulinginformersv1alpha3.CompositePodGroupInformer,
	podInformer coreinformers.PodInformer,
	kubeClient clientset.Interface,
	isCompositePodGroupEnabled bool,
) (*Controller, error) {
	c := &Controller{
		kubeClient:                 kubeClient,
		isCompositePodGroupEnabled: isCompositePodGroupEnabled,
		podGroupLister:             podGroupInformer.Lister(),
		podGroupSynced:             podGroupInformer.Informer().HasSynced,
		podGroupIndexer:            podGroupInformer.Informer().GetIndexer(),
		podIndexer:                 podInformer.Informer().GetIndexer(),
		podSynced:                  podInformer.Informer().HasSynced,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[fwk.EntityKey](),
			workqueue.TypedRateLimitingQueueConfig[fwk.EntityKey]{Name: "podgroupprotection"},
		),
	}

	if c.isCompositePodGroupEnabled {
		c.compositePodGroupLister = compositePodGroupInformer.Lister()
		c.compositePodGroupSynced = compositePodGroupInformer.Informer().HasSynced
		c.compositePodGroupIndexer = compositePodGroupInformer.Informer().GetIndexer()

		if err := addChildCompositePodGroupParentCompositeGroupIndexer(c.compositePodGroupIndexer); err != nil {
			return nil, fmt.Errorf("could not initialize CompositePodGroup parent indexer: %w", err)
		}
		if err := addChildPodGroupParentCompositeGroupIndexer(c.podGroupIndexer); err != nil {
			return nil, fmt.Errorf("could not initialize PodGroup parent indexer: %w", err)
		}

		if _, err := compositePodGroupInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj any) {
				c.handleCompositePodGroupUpdate(logger, nil, obj)
			},
			UpdateFunc: func(old, new any) {
				c.handleCompositePodGroupUpdate(logger, old, new)
			},
			DeleteFunc: func(obj any) {
				c.handleCompositePodGroupUpdate(logger, obj, nil)
			},
		}, cache.HandlerOptions{Logger: &logger}); err != nil {
			return nil, err
		}
	}

	if _, err := podGroupInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			c.handlePodGroupUpdate(logger, nil, obj)
		},
		UpdateFunc: func(old, new any) {
			c.handlePodGroupUpdate(logger, old, new)
		},
		DeleteFunc: func(obj any) {
			c.handlePodGroupUpdate(logger, obj, nil)
		},
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, err
	}

	if err := addActivePodSchedulingGroupIndexer(c.podIndexer); err != nil {
		return nil, fmt.Errorf("could not initialize PodGroup protection controller: %w", err)
	}

	if _, err := podInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			c.handlePodChange(logger, nil, obj)
		},
		DeleteFunc: func(obj any) {
			c.handlePodChange(logger, obj, nil)
		},
		UpdateFunc: func(old, new any) {
			c.handlePodChange(logger, old, new)
		},
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, err
	}

	return c, nil
}

// addActivePodSchedulingGroupIndexer adds an indexer to look up active
// pods by their schedulingGroup.podGroupName field so we can efficiently
// determine whether a PodGroup still has active pods.
func addActivePodSchedulingGroupIndexer(indexer cache.Indexer) error {
	return indexer.AddIndexers(cache.Indexers{
		activePodSchedulingGroupIndex: func(obj any) ([]string, error) {
			pod, ok := obj.(*v1.Pod)
			if !ok {
				return nil, nil
			}
			if isPodTerminated(pod) {
				return nil, nil
			}
			if pod.Spec.SchedulingGroup == nil || pod.Spec.SchedulingGroup.PodGroupName == nil {
				return nil, nil
			}
			return []string{pod.Namespace + "/" + *pod.Spec.SchedulingGroup.PodGroupName}, nil
		},
	})
}

// addChildPodGroupParentCompositeGroupIndexer adds an indexer to look up child PodGroups by their
// parentCompositePodGroupName field.
func addChildPodGroupParentCompositeGroupIndexer(indexer cache.Indexer) error {
	return indexer.AddIndexers(cache.Indexers{
		childPodGroupParentCompositeGroupIndex: func(obj any) ([]string, error) {
			pg, ok := obj.(*schedulingv1beta1.PodGroup)
			if !ok {
				return nil, nil
			}
			if pg.Spec.ParentCompositePodGroupName == nil {
				return nil, nil
			}
			return []string{pg.Namespace + "/" + *pg.Spec.ParentCompositePodGroupName}, nil
		},
	})
}

// addChildCompositePodGroupParentCompositeGroupIndexer adds an indexer to look up child CompositePodGroups by their
// parentCompositePodGroupName field.
func addChildCompositePodGroupParentCompositeGroupIndexer(indexer cache.Indexer) error {
	return indexer.AddIndexers(cache.Indexers{
		childCompositePodGroupParentCompositeGroupIndex: func(obj any) ([]string, error) {
			cpg, ok := obj.(*schedulingv1alpha3.CompositePodGroup)
			if !ok {
				return nil, nil
			}
			if cpg.Spec.ParentCompositePodGroupName == nil {
				return nil, nil
			}
			return []string{cpg.Namespace + "/" + *cpg.Spec.ParentCompositePodGroupName}, nil
		},
	})
}

// Run runs the controller goroutines.
func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()

	logger := klog.FromContext(ctx)
	logger.Info("Starting PodGroup protection controller")

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down PodGroup protection controller")
		c.queue.ShutDown()
		wg.Wait()
	}()

	synced := []cache.InformerSynced{c.podGroupSynced, c.podSynced}
	if c.compositePodGroupSynced != nil {
		synced = append(synced, c.compositePodGroupSynced)
	}

	if !cache.WaitForNamedCacheSyncWithContext(ctx, synced...) {
		return
	}

	for range workers {
		wg.Go(func() {
			wait.UntilWithContext(ctx, c.runWorker, time.Second)
		})
	}
	<-ctx.Done()
}

func (c *Controller) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	itemKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(itemKey)

	var err error
	switch itemKey.Type {
	case fwk.CompositePodGroupKeyType:
		err = c.processCompositePodGroup(ctx, itemKey)
	case fwk.PodGroupKeyType:
		err = c.processPodGroup(ctx, itemKey)
	default:
		// Fallback for any legacy queued keys (should not happen)
		err = c.processPodGroup(ctx, itemKey)
	}

	if err == nil {
		c.queue.Forget(itemKey)
		return true
	}

	c.queue.AddRateLimited(itemKey)
	utilruntime.HandleError(fmt.Errorf("work item %v failed with: %w", itemKey, err))

	return true
}

func (c *Controller) processPodGroup(ctx context.Context, pgKey fwk.EntityKey) error {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Processing PodGroup", "podGroup", pgKey)

	pg, err := c.podGroupLister.PodGroups(pgKey.Namespace).Get(pgKey.Name)
	if apierrors.IsNotFound(err) {
		logger.V(4).Info("PodGroup not found, ignoring", "podGroup", pgKey)
		return nil
	}
	if err != nil {
		return err
	}

	if !protectionutil.IsDeletionCandidate(pg, scheduling.PodGroupProtectionFinalizer) {
		return nil
	}

	isUsed, err := c.hasActivePods(ctx, pg)
	if err != nil {
		return err
	}
	if !isUsed {
		return c.removeFinalizer(ctx, pg)
	}
	logger.V(4).Info("Keeping PodGroup finalizer because it is still being used by pods", "podGroup", klog.KObj(pg))
	return nil
}

func (c *Controller) processCompositePodGroup(ctx context.Context, cpgKey fwk.EntityKey) error {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Processing CompositePodGroup", "compositePodGroup", cpgKey)

	if c.compositePodGroupLister == nil {
		return nil
	}

	cpg, err := c.compositePodGroupLister.CompositePodGroups(cpgKey.Namespace).Get(cpgKey.Name)
	if apierrors.IsNotFound(err) {
		logger.V(4).Info("CompositePodGroup not found, ignoring", "compositePodGroup", cpgKey)
		return nil
	}
	if err != nil {
		return err
	}

	if !protectionutil.IsDeletionCandidate(cpg, scheduling.CompositePodGroupProtectionFinalizer) {
		return nil
	}

	hasChildren, err := c.hasChildGroups(ctx, cpg)
	if err != nil {
		return err
	}
	if !hasChildren {
		return c.removeCompositePodGroupFinalizer(ctx, cpg)
	}
	logger.V(4).Info("Keeping CompositePodGroup finalizer because it still has child pod groups or composite pod groups", "compositePodGroup", klog.KObj(cpg))
	return nil
}

func (c *Controller) removeFinalizer(ctx context.Context, pg *schedulingv1beta1.PodGroup) error {
	logger := klog.FromContext(ctx)
	pgClone := pg.DeepCopy()

	pgClone.Finalizers = slice.RemoveString(pgClone.Finalizers, scheduling.PodGroupProtectionFinalizer, nil)
	_, err := c.kubeClient.SchedulingV1beta1().PodGroups(pgClone.Namespace).Update(ctx, pgClone, metav1.UpdateOptions{})
	if err != nil {
		logger.Error(err, "Error removing protection finalizer from PodGroup", "podGroup", klog.KObj(pg))
		return err
	}

	logger.V(3).Info("Removed protection finalizer from PodGroup", "podGroup", klog.KObj(pg))
	return nil
}

func (c *Controller) removeCompositePodGroupFinalizer(ctx context.Context, cpg *schedulingv1alpha3.CompositePodGroup) error {
	logger := klog.FromContext(ctx)
	cpgClone := cpg.DeepCopy()

	cpgClone.Finalizers = slice.RemoveString(cpgClone.Finalizers, scheduling.CompositePodGroupProtectionFinalizer, nil)
	_, err := c.kubeClient.SchedulingV1alpha3().CompositePodGroups(cpgClone.Namespace).Update(ctx, cpgClone, metav1.UpdateOptions{})
	if err != nil {
		logger.Error(err, "Error removing protection finalizer from CompositePodGroup", "compositePodGroup", klog.KObj(cpg))
		return err
	}

	logger.V(3).Info("Removed protection finalizer from CompositePodGroup", "compositePodGroup", klog.KObj(cpg))
	return nil
}

// hasActivePods returns true if any active pods reference the PodGroup
// via spec.schedulingGroup.podGroupName. The index only contains
// non-terminated pods, so a non-empty result means the PodGroup is still in use.
func (c *Controller) hasActivePods(ctx context.Context, pg *schedulingv1beta1.PodGroup) (bool, error) {
	logger := klog.FromContext(ctx)
	indexKey := pg.Namespace + "/" + pg.Name

	objs, err := c.podIndexer.ByIndex(activePodSchedulingGroupIndex, indexKey)
	if err != nil {
		return false, fmt.Errorf("index-based list of active pods failed for PodGroup %s: %w", indexKey, err)
	}

	if len(objs) > 0 {
		logger.V(4).Info("Pod is using PodGroup", "pod", klog.KObj(objs[0].(*v1.Pod)), "podGroup", klog.KObj(pg))
		return true, nil
	}

	logger.V(4).Info("No active pods found using PodGroup", "podGroup", klog.KObj(pg))
	return false, nil
}

// hasChildGroups returns true if any child PodGroups or child CompositePodGroups
// reference the parent CompositePodGroup via spec.parentCompositePodGroupName.
func (c *Controller) hasChildGroups(ctx context.Context, cpg *schedulingv1alpha3.CompositePodGroup) (bool, error) {
	logger := klog.FromContext(ctx)
	indexKey := cpg.Namespace + "/" + cpg.Name

	if c.podGroupIndexer != nil {
		pgObjs, err := c.podGroupIndexer.ByIndex(childPodGroupParentCompositeGroupIndex, indexKey)
		if err != nil {
			return false, fmt.Errorf("index-based list of child PodGroups failed for CompositePodGroup %s: %w", indexKey, err)
		}
		if len(pgObjs) > 0 {
			logger.V(4).Info("Child PodGroup is using CompositePodGroup", "childPodGroup", klog.KObj(pgObjs[0].(*schedulingv1beta1.PodGroup)), "compositePodGroup", klog.KObj(cpg))
			return true, nil
		}
	}

	if c.compositePodGroupIndexer != nil {
		cpgObjs, err := c.compositePodGroupIndexer.ByIndex(childCompositePodGroupParentCompositeGroupIndex, indexKey)
		if err != nil {
			return false, fmt.Errorf("index-based list of child CompositePodGroups failed for CompositePodGroup %s: %w", indexKey, err)
		}
		if len(cpgObjs) > 0 {
			logger.V(4).Info("Child CompositePodGroup is using CompositePodGroup", "childCompositePodGroup", klog.KObj(cpgObjs[0].(*schedulingv1alpha3.CompositePodGroup)), "compositePodGroup", klog.KObj(cpg))
			return true, nil
		}
	}

	logger.V(4).Info("No child PodGroups or CompositePodGroups found using CompositePodGroup", "compositePodGroup", klog.KObj(cpg))
	return false, nil
}

// isPodTerminated returns true if the pod has completed (Succeeded or Failed).
func isPodTerminated(pod *v1.Pod) bool {
	return pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed
}

// handlePodGroupUpdate handles PodGroup add/delete/update events.
func (c *Controller) handlePodGroupUpdate(logger klog.Logger, old, new any) {
	pg := getPodGroup(new)
	oldPg := getPodGroup(old)

	if pg != nil && protectionutil.IsDeletionCandidate(pg, scheduling.PodGroupProtectionFinalizer) {
		logger.V(4).Info("Got event on PodGroup", "podGroup", klog.KObj(pg))
		c.queue.Add(fwk.PodGroupKey(pg.Namespace, pg.Name))
	}

	// Since ParentCompositePodGroupName is immutable, we can extract it from either the new or old object.
	cpgToCheck := pg
	if oldPg != nil {
		cpgToCheck = oldPg
	}
	if cpgToCheck.Spec.ParentCompositePodGroupName != nil {
		c.queue.Add(fwk.CompositePodGroupKey(cpgToCheck.Namespace, *cpgToCheck.Spec.ParentCompositePodGroupName))
	}
}

// handleCompositePodGroupUpdate handles CompositePodGroup add/delete/update events.
func (c *Controller) handleCompositePodGroupUpdate(logger klog.Logger, old, new any) {
	cpg := getCompositePodGroup(new)
	oldCpg := getCompositePodGroup(old)

	if cpg != nil && protectionutil.IsDeletionCandidate(cpg, scheduling.CompositePodGroupProtectionFinalizer) {
		logger.V(4).Info("Got event on CompositePodGroup", "compositePodGroup", klog.KObj(cpg))
		c.queue.Add(fwk.CompositePodGroupKey(cpg.Namespace, cpg.Name))
	}

	// Since ParentCompositePodGroupName is immutable, we can extract it from either the new or old object.
	cpgToCheck := cpg
	if oldCpg != nil {
		cpgToCheck = oldCpg
	}
	if cpgToCheck.Spec.ParentCompositePodGroupName != nil {
		c.queue.Add(fwk.CompositePodGroupKey(cpgToCheck.Namespace, *cpgToCheck.Spec.ParentCompositePodGroupName))
	}
}

// handlePodChange handles Pod add/delete/update events.
// It enqueues the referenced PodGroup only when the event could affect
// finalizer decisions where the pod is deleted or transitioned to a terminal phase.
func (c *Controller) handlePodChange(logger klog.Logger, old, new any) {
	newPod := getPod(new)
	oldPod := getPod(old)

	if newPod != nil && isPodTerminated(newPod) {
		c.enqueuePodGroupForPod(logger, newPod)
	}

	// An update notification might mask the deletion of a pod X and the
	// following creation of a pod Y with the same namespaced name as X. If
	// that's the case, X needs to be processed as well to handle the case
	// where it was the last active pod keeping the finalizer on a PodGroup.
	if newPod != nil && oldPod != nil && oldPod.UID != newPod.UID {
		c.enqueuePodGroupForPod(logger, oldPod)
	}

	if newPod == nil && oldPod != nil {
		c.enqueuePodGroupForPod(logger, oldPod)
	}
}

// enqueuePodGroupForPod enqueues the PodGroup referenced by the pod.
// Callers are responsible for only passing pods whose state change could allow
// finalizer removal (deleted or transitioned to a terminal phase).
func (c *Controller) enqueuePodGroupForPod(logger klog.Logger, pod *v1.Pod) {
	if pod.Spec.SchedulingGroup == nil || pod.Spec.SchedulingGroup.PodGroupName == nil {
		return
	}

	pgKey := fwk.PodGroupKey(pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
	logger.V(4).Info("Enqueuing PodGroup for pod event", "pod", klog.KObj(pod), "podGroup", pgKey)
	c.queue.Add(pgKey)
}

func getPodGroup(obj any) *schedulingv1beta1.PodGroup {
	if obj == nil {
		return nil
	}
	pg, ok := obj.(*schedulingv1beta1.PodGroup)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
			return nil
		}
		pg, ok = tombstone.Obj.(*schedulingv1beta1.PodGroup)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a PodGroup %#v", obj))
			return nil
		}
	}
	return pg
}

func getCompositePodGroup(obj any) *schedulingv1alpha3.CompositePodGroup {
	if obj == nil {
		return nil
	}
	cpg, ok := obj.(*schedulingv1alpha3.CompositePodGroup)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
			return nil
		}
		cpg, ok = tombstone.Obj.(*schedulingv1alpha3.CompositePodGroup)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a CompositePodGroup %#v", obj))
			return nil
		}
	}
	return cpg
}

func getPod(obj any) *v1.Pod {
	if obj == nil {
		return nil
	}
	pod, ok := obj.(*v1.Pod)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
			return nil
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a Pod %#v", obj))
			return nil
		}
	}
	return pod
}
