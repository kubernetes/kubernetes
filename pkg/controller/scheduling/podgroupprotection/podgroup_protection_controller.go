/*
Copyright 2026 The Kubernetes Authors.

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
	schedulingv1alpha2 "k8s.io/api/scheduling/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	schedulinginformers "k8s.io/client-go/informers/scheduling/v1alpha2"
	clientset "k8s.io/client-go/kubernetes"
	schedulinglisters "k8s.io/client-go/listers/scheduling/v1alpha2"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/protectionutil"
	"k8s.io/kubernetes/pkg/util/slice"
)

const (
	// PodGroupProtectionFinalizer is the finalizer added to PodGroups to prevent
	// premature deletion while pods still reference them.
	PodGroupProtectionFinalizer = "scheduling.k8s.io/podgroup-protection"

	// PodSchedulingGroupIndex is the index name for looking up pods by their
	// schedulingGroup.podGroupName field.
	PodSchedulingGroupIndex = "spec.schedulingGroup.podGroupName"
)

// Controller manages the PodGroupProtectionFinalizer on PodGroup objects.
// It adds the finalizer to PodGroups that have running pods
// and removes it when no active pods reference the PodGroup.
type Controller struct {
	client clientset.Interface

	podGroupLister       schedulinglisters.PodGroupLister
	podGroupListerSynced cache.InformerSynced

	podListerSynced cache.InformerSynced
	podIndexer      cache.Indexer

	queue workqueue.TypedRateLimitingInterface[string]
}

// NewPodGroupProtectionController returns a new instance of the PodGroup protection controller.
func NewPodGroupProtectionController(
	logger klog.Logger,
	podGroupInformer schedulinginformers.PodGroupInformer,
	podInformer coreinformers.PodInformer,
	cl clientset.Interface,
) (*Controller, error) {
	c := &Controller{
		client: cl,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "podgroupprotection"},
		),
	}

	c.podGroupLister = podGroupInformer.Lister()
	c.podGroupListerSynced = podGroupInformer.Informer().HasSynced
	podGroupInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.podGroupAddedUpdated(logger, obj)
		},
		UpdateFunc: func(old, new interface{}) {
			c.podGroupAddedUpdated(logger, new)
		},
	})

	c.podListerSynced = podInformer.Informer().HasSynced
	c.podIndexer = podInformer.Informer().GetIndexer()

	if err := addPodSchedulingGroupIndexer(c.podIndexer); err != nil {
		return nil, fmt.Errorf("could not initialize PodGroup protection controller: %w", err)
	}

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.podAddedDeletedUpdated(logger, nil, obj, false)
		},
		DeleteFunc: func(obj interface{}) {
			c.podAddedDeletedUpdated(logger, nil, obj, true)
		},
		UpdateFunc: func(old, new interface{}) {
			c.podAddedDeletedUpdated(logger, old, new, false)
		},
	})

	return c, nil
}

// addPodSchedulingGroupIndexer adds an indexer to look up pods by their
// schedulingGroup.podGroupName field so we can efficiently look up which
// pods reference a given PodGroup.
func addPodSchedulingGroupIndexer(indexer cache.Indexer) error {
	return indexer.AddIndexers(cache.Indexers{
		PodSchedulingGroupIndex: func(obj interface{}) ([]string, error) {
			pod, ok := obj.(*v1.Pod)
			if !ok {
				return nil, nil
			}
			if pod.Spec.SchedulingGroup != nil && pod.Spec.SchedulingGroup.PodGroupName != nil {
				return []string{pod.Namespace + "/" + *pod.Spec.SchedulingGroup.PodGroupName}, nil
			}
			return nil, nil
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

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.podGroupListerSynced, c.podListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
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
	pgKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(pgKey)

	err := c.processPodGroup(ctx, pgKey)
	if err == nil {
		c.queue.Forget(pgKey)
	} else {
		c.queue.AddRateLimited(pgKey)
		utilruntime.HandleError(fmt.Errorf("PodGroup %v failed with: %w", pgKey, err))
	}
	return true
}

func (c *Controller) processPodGroup(ctx context.Context, pgKey string) error {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Processing PodGroup", "podGroup", pgKey)

	pgNamespace, pgName, err := cache.SplitMetaNamespaceKey(pgKey)
	if err != nil {
		return fmt.Errorf("error parsing PodGroup key %q: %w", pgKey, err)
	}

	pg, err := c.podGroupLister.PodGroups(pgNamespace).Get(pgName)
	if apierrors.IsNotFound(err) {
		logger.V(4).Info("PodGroup not found, ignoring", "podGroup", pgKey)
		return nil
	}
	if err != nil {
		return err
	}

	if protectionutil.IsDeletionCandidate(pg, PodGroupProtectionFinalizer) {
		// PodGroup is being deleted. Check if any active pods still reference it.
		isUsed, err := c.isBeingUsed(ctx, pg)
		if err != nil {
			return err
		}
		if !isUsed {
			return c.removeFinalizer(ctx, pg)
		}
		logger.V(2).Info("Keeping PodGroup finalizer because it is still being used by pods", "podGroup", klog.KObj(pg))
	}

	if protectionutil.NeedToAddFinalizer(pg, PodGroupProtectionFinalizer) {
		// PodGroup is not being deleted then it should have the finalizer if it has active pods.
		isUsed, err := c.isBeingUsed(ctx, pg)
		if err != nil {
			return err
		}
		if isUsed {
			return c.addFinalizer(ctx, pg)
		}
	}

	return nil
}

func (c *Controller) addFinalizer(ctx context.Context, pg *schedulingv1alpha2.PodGroup) error {
	pgClone := pg.DeepCopy()
	pgClone.Finalizers = append(pgClone.Finalizers, PodGroupProtectionFinalizer)
	_, err := c.client.SchedulingV1alpha2().PodGroups(pgClone.Namespace).Update(ctx, pgClone, metav1.UpdateOptions{})
	logger := klog.FromContext(ctx)
	if err != nil {
		logger.Error(err, "Error adding protection finalizer to PodGroup", "podGroup", klog.KObj(pg))
		return err
	}
	logger.V(3).Info("Added protection finalizer to PodGroup", "podGroup", klog.KObj(pg))
	return nil
}

func (c *Controller) removeFinalizer(ctx context.Context, pg *schedulingv1alpha2.PodGroup) error {
	pgClone := pg.DeepCopy()
	pgClone.Finalizers = slice.RemoveString(pgClone.Finalizers, PodGroupProtectionFinalizer, nil)
	_, err := c.client.SchedulingV1alpha2().PodGroups(pgClone.Namespace).Update(ctx, pgClone, metav1.UpdateOptions{})
	logger := klog.FromContext(ctx)
	if err != nil {
		logger.Error(err, "Error removing protection finalizer from PodGroup", "podGroup", klog.KObj(pg))
		return err
	}
	logger.V(3).Info("Removed protection finalizer from PodGroup", "podGroup", klog.KObj(pg))
	return nil
}

// isBeingUsed returns true if any active (non-terminated) pods reference the PodGroup
// via spec.schedulingGroup.podGroupName.
func (c *Controller) isBeingUsed(ctx context.Context, pg *schedulingv1alpha2.PodGroup) (bool, error) {
	logger := klog.FromContext(ctx)
	indexKey := pg.Namespace + "/" + pg.Name

	objs, err := c.podIndexer.ByIndex(PodSchedulingGroupIndex, indexKey)
	if err != nil {
		return false, fmt.Errorf("index-based list of pods failed for PodGroup %s: %w", indexKey, err)
	}

	for _, obj := range objs {
		pod, ok := obj.(*v1.Pod)
		if !ok {
			continue
		}
		// consider only non-terminated pods as "using" the PodGroup.
		if !isPodTerminated(pod) {
			logger.V(4).Info("Pod is using PodGroup", "pod", klog.KObj(pod), "podGroup", klog.KObj(pg))
			return true, nil
		}
	}

	logger.V(4).Info("No active pods found using PodGroup", "podGroup", klog.KObj(pg))
	return false, nil
}

// isPodTerminated returns true if the pod has completed (Succeeded or Failed).
func isPodTerminated(pod *v1.Pod) bool {
	return pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed
}

// podGroupAddedUpdated handles PodGroup add/update events.
func (c *Controller) podGroupAddedUpdated(logger klog.Logger, obj interface{}) {
	pg, ok := obj.(*schedulingv1alpha2.PodGroup)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("PodGroup informer returned non-PodGroup object: %#v", obj))
		return
	}
	key, err := cache.MetaNamespaceKeyFunc(pg)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for PodGroup %#v: %v", pg, err))
		return
	}
	logger.V(4).Info("Got event on PodGroup", "podGroup", klog.KObj(pg))

	if protectionutil.NeedToAddFinalizer(pg, PodGroupProtectionFinalizer) || protectionutil.IsDeletionCandidate(pg, PodGroupProtectionFinalizer) {
		c.queue.Add(key)
	}
}

// podAddedDeletedUpdated handles Pod add/delete/update events.
// It enqueues the referenced PodGroup only when the event could affect
// finalizer decisions (pod added, deleted, or transitioned to a terminal phase).
func (c *Controller) podAddedDeletedUpdated(logger klog.Logger, old, new interface{}, deleted bool) {
	pod := parsePod(new)
	if pod == nil {
		return
	}
	c.enqueuePodGroupForPod(logger, pod, deleted)

	// An update notification might mask the deletion of a pod X and the
	// following creation of a pod Y with the same namespaced name as X. If
	// that's the case, X needs to be processed as well to handle the case
	// where it was the last active pod keeping the finalizer on a PodGroup.
	if oldPod := parsePod(old); oldPod != nil && oldPod.UID != pod.UID {
		c.enqueuePodGroupForPod(logger, oldPod, true)
	}
}

// enqueuePodGroupForPod enqueues the PodGroup referenced by the pod, but only
// when the event is relevant to finalizer decisions:
//   - Pod was deleted (active count may have decreased → may remove finalizer)
//   - Pod reached a terminal phase (active count decreased → may remove finalizer)
//   - Pod was just created, not yet scheduled (active count increased → may add finalizer)
//
// Updates that don't affect the active pod count (e.g., label changes, container
// restarts, readiness transitions) are filtered out.
func (c *Controller) enqueuePodGroupForPod(logger klog.Logger, pod *v1.Pod, deleted bool) {
	if pod.Spec.SchedulingGroup == nil || pod.Spec.SchedulingGroup.PodGroupName == nil {
		return
	}

	// Skip non-deleted, non-terminated pods that are already scheduled.
	// These are running pods whose updates don't change the active pod count.
	// Newly created pods (NodeName == "") pass through so we can add the finalizer.
	if !deleted && !isPodTerminated(pod) && pod.Spec.NodeName != "" {
		return
	}

	pgKey := pod.Namespace + "/" + *pod.Spec.SchedulingGroup.PodGroupName
	logger.V(4).Info("Enqueuing PodGroup for pod event", "pod", klog.KObj(pod), "podGroup", pgKey)
	c.queue.Add(pgKey)
}

func parsePod(obj interface{}) *v1.Pod {
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
