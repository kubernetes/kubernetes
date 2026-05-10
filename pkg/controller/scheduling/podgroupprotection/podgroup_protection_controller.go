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
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/controller/util/protectionutil"
	"k8s.io/kubernetes/pkg/util/slice"
)

const (
	// The index name for looking up active pods by their
	// schedulingGroup.podGroupName field.
	activePodSchedulingGroupIndex = "activePodSchedulingGroup"
)

// Controller manages the PodGroupProtectionFinalizer on PodGroup objects.
// The finalizer is stamped at creation time by the PodGroupProtection admission
// plugin; this controller removes it when the PodGroup is being deleted and no
// active (non-terminated) pods still reference it.
type Controller struct {
	kubeClient clientset.Interface

	podGroupLister schedulinglisters.PodGroupLister
	podGroupSynced cache.InformerSynced

	podSynced cache.InformerSynced

	// podIndexer has the common Pod indexer installed to
	// limit iteration over pods to those of interest.
	podIndexer cache.Indexer

	queue workqueue.TypedRateLimitingInterface[string]
}

// NewPodGroupProtectionController returns a new instance of the PodGroup protection controller.
func NewPodGroupProtectionController(
	logger klog.Logger,
	podGroupInformer schedulinginformers.PodGroupInformer,
	podInformer coreinformers.PodInformer,
	kubeClient clientset.Interface,
) (*Controller, error) {
	c := &Controller{
		kubeClient:     kubeClient,
		podGroupLister: podGroupInformer.Lister(),
		podGroupSynced: podGroupInformer.Informer().HasSynced,
		podIndexer:     podInformer.Informer().GetIndexer(),
		podSynced:      podInformer.Informer().HasSynced,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "podgroupprotection"},
		),
	}

	if _, err := podGroupInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.handlePodGroupUpdate(logger, obj)
		},
		UpdateFunc: func(old, new interface{}) {
			c.handlePodGroupUpdate(logger, new)
		},
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, err
	}

	if err := addActivePodSchedulingGroupIndexer(c.podIndexer); err != nil {
		return nil, fmt.Errorf("could not initialize PodGroup protection controller: %w", err)
	}

	if _, err := podInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.handlePodChange(logger, nil, obj)
		},
		DeleteFunc: func(obj interface{}) {
			c.handlePodChange(logger, obj, nil)
		},
		UpdateFunc: func(old, new interface{}) {
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
		activePodSchedulingGroupIndex: func(obj interface{}) ([]string, error) {
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

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.podGroupSynced, c.podSynced) {
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
	pgKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(pgKey)

	err := c.processPodGroup(ctx, pgKey)
	if err == nil {
		c.queue.Forget(pgKey)
		return true
	}

	c.queue.AddRateLimited(pgKey)
	utilruntime.HandleError(fmt.Errorf("PodGroup %v failed with: %w", pgKey, err))

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

func (c *Controller) removeFinalizer(ctx context.Context, pg *schedulingv1alpha2.PodGroup) error {
	logger := klog.FromContext(ctx)
	pgClone := pg.DeepCopy()

	pgClone.Finalizers = slice.RemoveString(pgClone.Finalizers, scheduling.PodGroupProtectionFinalizer, nil)
	_, err := c.kubeClient.SchedulingV1alpha2().PodGroups(pgClone.Namespace).Update(ctx, pgClone, metav1.UpdateOptions{})
	if err != nil {
		logger.Error(err, "Error removing protection finalizer from PodGroup", "podGroup", klog.KObj(pg))
		return err
	}

	logger.V(3).Info("Removed protection finalizer from PodGroup", "podGroup", klog.KObj(pg))
	return nil
}

// hasActivePods returns true if any active pods reference the PodGroup
// via spec.schedulingGroup.podGroupName. The index only contains
// non-terminated pods, so a non-empty result means the PodGroup is still in use.
func (c *Controller) hasActivePods(ctx context.Context, pg *schedulingv1alpha2.PodGroup) (bool, error) {
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

// isPodTerminated returns true if the pod has completed (Succeeded or Failed).
func isPodTerminated(pod *v1.Pod) bool {
	return pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed
}

// handlePodGroupUpdate handles PodGroup add/update events.
// Only deletion candidates which are being deleted and have the finalizer need processing.
func (c *Controller) handlePodGroupUpdate(logger klog.Logger, obj interface{}) {
	pg, ok := obj.(*schedulingv1alpha2.PodGroup)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("PodGroup informer returned non-PodGroup object: %#v", obj))
		return
	}
	if !protectionutil.IsDeletionCandidate(pg, scheduling.PodGroupProtectionFinalizer) {
		return
	}
	key, err := cache.MetaNamespaceKeyFunc(pg)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for PodGroup %#v: %w", pg, err))
		return
	}
	logger.V(4).Info("Got event on PodGroup", "podGroup", klog.KObj(pg))
	c.queue.Add(key)
}

// handlePodChange handles Pod add/delete/update events.
// It enqueues the referenced PodGroup only when the event could affect
// finalizer decisions where the pod is deleted or transitioned to a terminal phase.
func (c *Controller) handlePodChange(logger klog.Logger, old, new interface{}) {
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

	pgKey := pod.Namespace + "/" + *pod.Spec.SchedulingGroup.PodGroupName
	logger.V(4).Info("Enqueuing PodGroup for pod event", "pod", klog.KObj(pod), "podGroup", pgKey)
	c.queue.Add(pgKey)
}

func getPod(obj interface{}) *v1.Pod {
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
