/*
Copyright 2020 The Kubernetes Authors.

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

package resourceclaim

import (
	"context"
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	corev1informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	kcache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-helpers/cdi/resourceclaim"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/resourceclaim/metrics"
)

const (
	// podResourceClaimIndex is the lookup name for the index function which indexes by pod ResourceClaim templates.
	podResourceClaimIndex = "pod-resource-claim-index"
)

// Controller creates ResourceClaims for ResourceClaimTemplates in a pod spec.
type Controller interface {
	Run(ctx context.Context, workers int)
}

type resourceClaimController struct {
	// kubeClient is the kube API client used to communicate with the API
	// server.
	kubeClient clientset.Interface

	// claimLister is the shared ResourceClaim lister used to fetch and store ResourceClaim
	// objects from the API server. It is shared with other controllers and
	// therefore the ResourceClaim objects in its store should be treated as immutable.
	claimLister  corev1listers.ResourceClaimLister
	claimsSynced kcache.InformerSynced

	// podLister is the shared Pod lister used to fetch Pod
	// objects from the API server. It is shared with other controllers and
	// therefore the Pod objects in its store should be treated as immutable.
	podLister corev1listers.PodLister
	podSynced kcache.InformerSynced

	// podIndexer has the common PodResourceClaim indexer indexer installed To
	// limit iteration over pods to those of interest.
	podIndexer cache.Indexer

	// recorder is used to record events in the API server
	recorder record.EventRecorder

	queue workqueue.RateLimitingInterface
}

const (
	claimKeyPrefix = "claim:"
	podKeyPrefix   = "pod:"
)

// NewController creates a ResourceClaim controller.
func NewController(
	kubeClient clientset.Interface,
	podInformer corev1informers.PodInformer,
	claimInformer corev1informers.ResourceClaimInformer) (Controller, error) {

	ec := &resourceClaimController{
		kubeClient:   kubeClient,
		podLister:    podInformer.Lister(),
		podIndexer:   podInformer.Informer().GetIndexer(),
		podSynced:    podInformer.Informer().HasSynced,
		claimLister:  claimInformer.Lister(),
		claimsSynced: claimInformer.Informer().HasSynced,
		queue:        workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "resource_claim"),
	}

	metrics.RegisterMetrics()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(klog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	ec.recorder = eventBroadcaster.NewRecorder(scheme.Scheme, corev1.EventSource{Component: "resource_claim"})

	podInformer.Informer().AddEventHandler(kcache.ResourceEventHandlerFuncs{
		AddFunc: ec.enqueuePod,
		UpdateFunc: func(old, updated interface{}) {
			ec.enqueuePod(updated)
		},
		DeleteFunc: ec.enqueuePod,
	})
	claimInformer.Informer().AddEventHandler(kcache.ResourceEventHandlerFuncs{
		DeleteFunc: ec.onResourceClaimDelete,
	})
	if err := ec.podIndexer.AddIndexers(cache.Indexers{podResourceClaimIndex: podResourceClaimIndexFunc}); err != nil {
		return nil, fmt.Errorf("could not initialize ResourceClaim controller: %w", err)
	}

	return ec, nil
}

func (ec *resourceClaimController) enqueuePod(obj interface{}) {
	if d, ok := obj.(kcache.DeletedFinalStateUnknown); ok {
		obj = d.Obj
	}
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		// Not a pod?!
		return
	}

	if len(pod.Spec.ResourceClaims) == 0 {
		// Nothing to do for it at all.
		return
	}

	// Release reservations of a deleted or completed pod?
	if pod.DeletionTimestamp != nil && pod.DeletionGracePeriodSeconds != nil && *pod.DeletionGracePeriodSeconds == 0 ||
		pod.Status.Phase == corev1.PodFailed ||
		pod.Status.Phase == corev1.PodSucceeded {
		for _, podClaim := range pod.Spec.ResourceClaims {
			claimName := resourceclaim.Name(pod, &podClaim)
			ec.queue.Add(claimKeyPrefix + pod.Namespace + "/" + claimName)
		}
	}

	// Create ResourceClaim for inline templates?
	if pod.DeletionTimestamp == nil {
		for _, podClaim := range pod.Spec.ResourceClaims {
			if podClaim.Claim.Template != nil {
				// It has at least one inline template, work on it.
				ec.queue.Add(podKeyPrefix + pod.Namespace + "/" + pod.Name)
				break
			}
		}
	}
}

func (ec *resourceClaimController) onResourceClaimDelete(obj interface{}) {
	claim, ok := obj.(*corev1.ResourceClaim)
	if !ok {
		return
	}

	// Someone deleted a ResourceClaim, either intentionally or
	// accidentally. If there is a pod referencing it because of
	// an inline resource, then we should re-create the ResourceClaim.
	// The common indexer does some prefiltering for us by
	// limiting the list to those pods which reference
	// the ResourceClaim.
	objs, err := ec.podIndexer.ByIndex(podResourceClaimIndex, fmt.Sprintf("%s/%s", claim.Namespace, claim.Name))
	if err != nil {
		runtime.HandleError(fmt.Errorf("listing pods from cache: %v", err))
		return
	}
	for _, obj := range objs {
		ec.enqueuePod(obj)
	}
}

func (ec *resourceClaimController) Run(ctx context.Context, workers int) {
	defer runtime.HandleCrash()
	defer ec.queue.ShutDown()

	klog.Infof("Starting ephemeral volume controller")
	defer klog.Infof("Shutting down ephemeral volume controller")

	if !cache.WaitForNamedCacheSync("ephemeral", ctx.Done(), ec.podSynced, ec.claimsSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, ec.runWorker, time.Second)
	}

	<-ctx.Done()
}

func (ec *resourceClaimController) runWorker(ctx context.Context) {
	for ec.processNextWorkItem(ctx) {
	}
}

func (ec *resourceClaimController) processNextWorkItem(ctx context.Context) bool {
	key, shutdown := ec.queue.Get()
	if shutdown {
		return false
	}
	defer ec.queue.Done(key)

	logger := klog.FromContext(ctx).WithValues("key", key)
	ctx = klog.NewContext(ctx, logger)
	err := ec.syncHandler(ctx, key.(string))
	if err == nil {
		ec.queue.Forget(key)
		return true
	}

	runtime.HandleError(fmt.Errorf("%v failed with: %v", key, err))
	ec.queue.AddRateLimited(key)

	return true
}

// syncHandler is invoked for each work item which might need to be processed.
// If an error is returned from this function, the item will be requeued.
func (ec *resourceClaimController) syncHandler(ctx context.Context, key string) error {
	sep := strings.Index(key, ":")
	if sep < 0 {
		return fmt.Errorf("unexpected key: %s", key)
	}
	prefix, object := key[0:sep+1], key[sep+1:]
	namespace, name, err := kcache.SplitMetaNamespaceKey(object)
	if err != nil {
		return err
	}

	switch prefix {
	case podKeyPrefix:
		return ec.syncPod(ctx, namespace, name)
	case claimKeyPrefix:
		return ec.syncClaim(ctx, namespace, name)
	default:
		return fmt.Errorf("unexpected key prefix: %s", prefix)
	}

}

func (ec *resourceClaimController) syncPod(ctx context.Context, namespace, name string) error {
	logger := klog.FromContext(ctx)
	pod, err := ec.podLister.Pods(namespace).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(5).Info("nothing to do for pod %s/%s, it is gone", namespace, name)
			return nil
		}
		return err
	}

	// Ignore pods which are already getting deleted.
	if pod.DeletionTimestamp != nil {
		logger.V(5).Info("nothing to do for pod %s/%s, it is marked for deletion", name, namespace)
		return nil
	}

	for _, podClaim := range pod.Spec.ResourceClaims {
		if err := ec.handleClaim(ctx, pod, podClaim); err != nil {
			ec.recorder.Event(pod, corev1.EventTypeWarning, "ResourceClaimCreation", fmt.Sprintf("PodResourceClaim %s: %v", podClaim.Name, err))
			return fmt.Errorf("pod %s/%s, PodResourceClaim %s: %v", namespace, name, podClaim.Name, err)
		}
	}

	return nil
}

// handleResourceClaim is invoked for each volume of a pod.
func (ec *resourceClaimController) handleClaim(ctx context.Context, pod *corev1.Pod, podClaim corev1.PodResourceClaim) error {
	logger := klog.FromContext(ctx)
	logger.V(5).Info("checking podClaim %s", podClaim.Name)
	if podClaim.Claim.Template == nil {
		return nil
	}

	claimName := resourceclaim.Name(pod, &podClaim)
	claim, err := ec.claimLister.ResourceClaims(pod.Namespace).Get(claimName)
	if err != nil && !errors.IsNotFound(err) {
		return err
	}
	if claim != nil {
		if err := resourceclaim.IsForPod(pod, claim); err != nil {
			return err
		}
		// Already created, nothing more to do.
		logger.V(5).Info("podClaim %s: ResourceClaim %s already created", podClaim.Name, claimName)
		return nil
	}

	// Create the ResourceClaim with pod as owner.
	isTrue := true
	claim = &corev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: claimName,
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion:         "v1",
					Kind:               "Pod",
					Name:               pod.Name,
					UID:                pod.UID,
					Controller:         &isTrue,
					BlockOwnerDeletion: &isTrue,
				},
			},
			Annotations: podClaim.Claim.Template.Annotations,
			Labels:      podClaim.Claim.Template.Labels,
		},
		Spec: podClaim.Claim.Template.Spec,
	}
	metrics.ResourceClaimCreateAttempts.Inc()
	_, err = ec.kubeClient.CoreV1().ResourceClaims(pod.Namespace).Create(ctx, claim, metav1.CreateOptions{})
	if err != nil {
		metrics.ResourceClaimCreateFailures.Inc()
		return fmt.Errorf("create ResourceClaim %s: %v", claimName, err)
	}
	return nil
}

func (ec *resourceClaimController) syncClaim(ctx context.Context, namespace, name string) error {
	logger := klog.FromContext(ctx)
	claim, err := ec.claimLister.ResourceClaims(namespace).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(5).Info("nothing to do for claim %s/%s, it is gone", namespace, name)
			return nil
		}
		return err
	}

	// Check if the ReservedFor entries are all still valid.
	valid := make([]corev1.ResourceClaimUserReference, 0, len(claim.Status.ReservedFor))
	for _, reservedFor := range claim.Status.ReservedFor {
		if reservedFor.Version == "v1" &&
			reservedFor.Group == "" &&
			reservedFor.Resource == "pods" {
			pod, err := ec.podLister.Pods(claim.Namespace).Get(reservedFor.Name)
			notFound := errors.IsNotFound(err)
			if err != nil && !notFound {
				return err
			}
			if pod != nil && pod.UID == reservedFor.UID {
				valid = append(valid, reservedFor)
			}
			continue
		}

		// TODO: support generic object lookup
		return fmt.Errorf("unsupported ReservedFor entry: %v", reservedFor)
	}

	if len(valid) < len(claim.Status.ReservedFor) {
		// TODO: patch
		claim := claim.DeepCopy()
		claim.Status.ReservedFor = valid
		_, err := ec.kubeClient.CoreV1().ResourceClaims(claim.Namespace).UpdateStatus(ctx, claim, metav1.UpdateOptions{})
		if err != nil {
			return err
		}
	}

	return nil
}

// podResourceClaimIndexFunc is an index function that returns ResourceClaim keys (=
// namespace/name) for ResourceClaimTemplates in a given pod.
func podResourceClaimIndexFunc(obj interface{}) ([]string, error) {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return []string{}, nil
	}
	keys := []string{}
	for _, podClaim := range pod.Spec.ResourceClaims {
		if podClaim.Claim.Template != nil {
			claimName := resourceclaim.Name(pod, &podClaim)
			keys = append(keys, fmt.Sprintf("%s/%s", pod.Namespace, claimName))
		}
	}
	return keys, nil
}
