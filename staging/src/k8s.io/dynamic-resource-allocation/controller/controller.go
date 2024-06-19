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

package controller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1alpha3"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	corev1types "k8s.io/client-go/kubernetes/typed/core/v1"
	resourcelisters "k8s.io/client-go/listers/resource/v1alpha3"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
)

// Controller watches ResourceClaims and triggers allocation and deallocation
// as needed.
type Controller interface {
	// Run starts the controller.
	Run(workers int)

	// SetReservedFor can be used to disable adding the Pod which
	// triggered allocation to the status.reservedFor. Normally,
	// DRA drivers should always do that, so it's the default.
	// But nothing in the protocol between the scheduler and
	// a driver requires it, so at least for testing the control
	// plane components it is useful to disable it.
	SetReservedFor(enabled bool)
}

// Driver provides the actual allocation and deallocation operations.
type Driver interface {
	// Allocate is called when all same-driver ResourceClaims for Pod are ready
	// to be allocated. The selectedNode is empty for ResourceClaims with immediate
	// allocation, in which case the resource driver decides itself where
	// to allocate. If there is already an on-going allocation, the driver
	// may finish it and ignore the new parameters or abort the on-going
	// allocation and try again with the new parameters.
	//
	// Parameters have been retrieved earlier.
	//
	// Driver must set the result of allocation for every claim in "claims"
	// parameter items. If there is no error and allocation
	// is successful - claims[i].Allocation field should be set. In case of
	// particular claim allocation failure - respective item's claims[i].Error field
	// should be set and claims[i].Allocation will be ignored.
	//
	// If selectedNode is set, the driver must attempt to allocate for that
	// node. If that is not possible, it must return an error. The
	// controller will call UnsuitableNodes and pass the new information to
	// the scheduler, which will then lead to selecting a different node
	// if the current one is not suitable.
	//
	// The Claim, ClaimParameters, Class, ClassParameters fields of "claims" parameter
	// items are read-only and must not be modified. This call must be idempotent.
	Allocate(ctx context.Context, claims []*ClaimAllocation, selectedNode string)

	// Deallocate gets called when a ResourceClaim is ready to be
	// freed.
	//
	// The claim is read-only and must not be modified. This call must be
	// idempotent. In particular it must not return an error when the claim
	// is currently not allocated.
	//
	// Deallocate may be called when a previous allocation got
	// interrupted. Deallocate must then stop any on-going allocation
	// activity and free resources before returning without an error.
	Deallocate(ctx context.Context, claim *resourceapi.ResourceClaim) error

	// UnsuitableNodes checks all pending claims with delayed allocation
	// for a pod. All claims are ready for allocation by the driver
	// and parameters have been retrieved.
	//
	// The driver may consider each claim in isolation, but it's better
	// to mark nodes as unsuitable for all claims, if all claims
	// cannot be allocated for it (for example, two GPUs requested but
	// the node only has one).
	//
	// The potentialNodes slice contains all potential nodes selected
	// by the scheduler plus the selected node. The response must
	// not contain any other nodes. Implementations do not have to
	// care about size limits in the PodSchedulingContext status, the
	// caller will handle that.
	//
	// The result of the check is in ClaimAllocation.UnsuitableNodes.
	// An error indicates that the entire check must be repeated.
	UnsuitableNodes(ctx context.Context, pod *v1.Pod, claims []*ClaimAllocation, potentialNodes []string) error
}

// ClaimAllocation represents information about one particular
// pod.Spec.ResourceClaim entry.
type ClaimAllocation struct {
	PodClaimName  string
	Claim         *resourceapi.ResourceClaim
	DeviceClasses map[string]*resourceapi.DeviceClass

	// UnsuitableNodes needs to be filled in by the driver when
	// Driver.UnsuitableNodes gets called.
	UnsuitableNodes []string

	// Driver must populate this field with resources that were
	// allocated for the claim in case of successful allocation.
	Allocation *resourceapi.AllocationResult
	// In case of error allocating particular claim, driver must
	// populate this field.
	Error error
}

type controller struct {
	ctx                 context.Context
	logger              klog.Logger
	name                string
	finalizer           string
	driver              Driver
	setReservedFor      bool
	kubeClient          kubernetes.Interface
	claimNameLookup     *resourceclaim.Lookup
	queue               workqueue.TypedRateLimitingInterface[string]
	eventRecorder       record.EventRecorder
	dcLister            resourcelisters.DeviceClassLister
	claimCache          cache.MutationCache
	schedulingCtxLister resourcelisters.PodSchedulingContextLister
	synced              []cache.InformerSynced
}

// TODO: make it configurable
var recheckDelay = 30 * time.Second

// New creates a new controller.
func New(
	ctx context.Context,
	name string,
	driver Driver,
	kubeClient kubernetes.Interface,
	informerFactory informers.SharedInformerFactory) Controller {
	logger := klog.LoggerWithName(klog.FromContext(ctx), "resource controller")
	dcInformer := informerFactory.Resource().V1alpha3().DeviceClasses()
	claimInformer := informerFactory.Resource().V1alpha3().ResourceClaims()
	schedulingCtxInformer := informerFactory.Resource().V1alpha3().PodSchedulingContexts()
	claimNameLookup := resourceclaim.NewNameLookup(kubeClient)

	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	go func() {
		<-ctx.Done()
		eventBroadcaster.Shutdown()
	}()
	// TODO: use contextual logging in eventBroadcaster once it
	// supports it. There is a StartStructuredLogging API, but it
	// uses the global klog, which is worse than redirecting an unstructured
	// string into our logger, in particular during testing.
	eventBroadcaster.StartLogging(func(format string, args ...interface{}) {
		helper, logger := logger.WithCallStackHelper()
		helper()
		logger.V(2).Info(fmt.Sprintf(format, args...))
	})
	eventBroadcaster.StartRecordingToSink(&corev1types.EventSinkImpl{Interface: kubeClient.CoreV1().Events(v1.NamespaceAll)})
	eventRecorder := eventBroadcaster.NewRecorder(scheme.Scheme,
		v1.EventSource{Component: fmt.Sprintf("resource driver %s", name)})

	// The work queue contains either keys for claims or PodSchedulingContext objects.
	queue := workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[string](),
		workqueue.TypedRateLimitingQueueConfig[string]{Name: fmt.Sprintf("%s-queue", name)},
	)

	// The mutation cache acts as an additional layer for the informer
	// cache and after an update made by the controller returns a more
	// recent copy until the informer catches up.
	claimInformerCache := claimInformer.Informer().GetIndexer()
	claimCache := cache.NewIntegerResourceVersionMutationCache(claimInformerCache, claimInformerCache, 60*time.Second,
		false /* only cache updated claims that exist in the informer cache */)

	ctrl := &controller{
		ctx:                 ctx,
		logger:              logger,
		name:                name,
		finalizer:           name + "/deletion-protection",
		driver:              driver,
		setReservedFor:      true,
		kubeClient:          kubeClient,
		claimNameLookup:     claimNameLookup,
		dcLister:            dcInformer.Lister(),
		claimCache:          claimCache,
		schedulingCtxLister: schedulingCtxInformer.Lister(),
		queue:               queue,
		eventRecorder:       eventRecorder,
		synced: []cache.InformerSynced{
			dcInformer.Informer().HasSynced,
			claimInformer.Informer().HasSynced,
			schedulingCtxInformer.Informer().HasSynced,
		},
	}

	loggerV6 := logger.V(6)
	if loggerV6.Enabled() {
		resourceClaimLogger := klog.LoggerWithValues(loggerV6, "type", "ResourceClaim")
		_, _ = claimInformer.Informer().AddEventHandler(resourceEventHandlerFuncs(&resourceClaimLogger, ctrl))
		schedulingCtxLogger := klog.LoggerWithValues(loggerV6, "type", "PodSchedulingContext")
		_, _ = schedulingCtxInformer.Informer().AddEventHandler(resourceEventHandlerFuncs(&schedulingCtxLogger, ctrl))
	} else {
		_, _ = claimInformer.Informer().AddEventHandler(resourceEventHandlerFuncs(nil, ctrl))
		_, _ = schedulingCtxInformer.Informer().AddEventHandler(resourceEventHandlerFuncs(nil, ctrl))
	}

	return ctrl
}

func (ctrl *controller) SetReservedFor(enabled bool) {
	ctrl.setReservedFor = enabled
}

func resourceEventHandlerFuncs(logger *klog.Logger, ctrl *controller) cache.ResourceEventHandlerFuncs {
	return cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			ctrl.add(logger, obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			ctrl.update(logger, oldObj, newObj)
		},
		DeleteFunc: ctrl.delete,
	}
}

const (
	claimKeyPrefix         = "claim:"
	schedulingCtxKeyPrefix = "schedulingCtx:"
)

func (ctrl *controller) add(loggerV6 *klog.Logger, obj interface{}) {
	var logger klog.Logger
	if loggerV6 != nil {
		logger = loggerV6.WithValues("object", prettyPrint(obj))
	} else {
		logger = ctrl.logger.V(5)
	}
	ctrl.addNewOrUpdated(logger, "Adding new work item", obj)
}

func (ctrl *controller) update(loggerV6 *klog.Logger, oldObj, newObj interface{}) {
	var logger klog.Logger
	if loggerV6 != nil {
		diff := cmp.Diff(oldObj, newObj)
		logger = loggerV6.WithValues("object", prettyPrint(newObj), "diff", diff)
	} else {
		logger = ctrl.logger.V(5)
	}
	ctrl.addNewOrUpdated(logger, "Adding updated work item", newObj)
}

func (ctrl *controller) addNewOrUpdated(loggerV klog.Logger, msg string, obj interface{}) {
	objKey, err := getKey(obj)
	if err != nil {
		loggerV.Error(err, "Failed to get key", "obj", obj)
		return
	}
	loggerV.Info(msg, "key", objKey)
	ctrl.queue.Add(objKey)
}

func (ctrl *controller) delete(obj interface{}) {
	objKey, err := getKey(obj)
	if err != nil {
		return
	}
	ctrl.logger.V(5).Info("Removing deleted work item", "key", objKey)
	ctrl.queue.Forget(objKey)
}

func getKey(obj interface{}) (string, error) {
	objKey, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		return "", err
	}
	prefix := ""
	switch obj.(type) {
	case *resourceapi.ResourceClaim:
		prefix = claimKeyPrefix
	case *resourceapi.PodSchedulingContext:
		prefix = schedulingCtxKeyPrefix
	default:
		return "", fmt.Errorf("unexpected object: %T", obj)
	}

	return prefix + objKey, nil
}

// Run starts the controller.
func (ctrl *controller) Run(workers int) {
	defer ctrl.queue.ShutDown()

	ctrl.logger.Info("Starting", "driver", ctrl.name)
	defer ctrl.logger.Info("Shutting down", "driver", ctrl.name)

	stopCh := ctrl.ctx.Done()

	if !cache.WaitForCacheSync(stopCh, ctrl.synced...) {
		ctrl.logger.Error(nil, "Cannot sync caches")
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(ctrl.sync, 0, stopCh)
	}

	<-stopCh
}

// errRequeue is a special error instance that functions can return
// to request silent requeueing (not logged as error, no event).
// Uses exponential backoff.
var errRequeue = errors.New("requeue")

// errPeriodic is a special error instance that functions can return
// to request silent retrying at a fixed rate.
var errPeriodic = errors.New("periodic")

// sync is the main worker.
func (ctrl *controller) sync() {
	key, quit := ctrl.queue.Get()
	if quit {
		return
	}
	defer ctrl.queue.Done(key)

	logger := klog.LoggerWithValues(ctrl.logger, "key", key)
	ctx := klog.NewContext(ctrl.ctx, logger)
	logger.V(4).Info("processing")
	obj, err := ctrl.syncKey(ctx, key)
	switch err {
	case nil:
		logger.V(5).Info("completed")
		ctrl.queue.Forget(key)
	case errRequeue:
		logger.V(5).Info("requeue")
		ctrl.queue.AddRateLimited(key)
	case errPeriodic:
		logger.V(5).Info("recheck periodically")
		ctrl.queue.AddAfter(key, recheckDelay)
	default:
		logger.Error(err, "processing failed")
		if obj != nil {
			// TODO: We don't know here *what* failed. Determine based on error?
			ctrl.eventRecorder.Event(obj, v1.EventTypeWarning, "Failed", err.Error())
		}
		ctrl.queue.AddRateLimited(key)
	}
}

// syncKey looks up a ResourceClaim by its key and processes it.
func (ctrl *controller) syncKey(ctx context.Context, key string) (obj runtime.Object, finalErr error) {
	sep := strings.Index(key, ":")
	if sep < 0 {
		return nil, fmt.Errorf("unexpected key: %s", key)
	}
	prefix, object := key[0:sep+1], key[sep+1:]
	namespace, name, err := cache.SplitMetaNamespaceKey(object)
	if err != nil {
		return nil, err
	}

	switch prefix {
	case claimKeyPrefix:
		claim, err := ctrl.getCachedClaim(ctx, object)
		if claim == nil || err != nil {
			return nil, err
		}
		obj, finalErr = claim, ctrl.syncClaim(ctx, claim)
	case schedulingCtxKeyPrefix:
		schedulingCtx, err := ctrl.schedulingCtxLister.PodSchedulingContexts(namespace).Get(name)
		if err != nil {
			if k8serrors.IsNotFound(err) {
				klog.FromContext(ctx).V(5).Info("PodSchedulingContext was deleted, no need to process it")
				return nil, nil
			}
			return nil, err
		}
		obj, finalErr = schedulingCtx, ctrl.syncPodSchedulingContexts(ctx, schedulingCtx)
	}
	return
}

func (ctrl *controller) getCachedClaim(ctx context.Context, key string) (*resourceapi.ResourceClaim, error) {
	claimObj, exists, err := ctrl.claimCache.GetByKey(key)
	if !exists || k8serrors.IsNotFound(err) {
		klog.FromContext(ctx).V(5).Info("ResourceClaim not found, no need to process it")
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	claim, ok := claimObj.(*resourceapi.ResourceClaim)
	if !ok {
		return nil, fmt.Errorf("internal error: got %T instead of *resourceapi.ResourceClaim from claim cache", claimObj)
	}
	return claim, nil
}

// syncClaim determines which next action may be needed for a ResourceClaim
// and does it.
func (ctrl *controller) syncClaim(ctx context.Context, claim *resourceapi.ResourceClaim) error {
	var err error
	logger := klog.FromContext(ctx)

	if len(claim.Status.ReservedFor) > 0 {
		// In use. Nothing that we can do for it now.
		if loggerV6 := logger.V(6); loggerV6.Enabled() {
			loggerV6.Info("ResourceClaim in use", "reservedFor", claim.Status.ReservedFor)
		} else {
			logger.V(5).Info("ResourceClaim in use")
		}
		return nil
	}

	if claim.DeletionTimestamp != nil ||
		claim.Status.DeallocationRequested {
		// Ready for deallocation. We might have our finalizer set. The
		// finalizer is specific to the driver, therefore we know that
		// this claim is "ours" when the finalizer is set.
		hasFinalizer := ctrl.hasFinalizer(claim)
		logger.V(5).Info("ResourceClaim ready for deallocation", "deallocationRequested", claim.Status.DeallocationRequested, "deletionTimestamp", claim.DeletionTimestamp, "allocated", claim.Status.Allocation != nil, "hasFinalizer", hasFinalizer)
		if hasFinalizer {
			claim = claim.DeepCopy()
			if claim.Status.Allocation != nil {
				// Allocation was completed. Deallocate before proceeding.
				if err := ctrl.driver.Deallocate(ctx, claim); err != nil {
					return fmt.Errorf("deallocate: %v", err)
				}
				claim.Status.Allocation = nil
				claim.Status.DeallocationRequested = false
				claim, err = ctrl.kubeClient.ResourceV1alpha3().ResourceClaims(claim.Namespace).UpdateStatus(ctx, claim, metav1.UpdateOptions{})
				if err != nil {
					return fmt.Errorf("remove allocation: %v", err)
				}
				ctrl.claimCache.Mutation(claim)
			} else {
				// Ensure that there is no on-going allocation.
				if err := ctrl.driver.Deallocate(ctx, claim); err != nil {
					return fmt.Errorf("stop allocation: %v", err)
				}
			}

			if claim.Status.DeallocationRequested {
				// Still need to remove it.
				claim.Status.DeallocationRequested = false
				claim, err = ctrl.kubeClient.ResourceV1alpha3().ResourceClaims(claim.Namespace).UpdateStatus(ctx, claim, metav1.UpdateOptions{})
				if err != nil {
					return fmt.Errorf("remove deallocation: %v", err)
				}
				ctrl.claimCache.Mutation(claim)
			}

			claim.Finalizers = ctrl.removeFinalizer(claim.Finalizers)
			claim, err = ctrl.kubeClient.ResourceV1alpha3().ResourceClaims(claim.Namespace).Update(ctx, claim, metav1.UpdateOptions{})
			if err != nil {
				return fmt.Errorf("remove finalizer: %v", err)
			}
			ctrl.claimCache.Mutation(claim)
		}

		// Nothing further to do. The apiserver should remove it shortly.
		return nil

	}

	if claim.Status.Allocation != nil {
		logger.V(5).Info("ResourceClaim is allocated")
		return nil
	}
	logger.V(5).Info("ResourceClaim waiting for first consumer")
	return nil
}

// allocateClaims filters list of claims, keeps those needing allocation and asks driver to do the allocations.
// Driver is supposed to write the AllocationResult and Error field into argument claims slice.
func (ctrl *controller) allocateClaims(ctx context.Context, claims []*ClaimAllocation, selectedNode string, selectedUser *resourceapi.ResourceClaimConsumerReference) {
	logger := klog.FromContext(ctx)

	needAllocation := make([]*ClaimAllocation, 0, len(claims))
	for _, claim := range claims {
		if claim.Claim.Status.Allocation != nil {
			// This can happen when two PodSchedulingContext objects trigger
			// allocation attempts (first one wins) or when we see the
			// update of the PodSchedulingContext object.
			logger.V(5).Info("Claim is already allocated, skipping allocation", "claim", claim.PodClaimName)
			continue
		}
		needAllocation = append(needAllocation, claim)
	}

	if len(needAllocation) == 0 {
		logger.V(5).Info("No claims need allocation, nothing to do")
		return
	}

	// Keep separately claims that succeeded adding finalizers,
	// they will be sent for Allocate to the driver.
	claimsWithFinalizers := make([]*ClaimAllocation, 0, len(needAllocation))
	for _, claimAllocation := range needAllocation {
		if !ctrl.hasFinalizer(claimAllocation.Claim) {
			claim := claimAllocation.Claim.DeepCopy()
			// Set finalizer before doing anything. We continue with the updated claim.
			logger.V(5).Info("Adding finalizer", "claim", claim.Name)
			claim.Finalizers = append(claim.Finalizers, ctrl.finalizer)
			var err error
			claim, err = ctrl.kubeClient.ResourceV1alpha3().ResourceClaims(claim.Namespace).Update(ctx, claim, metav1.UpdateOptions{})
			if err != nil {
				logger.Error(err, "add finalizer", "claim", claim.Name)
				claimAllocation.Error = fmt.Errorf("add finalizer: %v", err)
				// Do not save claim to ask for Allocate from Driver.
				continue
			}
			ctrl.claimCache.Mutation(claim)
			claimAllocation.Claim = claim
		}
		claimsWithFinalizers = append(claimsWithFinalizers, claimAllocation)
	}

	// Beyond here we only operate with claimsWithFinalizers because those are ready for allocation.

	logger.V(5).Info("Allocating")
	ctrl.driver.Allocate(ctx, claimsWithFinalizers, selectedNode)

	// Update successfully allocated claims' status with allocation info.
	for _, claimAllocation := range claimsWithFinalizers {
		if claimAllocation.Error != nil {
			logger.Error(claimAllocation.Error, "allocating claim", "claim", claimAllocation.Claim.Name)
			continue
		}
		if claimAllocation.Allocation == nil {
			logger.Error(nil, "allocating claim: missing allocation from driver", "claim", claimAllocation.Claim.Name)
			claimAllocation.Error = fmt.Errorf("allocating claim: missing allocation from driver")
			// Do not update this claim with allocation, it might succeed next time.
			continue
		}
		logger.V(5).Info("successfully allocated", "claim", klog.KObj(claimAllocation.Claim))
		claim := claimAllocation.Claim.DeepCopy()
		claim.Status.Allocation = claimAllocation.Allocation
		claim.Status.Allocation.Controller = ctrl.name
		if selectedUser != nil && ctrl.setReservedFor {
			claim.Status.ReservedFor = append(claim.Status.ReservedFor, *selectedUser)
		}
		logger.V(6).Info("Updating claim after allocation", "claim", claim)
		claim, err := ctrl.kubeClient.ResourceV1alpha3().ResourceClaims(claim.Namespace).UpdateStatus(ctx, claim, metav1.UpdateOptions{})
		if err != nil {
			claimAllocation.Error = fmt.Errorf("add allocation: %v", err)
			continue
		}

		ctrl.claimCache.Mutation(claim)
	}
	return
}

func (ctrl *controller) checkPodClaim(ctx context.Context, pod *v1.Pod, podClaim v1.PodResourceClaim) (*ClaimAllocation, error) {
	claimName, mustCheckOwner, err := ctrl.claimNameLookup.Name(pod, &podClaim)
	if err != nil {
		return nil, err
	}
	if claimName == nil {
		// Nothing to do.
		return nil, nil
	}
	key := pod.Namespace + "/" + *claimName
	claim, err := ctrl.getCachedClaim(ctx, key)
	if claim == nil || err != nil {
		return nil, err
	}
	if mustCheckOwner {
		if err := resourceclaim.IsForPod(pod, claim); err != nil {
			return nil, err
		}
	}
	if claim.Status.Allocation != nil {
		// Already allocated, class and parameter are not needed and nothing
		// need to be done for the claim either.
		return nil, nil
	}
	if claim.Spec.Controller != ctrl.name {
		return nil, nil
	}

	// Sanity checks and preparations...
	ca := &ClaimAllocation{
		PodClaimName:  podClaim.Name,
		Claim:         claim,
		DeviceClasses: make(map[string]*resourceapi.DeviceClass),
	}
	for _, request := range claim.Spec.Devices.Requests {
		deviceRequest := request.Device
		if deviceRequest == nil {
			// Some unknown request. Abort!
			return nil, fmt.Errorf("claim %s: unknown request type in request %s", klog.KObj(claim), request.Name)
		}
		deviceClassName := deviceRequest.DeviceClassName
		if deviceClassName == "" {
			// Should be set, but we don't care, so no error.
			continue
		}
		class, err := ctrl.dcLister.Get(deviceClassName)
		if err != nil {
			return nil, fmt.Errorf("claim %s: request %s: class %s: %v", klog.KObj(claim), request.Name, deviceClassName, err)
		}
		ca.DeviceClasses[deviceClassName] = class
	}

	return ca, nil
}

// syncPodSchedulingContext determines which next action may be needed for a PodSchedulingContext object
// and does it.
func (ctrl *controller) syncPodSchedulingContexts(ctx context.Context, schedulingCtx *resourceapi.PodSchedulingContext) error {
	logger := klog.FromContext(ctx)

	// Ignore deleted objects.
	if schedulingCtx.DeletionTimestamp != nil {
		logger.V(5).Info("PodSchedulingContext marked for deletion")
		return nil
	}

	if schedulingCtx.Spec.SelectedNode == "" &&
		len(schedulingCtx.Spec.PotentialNodes) == 0 {
		// Nothing to do? Shouldn't occur.
		logger.V(5).Info("Waiting for scheduler to set fields")
		return nil
	}

	// Check pod.
	// TODO (?): use an informer - only useful when many (most?) pods have claims
	// TODO (?): let the scheduler copy all claim names + UIDs into PodSchedulingContext - then we don't need the pod
	pod, err := ctrl.kubeClient.CoreV1().Pods(schedulingCtx.Namespace).Get(ctx, schedulingCtx.Name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	if pod.DeletionTimestamp != nil {
		logger.V(5).Info("Pod marked for deletion")
		return nil
	}

	// Still the owner?
	if !metav1.IsControlledBy(schedulingCtx, pod) {
		// Must be obsolete object, do nothing for it.
		logger.V(5).Info("Pod not owner, PodSchedulingContext is obsolete")
		return nil
	}

	// Find all pending claims that are owned by us. We bail out if any of the pre-requisites
	// for pod scheduling (claims exist, classes exist, parameters exist) are not met.
	// The scheduler will do the same, except for checking parameters, so usually
	// everything should be ready once the PodSchedulingContext object exists.
	var claims claimAllocations
	for _, podClaim := range pod.Spec.ResourceClaims {
		delayed, err := ctrl.checkPodClaim(ctx, pod, podClaim)
		if err != nil {
			return fmt.Errorf("pod claim %s: %v", podClaim.Name, err)
		}
		if delayed == nil {
			// Nothing to do for it. This can change, so keep checking.
			continue
		}
		claims = append(claims, delayed)
	}
	if len(claims) == 0 {
		logger.V(5).Info("Found no pending pod claims")
		return errPeriodic
	}

	// Check current resource availability *before* triggering the
	// allocations. If we find that any of the claims cannot be allocated
	// for the selected node, we don't need to try for the others either
	// and shouldn't, because those allocations might have to be undone to
	// pick a better node. If we don't need to allocate now, then we'll
	// simply report back the gather information.
	//
	// We shouldn't assume that the scheduler has included the selected node
	// in the list of potential nodes. Usually it does, but let's make sure
	// that we check it.
	selectedNode := schedulingCtx.Spec.SelectedNode
	potentialNodes := schedulingCtx.Spec.PotentialNodes
	if selectedNode != "" && !hasString(potentialNodes, selectedNode) {
		potentialNodes = append(potentialNodes, selectedNode)
	}
	if len(schedulingCtx.Spec.PotentialNodes) > 0 {
		if err := ctrl.driver.UnsuitableNodes(ctx, pod, claims, potentialNodes); err != nil {
			return fmt.Errorf("checking potential nodes: %v", err)
		}
	}
	logger.V(5).Info("pending pod claims", "claims", claims, "selectedNode", selectedNode)
	if selectedNode != "" {
		unsuitable := false
		for _, delayed := range claims {
			if hasString(delayed.UnsuitableNodes, selectedNode) {
				unsuitable = true
				break
			}
		}

		if unsuitable {
			logger.V(2).Info("skipping allocation for unsuitable selected node", "node", selectedNode)
		} else {
			logger.V(2).Info("allocation for selected node", "node", selectedNode)
			selectedUser := &resourceapi.ResourceClaimConsumerReference{
				Resource: "pods",
				Name:     pod.Name,
				UID:      pod.UID,
			}

			ctrl.allocateClaims(ctx, claims, selectedNode, selectedUser)

			var allErrors []error
			for _, delayed := range claims {
				if delayed.Error != nil {
					if strings.Contains(delayed.Error.Error(), delayed.Claim.Name) {
						// Avoid adding redundant information.
						allErrors = append(allErrors, delayed.Error)
					} else {
						// Include claim name, it's not in the underlying error.
						allErrors = append(allErrors, fmt.Errorf("claim %s: %v", delayed.Claim.Name, delayed.Error))
					}
				}
			}
			if len(allErrors) > 0 {
				return errors.Join(allErrors...)
			}
		}
	}

	// Now update unsuitable nodes. This is useful information for the scheduler even if
	// we managed to allocate because we might have to undo that.
	// TODO: replace with patching the array. We can do that without race conditions
	// because each driver is responsible for its own entries.
	modified := false
	schedulingCtx = schedulingCtx.DeepCopy()
	for _, delayed := range claims {
		i := findClaim(schedulingCtx.Status.ResourceClaims, delayed.PodClaimName)
		if i < 0 {
			// Add new entry.
			schedulingCtx.Status.ResourceClaims = append(schedulingCtx.Status.ResourceClaims,
				resourceapi.ResourceClaimSchedulingStatus{
					Name:            delayed.PodClaimName,
					UnsuitableNodes: truncateNodes(delayed.UnsuitableNodes, selectedNode),
				})
			modified = true
		} else if stringsDiffer(schedulingCtx.Status.ResourceClaims[i].UnsuitableNodes, delayed.UnsuitableNodes) {
			// Update existing entry.
			schedulingCtx.Status.ResourceClaims[i].UnsuitableNodes = truncateNodes(delayed.UnsuitableNodes, selectedNode)
			modified = true
		}
	}
	if modified {
		logger.V(6).Info("Updating pod scheduling with modified unsuitable nodes", "podSchedulingCtx", schedulingCtx)
		if _, err := ctrl.kubeClient.ResourceV1alpha3().PodSchedulingContexts(schedulingCtx.Namespace).UpdateStatus(ctx, schedulingCtx, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("update unsuitable node status: %v", err)
		}
	}

	// We must keep the object in our queue and keep updating the
	// UnsuitableNodes fields.
	return errPeriodic
}

func truncateNodes(nodes []string, selectedNode string) []string {
	// We might have checked "potential nodes + selected node" above, so
	// this list might be too long by one element. When truncating it, make
	// sure that the selected node is listed.
	lenUnsuitable := len(nodes)
	if lenUnsuitable > resourceapi.PodSchedulingNodeListMaxSize {
		if nodes[0] == selectedNode {
			// Truncate at the end and keep selected node in the first element.
			nodes = nodes[0 : lenUnsuitable-1]
		} else {
			// Truncate at the front, it's not the selected node.
			nodes = nodes[1:lenUnsuitable]
		}
	}
	return nodes
}

type claimAllocations []*ClaimAllocation

// MarshalLog replaces the pointers with the actual structs because
// we care about the content, not the pointer values.
func (claims claimAllocations) MarshalLog() interface{} {
	content := make([]ClaimAllocation, 0, len(claims))
	for _, claim := range claims {
		content = append(content, *claim)
	}
	return content
}

var _ logr.Marshaler = claimAllocations{}

// findClaim returns the index of the specified pod claim, -1 if not found.
func findClaim(claims []resourceapi.ResourceClaimSchedulingStatus, podClaimName string) int {
	for i := range claims {
		if claims[i].Name == podClaimName {
			return i
		}
	}
	return -1
}

// hasString checks for a string in a slice.
func hasString(strings []string, str string) bool {
	for _, s := range strings {
		if s == str {
			return true
		}
	}
	return false
}

// stringsDiffer does a strict comparison of two string arrays, order of entries matters.
func stringsDiffer(a, b []string) bool {
	if len(a) != len(b) {
		return true
	}
	for i := range a {
		if a[i] != b[i] {
			return true
		}
	}
	return false
}

// hasFinalizer checks if the claim has the finalizer of the driver.
func (ctrl *controller) hasFinalizer(claim *resourceapi.ResourceClaim) bool {
	for _, finalizer := range claim.Finalizers {
		if finalizer == ctrl.finalizer {
			return true
		}
	}
	return false
}

// removeFinalizer creates a new slice without the finalizer of the driver.
func (ctrl *controller) removeFinalizer(in []string) []string {
	out := make([]string, 0, len(in))
	for _, finalizer := range in {
		if finalizer != ctrl.finalizer {
			out = append(out, finalizer)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// prettyPrint formats arbitrary objects as JSON or, if that fails, with Sprintf.
func prettyPrint(obj interface{}) string {
	buffer, err := json.Marshal(obj)
	if err != nil {
		return fmt.Sprintf("%s", obj)
	}
	return string(buffer)
}
