/*
Copyright 2024 The Kubernetes Authors.

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

package storageversionmigrator

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"

	svmv1beta1 "k8s.io/api/storagemigration/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	svminformers "k8s.io/client-go/informers/storagemigration/v1beta1"
	svmlisters "k8s.io/client-go/listers/storagemigration/v1beta1"
)

const (
	workers                      = 5
	migrationSuccessStatusReason = "StorageVersionMigrationSucceeded"
	migrationRunningStatusReason = "StorageVersionMigrationInProgress"
	migrationFailedStatusReason  = "StorageVersionMigrationFailed"
)

type graphBuilder interface {
	GetMonitor(ctx context.Context, gvr schema.GroupVersionResource) (*garbagecollector.Monitor, bool, error)
}

type SVMController struct {
	controllerName         string
	kubeClient             kubernetes.Interface
	dynamicClient          dynamic.Interface
	svmListers             svmlisters.StorageVersionMigrationLister
	svmSynced              cache.InformerSynced
	queue                  workqueue.TypedRateLimitingInterface[string]
	rateLimiter            workqueue.TypedRateLimiter[string]
	restMapper             meta.RESTMapper
	dependencyGraphBuilder graphBuilder
}

func NewSVMController(
	ctx context.Context,
	kubeClient kubernetes.Interface,
	dynamicClient *dynamic.DynamicClient,
	svmInformer svminformers.StorageVersionMigrationInformer,
	controllerName string,
	mapper meta.ResettableRESTMapper,
	dependencyGraphBuilder *garbagecollector.GraphBuilder,
) *SVMController {
	logger := klog.FromContext(ctx)
	rateLimiter := workqueue.DefaultTypedControllerRateLimiter[string]()

	svmController := &SVMController{
		kubeClient:             kubeClient,
		dynamicClient:          dynamicClient,
		controllerName:         controllerName,
		svmListers:             svmInformer.Lister(),
		svmSynced:              svmInformer.Informer().HasSynced,
		restMapper:             mapper,
		dependencyGraphBuilder: dependencyGraphBuilder,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			rateLimiter,
			workqueue.TypedRateLimitingQueueConfig[string]{Name: controllerName},
		),
		rateLimiter: rateLimiter,
	}

	_, _ = svmInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			svmController.addSVM(logger, obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			svmController.updateSVM(logger, oldObj, newObj)
		},
	})

	return svmController
}

func (svmc *SVMController) Name() string {
	return svmc.controllerName
}

func (svmc *SVMController) addSVM(logger klog.Logger, obj interface{}) {
	svm := obj.(*svmv1beta1.StorageVersionMigration)
	logger.V(4).Info("Adding", "svm", klog.KObj(svm))
	svmc.enqueue(logger, svm)
}

func (svmc *SVMController) updateSVM(logger klog.Logger, oldObj, newObj interface{}) {
	oldSVM := oldObj.(*svmv1beta1.StorageVersionMigration)
	newSVM := newObj.(*svmv1beta1.StorageVersionMigration)
	logger.V(4).Info("Updating", "svm", klog.KObj(oldSVM))
	svmc.enqueue(logger, newSVM)
}

func (svmc *SVMController) enqueue(logger klog.Logger, svm *svmv1beta1.StorageVersionMigration) {
	key, err := controller.KeyFunc(svm)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Couldn't get key for object", "svm", klog.KObj(svm))
		return
	}

	svmc.queue.Add(key)
}

func (svmc *SVMController) Run(ctx context.Context) {
	defer utilruntime.HandleCrashWithContext(ctx)

	logger := klog.FromContext(ctx)
	logger.Info("Starting", "controller", svmc.controllerName)

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down", "controller", svmc.controllerName)
		svmc.queue.ShutDown()
		wg.Wait()
	}()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, svmc.svmSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		wg.Go(func() {
			wait.UntilWithContext(ctx, svmc.worker, time.Second)
		})
	}
	<-ctx.Done()
}

func (svmc *SVMController) worker(ctx context.Context) {
	for svmc.processNext(ctx) {
	}
}

func (svmc *SVMController) processNext(ctx context.Context) bool {
	key, quit := svmc.queue.Get()
	if quit {
		return false
	}
	defer svmc.queue.Done(key)

	err := svmc.sync(ctx, key)
	if err == nil {
		svmc.queue.Forget(key)
		return true
	}

	klog.FromContext(ctx).V(2).Info("Error syncing SVM resource, retrying", "svm", key, "err", err)
	if suggestDelay, ok := apierrors.SuggestsClientDelay(err); ok {
		svmc.queue.AddAfter(key, max(time.Second*time.Duration(suggestDelay), svmc.rateLimiter.When(key)))
		return true
	}
	svmc.queue.AddRateLimited(key)

	return true
}

func (svmc *SVMController) sync(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	startTime := time.Now()

	// SVM is a cluster scoped resource so we don't care about the namespace
	_, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	svm, err := svmc.svmListers.Get(name)
	if apierrors.IsNotFound(err) {
		// no work to do, don't fail and requeue
		return nil
	}
	if err != nil {
		return err
	}
	// working with a copy to avoid race condition between this and resource version controller
	toBeProcessedSVM := svm.DeepCopy()

	if meta.IsStatusConditionTrue(toBeProcessedSVM.Status.Conditions, string(svmv1beta1.MigrationSucceeded)) ||
		meta.IsStatusConditionTrue(toBeProcessedSVM.Status.Conditions, string(svmv1beta1.MigrationFailed)) {
		logger.V(4).Info("Migration has already succeeded or failed previously, skipping", "svm", name)
		return nil
	}

	if len(toBeProcessedSVM.Status.ResourceVersion) == 0 {
		logger.V(4).Info("The latest resource version is empty. We will attempt to migrate once the resource version is available.")
		return nil
	}
	gvr, exists, err := resourceFor(svmc.restMapper, toBeProcessedSVM.Spec.Resource)
	if err != nil {
		return err
	}
	if !exists {
		logger.V(4).Info("resource does not exist in our rest mapper", "gvr", gvr.String())
		if toBeProcessedSVM.CreationTimestamp.Add(time.Minute).After(time.Now()) {
			return fmt.Errorf("resource does not exist in rest mapper, requeuing to attempt again: %w", err)
		}
		return svmc.failMigration(ctx, toBeProcessedSVM, fmt.Errorf("unable to obtain GVR from discovery"))
	}

	// prevent unsynced monitor from blocking forever
	// use a short timeout so that we can fail quickly and possibly handle other migrations while this monitor gets ready.
	monCtx, monCtxCancel := context.WithTimeout(ctx, 10*time.Second)
	defer monCtxCancel()

	resourceMonitor, hasSynced, errMonitor := svmc.dependencyGraphBuilder.GetMonitor(monCtx, *gvr)
	if errMonitor != nil {
		return errMonitor
	}
	if !hasSynced {
		logger.V(4).Info("resource does not exist in GC", "gvr", gvr.String())

		// our mapper could be missing a recently created custom resource, so give it some time to catch up
		// we resync discovery every 30 seconds so twice that should be sufficient
		if toBeProcessedSVM.CreationTimestamp.Add(time.Minute).After(time.Now()) {
			return fmt.Errorf("resource does not exist in GC, requeuing to attempt again: %w", errMonitor)
		}
		return svmc.failMigration(ctx, toBeProcessedSVM, fmt.Errorf("resource does not exist in GC: %w", errMonitor))
	}

	gcListResourceVersion := resourceMonitor.Controller.LastSyncResourceVersion()
	listResourceVersion := toBeProcessedSVM.Status.ResourceVersion

	rvCmp, err := resourceversion.CompareResourceVersion(gcListResourceVersion, listResourceVersion)
	if err != nil {
		return svmc.failMigration(ctx, toBeProcessedSVM, fmt.Errorf("error comparing resource versions between GC and SVM resource: %w", err))
	}
	if rvCmp == -1 {
		return fmt.Errorf("GC cache is not up to date, requeuing to attempt again. gcListResourceVersion: %s, listResourceVersion: %s", gcListResourceVersion, listResourceVersion)
	}

	toBeProcessedSVM, err = svmc.kubeClient.StoragemigrationV1beta1().
		StorageVersionMigrations().
		UpdateStatus(
			ctx,
			setStatusConditions(toBeProcessedSVM, svmv1beta1.MigrationRunning, migrationRunningStatusReason, ""),
			metav1.UpdateOptions{},
		)
	if err != nil {
		return err
	}

	err, failedMigration := svmc.runMigration(ctx, logger, *gvr, resourceMonitor, toBeProcessedSVM, listResourceVersion)
	if err != nil {
		return err
	}
	if failedMigration {
		return nil
	}

	_, err = svmc.kubeClient.StoragemigrationV1beta1().
		StorageVersionMigrations().
		UpdateStatus(
			ctx,
			setStatusConditions(toBeProcessedSVM, svmv1beta1.MigrationSucceeded, migrationSuccessStatusReason, ""),
			metav1.UpdateOptions{},
		)
	if err != nil {
		return err
	}

	logger.V(4).Info("Finished syncing svm resource", "key", key, "gr", toBeProcessedSVM.Spec.Resource.String(), "elapsed", time.Since(startTime))
	return nil
}

func (svmc *SVMController) runMigration(ctx context.Context, logger klog.Logger, gvr schema.GroupVersionResource, resourceMonitor *garbagecollector.Monitor, toBeProcessedSVM *svmv1beta1.StorageVersionMigration, listResourceVersion string) (err error, failed bool) {
	gvk, err := svmc.restMapper.KindFor(gvr)
	if err != nil {
		return svmc.failMigration(ctx, toBeProcessedSVM, err), true
	}
	for _, obj := range resourceMonitor.Store.List() {
		accessor, err := meta.Accessor(obj)
		if err != nil {
			return svmc.failMigration(ctx, toBeProcessedSVM, err), true
		}
		rvCmp, err := resourceversion.CompareResourceVersion(accessor.GetResourceVersion(), listResourceVersion)
		if err != nil {
			logger.Error(err, "Unable to compare the resource version of the resource", "namespace", accessor.GetNamespace(), "name", accessor.GetName(), "gvr", gvr.String(), "accessorRV", accessor.GetResourceVersion(), "listResourceVersion", listResourceVersion, "error", err.Error())
			return svmc.failMigration(ctx, toBeProcessedSVM, err), true
		}
		if rvCmp == 1 {
			logger.V(6).Info("Resource ignored due to resource version being greater than the SVM checkpoint", "namespace", accessor.GetNamespace(), "name", accessor.GetName(), "gvr", gvr.String(), "accessorRV", accessor.GetResourceVersion(), "listResourceVersion", listResourceVersion)
			continue
		}

		typeMeta := typeMetaUIDRV{}
		typeMeta.APIVersion, typeMeta.Kind = gvk.ToAPIVersionAndKind()
		// set UID so that when a resource gets deleted, we get an "uid mismatch"
		// conflict error instead of trying to create it.
		typeMeta.UID = accessor.GetUID()
		// set RV so that when a resources gets updated or deleted+recreated, we get an "object has been modified"
		// conflict error.  we do not actually need to do anything special for the updated case because if RV
		// was not set, it would just result in no-op request.  but for the deleted+recreated case, if RV is
		// not set but UID is set, we would get an immutable field validation error.  hence we must set both.
		typeMeta.ResourceVersion = accessor.GetResourceVersion()
		data, err := json.Marshal(typeMeta)
		if err != nil {
			return svmc.failMigration(ctx, toBeProcessedSVM, err), true
		}

		_, errPatch := svmc.dynamicClient.Resource(gvr).
			Namespace(accessor.GetNamespace()).
			Patch(ctx,
				accessor.GetName(),
				types.ApplyPatchType,
				data,
				metav1.PatchOptions{
					FieldManager: svmc.controllerName,
				},
			)

		// in case of conflict, we can stop processing migration for that resource because it has either been
		// - updated, meaning that migration has already been performed
		// - deleted, meaning that migration is not needed
		// - deleted and recreated, meaning that migration has already been performed
		if apierrors.IsConflict(errPatch) {
			logger.V(6).Info("Resource ignored due to conflict", "namespace", accessor.GetNamespace(), "name", accessor.GetName(), "gvr", gvr.String(), "err", errPatch)
			continue
		}

		// in case of retriable errors like server throttling, we can return an error since that will cause the migration to be reattempted.
		if isRetriableError(errPatch) {
			logger.V(6).Info("Resource patch failed due to an error that can be retried", "namespace", accessor.GetNamespace(), "name", accessor.GetName(), "gvr", gvr.String(), "err", errPatch)
			return errPatch, false
		}

		if errPatch != nil {
			logger.Error(errPatch, "Failed to migrate the resource", "namespace", accessor.GetNamespace(), "name", accessor.GetName(), "gvr", gvr.String(), "reason", apierrors.ReasonForError(errPatch))
			errStatus := svmc.failMigration(ctx, toBeProcessedSVM, errPatch)
			return errStatus, true
		}
		logger.V(4).Info("Successfully migrated the resource", "namespace", accessor.GetNamespace(), "name", accessor.GetName(), "gvr", gvr.String())
	}
	return nil, false
}

func isRetriableError(k8sError error) bool {
	return utilnet.IsConnectionReset(k8sError) ||
		utilnet.IsHTTP2ConnectionLost(k8sError) ||
		utilnet.IsProbableEOF(k8sError) ||
		apierrors.IsServerTimeout(k8sError) ||
		apierrors.IsTooManyRequests(k8sError) ||
		apierrors.IsServiceUnavailable(k8sError) ||
		apierrors.IsInternalError(k8sError) ||
		apierrors.IsTimeout(k8sError)
}

func (svmc *SVMController) failMigration(ctx context.Context, toBeProcessedSVM *svmv1beta1.StorageVersionMigration, err error) error {
	errMsg := fmt.Sprintf("migration encountered unhandled error: %s", err)

	_, errStatus := svmc.kubeClient.StoragemigrationV1beta1().
		StorageVersionMigrations().
		UpdateStatus(
			ctx,
			setStatusConditions(toBeProcessedSVM, svmv1beta1.MigrationFailed, migrationFailedStatusReason, errMsg),
			metav1.UpdateOptions{},
		)
	return errStatus
}

type typeMetaUIDRV struct {
	metav1.TypeMeta    `json:",inline"`
	objectMetaUIDandRV `json:"metadata,omitempty"`
}

type objectMetaUIDandRV struct {
	UID             types.UID `json:"uid,omitempty"`
	ResourceVersion string    `json:"resourceVersion,omitempty"`
}
