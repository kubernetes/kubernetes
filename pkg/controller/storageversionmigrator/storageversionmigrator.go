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
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"

	svmv1alpha1 "k8s.io/api/storagemigration/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	svminformers "k8s.io/client-go/informers/storagemigration/v1alpha1"
	svmlisters "k8s.io/client-go/listers/storagemigration/v1alpha1"
)

const (
	workers                      = 5
	migrationSuccessStatusReason = "storage_version_migration_succeeded"
	migrationRunningStatusReason = "storage_version_migration_in_progress"
	migrationFailedStatusReason  = "storage_version_migration_failed"
)

type SVMController struct {
	controllerName         string
	kubeClient             kubernetes.Interface
	dynamicClient          *dynamic.DynamicClient
	svmListers             svmlisters.StorageVersionMigrationLister
	svmSynced              cache.InformerSynced
	queue                  workqueue.RateLimitingInterface
	restMapper             meta.RESTMapper
	dependencyGraphBuilder *garbagecollector.GraphBuilder
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

	svmController := &SVMController{
		kubeClient:             kubeClient,
		dynamicClient:          dynamicClient,
		controllerName:         controllerName,
		svmListers:             svmInformer.Lister(),
		svmSynced:              svmInformer.Informer().HasSynced,
		restMapper:             mapper,
		dependencyGraphBuilder: dependencyGraphBuilder,
		queue:                  workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), controllerName),
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
	svm := obj.(*svmv1alpha1.StorageVersionMigration)
	logger.V(4).Info("Adding", "svm", klog.KObj(svm))
	svmc.enqueue(svm)
}

func (svmc *SVMController) updateSVM(logger klog.Logger, oldObj, newObj interface{}) {
	oldSVM := oldObj.(*svmv1alpha1.StorageVersionMigration)
	newSVM := newObj.(*svmv1alpha1.StorageVersionMigration)
	logger.V(4).Info("Updating", "svm", klog.KObj(oldSVM))
	svmc.enqueue(newSVM)
}

func (svmc *SVMController) enqueue(svm *svmv1alpha1.StorageVersionMigration) {
	key, err := controller.KeyFunc(svm)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %w", svm, err))
		return
	}

	svmc.queue.Add(key)
}

func (svmc *SVMController) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()
	defer svmc.queue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting", "controller", svmc.controllerName)
	defer logger.Info("Shutting down", "controller", svmc.controllerName)

	if !cache.WaitForNamedCacheSync(svmc.controllerName, ctx.Done(), svmc.svmSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, svmc.worker, time.Second)
	}

	<-ctx.Done()
}

func (svmc *SVMController) worker(ctx context.Context) {
	for svmc.processNext(ctx) {
	}
}

func (svmc *SVMController) processNext(ctx context.Context) bool {
	svmKey, quit := svmc.queue.Get()
	if quit {
		return false
	}
	defer svmc.queue.Done(svmKey)

	key := svmKey.(string)
	err := svmc.sync(ctx, key)
	if err == nil {
		svmc.queue.Forget(key)
		return true
	}

	klog.FromContext(ctx).V(2).Info("Error syncing SVM resource, retrying", "svm", key, "err", err)
	svmc.queue.AddRateLimited(key)

	return true
}

func (svmc *SVMController) sync(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	startTime := time.Now()

	if svmc.dependencyGraphBuilder == nil {
		logger.V(4).Info("dependency graph builder is not set. we will skip migration")
		return nil
	}

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

	if IsConditionTrue(toBeProcessedSVM, svmv1alpha1.MigrationSucceeded) || IsConditionTrue(toBeProcessedSVM, svmv1alpha1.MigrationFailed) {
		logger.V(4).Info("Migration has already succeeded or failed previously, skipping", "svm", name)
		return nil
	}

	if len(toBeProcessedSVM.Status.ResourceVersion) == 0 {
		logger.V(4).Info("The latest resource version is empty. We will attempt to migrate once the resource version is available.")
		return nil
	}
	gvr := getGVRFromResource(toBeProcessedSVM)

	// wait for the dependency graph to be synchronized before migration to avoid a nil pointer.
	// this prevents deadlocks in scenarios where available resources have changed, and one of the informer caches will never fill.
	if !cache.WaitForNamedCacheSync(svmc.dependencyGraphBuilder.Name(), ctx.Done(), func() bool {
		return svmc.dependencyGraphBuilder.IsResourceSynced(gvr)
	}) {
		return fmt.Errorf("dependency graph is not synced, requeuing to attempt again")
	}

	gcListResourceVersion, err := convertResourceVersionToInt(svmc.dependencyGraphBuilder.GetMonitorController(gvr).LastSyncResourceVersion())
	if err != nil {
		return err
	}
	listResourceVersion, err := convertResourceVersionToInt(toBeProcessedSVM.Status.ResourceVersion)
	if err != nil {
		return err
	}

	if gcListResourceVersion < listResourceVersion {
		return fmt.Errorf("GC cache is not up to date, requeuing to attempt again. gcListResourceVersion: %d, listResourceVersion: %d", gcListResourceVersion, listResourceVersion)
	}

	toBeProcessedSVM, err = svmc.kubeClient.StoragemigrationV1alpha1().
		StorageVersionMigrations().
		UpdateStatus(
			ctx,
			setStatusConditions(toBeProcessedSVM, svmv1alpha1.MigrationRunning, migrationRunningStatusReason),
			metav1.UpdateOptions{},
		)
	if err != nil {
		return err
	}

	gvk, err := svmc.restMapper.KindFor(gvr)
	if err != nil {
		return err
	}
	typeMeta := metav1.TypeMeta{}
	typeMeta.APIVersion, typeMeta.Kind = gvk.ToAPIVersionAndKind()
	data, err := json.Marshal(typeMeta)
	if err != nil {
		return err
	}

	// ToDo: implement a mechanism to resume migration from the last migrated resource in case of a failure
	// process storage migration
	for _, gvrKey := range svmc.dependencyGraphBuilder.GetMonitorStore(gvr).ListKeys() {
		namespace, name, err := cache.SplitMetaNamespaceKey(gvrKey)
		if err != nil {
			return err
		}

		_, err = svmc.dynamicClient.Resource(gvr).
			Namespace(namespace).
			Patch(ctx,
				name,
				types.ApplyPatchType,
				data,
				metav1.PatchOptions{
					FieldManager: svmc.controllerName,
				},
			)
		if err != nil {
			// in case of NotFound or Conflict, we can stop processing migration for that resource
			if apierrors.IsNotFound(err) || apierrors.IsConflict(err) {
				continue
			}

			_, err = svmc.kubeClient.StoragemigrationV1alpha1().
				StorageVersionMigrations().
				UpdateStatus(
					ctx,
					setStatusConditions(toBeProcessedSVM, svmv1alpha1.MigrationFailed, migrationFailedStatusReason),
					metav1.UpdateOptions{},
				)
			if err != nil {
				return err
			}
			logger.V(4).Error(err, "Failed to migrate the resource", "name", gvrKey, "gvr", gvr.String(), "reason", apierrors.ReasonForError(err))

			return nil
			//Todo: add retry for scenarios where API server returns rate limiting error
		}
		logger.V(4).Info("Successfully migrated the resource", "name", gvrKey, "gvr", gvr.String())
	}

	_, err = svmc.kubeClient.StoragemigrationV1alpha1().
		StorageVersionMigrations().
		UpdateStatus(
			ctx,
			setStatusConditions(toBeProcessedSVM, svmv1alpha1.MigrationSucceeded, migrationSuccessStatusReason),
			metav1.UpdateOptions{},
		)
	if err != nil {
		return err
	}

	logger.V(4).Info("Finished syncing svm resource", "key", key, "gvr", gvr.String(), "elapsed", time.Since(startTime))
	return nil
}
