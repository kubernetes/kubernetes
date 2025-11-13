/*
Copyright 2025 The Kubernetes Authors.

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
	"fmt"
	"sort"
	"sync"
	"time"

	"k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/typed/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"

	svmv1beta1 "k8s.io/api/storagemigration/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	svminformers "k8s.io/client-go/informers/storagemigration/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	svmlisters "k8s.io/client-go/listers/storagemigration/v1beta1"
)

const CustomResourceControllerName string = "custom-resource-controller"

// CustomResourceController adds the CRD generation obtained to the SVM
// conditions if it exists. It will also update the
type CustomResourceController struct {
	svmListers svmlisters.StorageVersionMigrationLister
	svmSynced  cache.InformerSynced
	queue      workqueue.TypedRateLimitingInterface[string]
	kubeClient clientset.Interface
	crdClient  apiextensionsclient.CustomResourceDefinitionInterface
}

func NewCustomResourceController(
	ctx context.Context,
	kubeClient clientset.Interface,
	svmInformer svminformers.StorageVersionMigrationInformer,
	crdClient apiextensionsclient.CustomResourceDefinitionInterface,
) *CustomResourceController {
	logger := klog.FromContext(ctx)

	crController := &CustomResourceController{
		kubeClient: kubeClient,
		svmListers: svmInformer.Lister(),
		svmSynced:  svmInformer.Informer().HasSynced,
		crdClient:  crdClient,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: ResourceVersionControllerName},
		),
	}

	_, _ = svmInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			crController.addSVM(logger, obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			crController.updateSVM(logger, oldObj, newObj)
		},
	})

	return crController
}

func (crc *CustomResourceController) addSVM(logger klog.Logger, obj interface{}) {
	svm := obj.(*svmv1beta1.StorageVersionMigration)
	logger.V(4).Info("Adding", "svm", klog.KObj(svm))
	crc.enqueue(svm)
}

func (crc *CustomResourceController) updateSVM(logger klog.Logger, oldObj, newObj interface{}) {
	oldSVM := oldObj.(*svmv1beta1.StorageVersionMigration)
	newSVM := newObj.(*svmv1beta1.StorageVersionMigration)
	logger.V(4).Info("Updating", "svm", klog.KObj(oldSVM))
	crc.enqueue(newSVM)
}

func (crc *CustomResourceController) enqueue(svm *svmv1beta1.StorageVersionMigration) {
	key, err := controller.KeyFunc(svm)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %w", svm, err))
		return
	}

	crc.queue.Add(key)
}

func (crc *CustomResourceController) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()

	logger := klog.FromContext(ctx)
	logger.Info("Starting", "controller", CustomResourceControllerName)

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down", "controller", CustomResourceControllerName)
		crc.queue.ShutDown()
		wg.Wait()
	}()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, crc.svmSynced) {
		return
	}

	wg.Go(func() {
		wait.UntilWithContext(ctx, crc.worker, time.Second)
	})
	<-ctx.Done()
}

func (crc *CustomResourceController) worker(ctx context.Context) {
	for crc.processNext(ctx) {
	}
}

func (crc *CustomResourceController) processNext(ctx context.Context) bool {
	key, quit := crc.queue.Get()
	if quit {
		return false
	}
	defer crc.queue.Done(key)

	err := crc.sync(ctx, key)
	if err == nil {
		crc.queue.Forget(key)
		return true
	}

	klog.FromContext(ctx).V(2).Info("Error syncing SVM resource, retrying", "svm", key, "err", err)
	crc.queue.AddRateLimited(key)

	return true
}

func (crc *CustomResourceController) sync(ctx context.Context, key string) error {
	// SVM is a cluster scoped resource so we don't care about the namespace
	_, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	svm, err := crc.svmListers.Get(name)
	if apierrors.IsNotFound(err) {
		// no work to do, don't fail and requeue
		return nil
	}
	if err != nil {
		return err
	}
	// working with copy to avoid race condition between this and migration controller
	toBeProcessedSVM := svm.DeepCopy()

	if err := crc.manageAdmission(ctx, toBeProcessedSVM); err != nil {
		return err
	}
	return nil
}

func (crc *CustomResourceController) manageAdmission(ctx context.Context, svm *svmv1beta1.StorageVersionMigration) error {
	if isTerminal(svm) {
		return crc.cleanupAdmission(ctx, svm)
	}

	if meta.IsStatusConditionTrue(svm.Status.Conditions, string(svmv1beta1.MigrationRunning)) {
		return nil
	}

	allSVMs, err := crc.getSVMsForResource(svm.Spec.Resource)
	if err != nil {
		return err
	}

	// Check if any other SVM is active
	for _, other := range allSVMs {
		if meta.IsStatusConditionTrue(other.Status.Conditions, string(svmv1beta1.MigrationRunning)) {
			return nil
		}
	}

	// Filter potentially eligible SVMs (non-terminal)
	var candidates []*svmv1beta1.StorageVersionMigration
	for _, other := range allSVMs {
		if !isTerminal(other) {
			candidates = append(candidates, other)
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		switch candidates[i].CreationTimestamp.Compare(candidates[j].CreationTimestamp.Time) {
		case -1:
			return true
		case 1:
			return false
		case 0:
			return candidates[i].Name < candidates[j].Name
		}
		return false
	})

	if len(candidates) > 0 && candidates[0].Name == svm.Name {
		return crc.markAsActive(ctx, svm)
	}

	return nil
}

func (crc *CustomResourceController) cleanupAdmission(ctx context.Context, svm *svmv1beta1.StorageVersionMigration) error {
	needsUpdate := false
	migratingCond := meta.FindStatusCondition(svm.Status.Conditions, string(svmv1beta1.MigrationRunning))
	if migratingCond != nil {
		meta.RemoveStatusCondition(&svm.Status.Conditions, string(svmv1beta1.MigrationRunning))
		needsUpdate = true
	}

	if needsUpdate {
		if err := crc.cleanupCRD(ctx, svm); err != nil {
			return err
		}

		_, err := crc.kubeClient.StoragemigrationV1beta1().
			StorageVersionMigrations().
			UpdateStatus(
				ctx,
				svm,
				metav1.UpdateOptions{},
			)
		if err != nil {
			return err
		}
	}

	// Trigger other pending SVMs for this resource
	allSVMs, err := crc.getSVMsForResource(svm.Spec.Resource)
	if err != nil {
		return err
	}
	for _, other := range allSVMs {
		if other.Name == svm.Name {
			continue
		}
		crc.enqueue(other)
	}

	return nil
}

func (crc *CustomResourceController) markAsActive(ctx context.Context, svm *svmv1beta1.StorageVersionMigration) error {
	// Mark CRD as undergoing migration.
	crd, err := crc.crdForGroupResource(ctx, svm.Spec.Resource)
	if err != nil {
		return err
	}
	if crd != nil {
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			crd, err := crc.crdClient.Get(ctx, crd.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			apihelpers.SetCRDCondition(crd, apiextensionsv1.CustomResourceDefinitionCondition{
				Type:               apiextensionsv1.StorageMigrating,
				Status:             apiextensionsv1.ConditionTrue,
				LastTransitionTime: metav1.Now(),
				Reason:             "MigrationRunning",
				Message:            fmt.Sprintf("Migration %s is running", svm.Name),
				ObservedGeneration: crd.Generation,
			})
			_, err = crc.crdClient.UpdateStatus(ctx, crd, metav1.UpdateOptions{})
			return err
		})
		if err != nil {
			return err
		}
	}

	svm = setStatusConditions(svm, svmv1beta1.MigrationRunning, "MigrationRunning", "The migration is running")
	_, err = crc.kubeClient.StoragemigrationV1beta1().
		StorageVersionMigrations().
		UpdateStatus(
			ctx,
			svm,
			metav1.UpdateOptions{},
		)
	if err != nil {
		return err
	}

	return nil
}

func (crc *CustomResourceController) getSVMsForResource(resource metav1.GroupResource) ([]*svmv1beta1.StorageVersionMigration, error) {
	svms, err := crc.svmListers.List(labels.Everything())
	if err != nil {
		return nil, err
	}
	var sameResourceSVMs []*svmv1beta1.StorageVersionMigration
	for _, svm := range svms {
		if svm.Spec.Resource == resource {
			sameResourceSVMs = append(sameResourceSVMs, svm)
		}
	}
	return sameResourceSVMs, nil
}

func isTerminal(svm *svmv1beta1.StorageVersionMigration) bool {
	return meta.IsStatusConditionTrue(svm.Status.Conditions, string(svmv1beta1.MigrationSucceeded)) ||
		meta.IsStatusConditionTrue(svm.Status.Conditions, string(svmv1beta1.MigrationFailed))
}

func (crc *CustomResourceController) cleanupCRD(ctx context.Context, toBeProcessedSVM *svmv1beta1.StorageVersionMigration) error {
	migratingSuccessfulCond := meta.FindStatusCondition(toBeProcessedSVM.Status.Conditions, string(svmv1beta1.MigrationSucceeded))
	success := migratingSuccessfulCond != nil && migratingSuccessfulCond.Status == metav1.ConditionTrue
	crd, err := crc.crdForGroupResource(ctx, toBeProcessedSVM.Spec.Resource)
	if err != nil {
		return err
	}
	if crd == nil {
		return nil
	}
	if err := crc.finishCRDMigration(ctx, crd, success); err != nil {
		return err
	}

	return nil
}

func (crc *CustomResourceController) finishCRDMigration(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition, success bool) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		crd, err := crc.crdClient.Get(ctx, crd.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		migratingCond := apihelpers.FindCRDCondition(crd, apiextensionsv1.StorageMigrating)
		var canUpdateStoredVersions bool
		if migratingCond != nil && migratingCond.ObservedGeneration == crd.Generation {
			canUpdateStoredVersions = true
		}

		// Update the CRD condition
		if success {
			msg := "The migration has succeeded"
			if canUpdateStoredVersions {
				msg += " and the stored versions have been updated"
			} else {
				msg += " but the stored versions have not been updated due to a generation mismatch"
			}
			apihelpers.SetCRDCondition(crd, apiextensionsv1.CustomResourceDefinitionCondition{
				Type:               apiextensionsv1.StorageMigrating,
				Status:             apiextensionsv1.ConditionFalse,
				LastTransitionTime: metav1.Now(),
				Reason:             "MigrationSucceeded",
				Message:            msg,
				ObservedGeneration: crd.Generation,
			})
		} else {
			apihelpers.SetCRDCondition(crd, apiextensionsv1.CustomResourceDefinitionCondition{
				Type:               apiextensionsv1.StorageMigrating,
				Status:             apiextensionsv1.ConditionTrue,
				LastTransitionTime: metav1.Now(),
				Reason:             "MigrationFailed",
				Message:            "The migration has failed",
				ObservedGeneration: crd.Generation,
			})
		}

		if canUpdateStoredVersions {
			// Wipe out all stored versions that we have migrated off of if nothing else
			// has transpired while the migration was in progress.
			var storedVersion string
			for _, version := range crd.Spec.Versions {
				if version.Storage {
					storedVersion = version.Name
				}
			}
			crd.Status.StoredVersions = []string{storedVersion}
		}
		_, err = crc.crdClient.UpdateStatus(ctx, crd, metav1.UpdateOptions{})
		return err
	})
}

func (crc *CustomResourceController) crdForGroupResource(ctx context.Context, gr metav1.GroupResource) (*apiextensionsv1.CustomResourceDefinition, error) {
	crdName := fmt.Sprintf("%s.%s", gr.Resource, gr.Group)
	crd, err := crc.crdClient.Get(ctx, crdName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return crd, nil
}
