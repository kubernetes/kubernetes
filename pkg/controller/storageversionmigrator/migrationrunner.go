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
	applyconfigurationapiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/client/applyconfiguration/apiextensions/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/typed/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	svmv1beta1 "k8s.io/api/storagemigration/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	svminformers "k8s.io/client-go/informers/storagemigration/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	svmlisters "k8s.io/client-go/listers/storagemigration/v1beta1"
)

const MigrationRunnerControllerName string = "migration-runner-controller"

// MigrationRunnerController is responsible for managing the lifecycle of
// StorageVersionMigration resources. It will ensure that only a single StorageVersionMigration
// will be active at a time for a given GroupResource, by adding the MigrationRunning condition
// to the SVMs that are active. Only these SVMs will be processed by the downstream migration controllers.
//
// It adds the StorageMigrating condition to the CRD and is in charge of removing old
// stored versions from the CRD status.
type MigrationRunnerController struct {
	svmListers svmlisters.StorageVersionMigrationLister
	svmSynced  cache.InformerSynced
	queue      workqueue.TypedRateLimitingInterface[metav1.GroupResource]
	kubeClient clientset.Interface
	crdClient  apiextensionsclient.CustomResourceDefinitionInterface
}

func NewCustomResourceController(
	ctx context.Context,
	kubeClient clientset.Interface,
	svmInformer svminformers.StorageVersionMigrationInformer,
	crdClient apiextensionsclient.CustomResourceDefinitionInterface,
) *MigrationRunnerController {
	logger := klog.FromContext(ctx)

	crController := &MigrationRunnerController{
		kubeClient: kubeClient,
		svmListers: svmInformer.Lister(),
		svmSynced:  svmInformer.Informer().HasSynced,
		crdClient:  crdClient,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[metav1.GroupResource](),
			workqueue.TypedRateLimitingQueueConfig[metav1.GroupResource]{Name: ResourceVersionControllerName},
		),
	}

	_, _ = svmInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			crController.addSVM(logger, obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			crController.updateSVM(logger, oldObj, newObj)
		},
		DeleteFunc: func(obj interface{}) {
			crController.deleteSVM(logger, obj)
		},
	})

	return crController
}

func (crc *MigrationRunnerController) addSVM(logger klog.Logger, obj interface{}) {
	svm := obj.(*svmv1beta1.StorageVersionMigration)
	logger.V(4).Info("Adding", "svm", klog.KObj(svm))
	crc.enqueue(svm)
}

func (crc *MigrationRunnerController) updateSVM(logger klog.Logger, oldObj, newObj interface{}) {
	oldSVM := oldObj.(*svmv1beta1.StorageVersionMigration)
	newSVM := newObj.(*svmv1beta1.StorageVersionMigration)
	logger.V(4).Info("Updating", "svm", klog.KObj(oldSVM))
	crc.enqueue(newSVM)
}

func (crc *MigrationRunnerController) deleteSVM(logger klog.Logger, obj interface{}) {
	svm := obj.(*svmv1beta1.StorageVersionMigration)
	logger.V(4).Info("Deleting", "svm", klog.KObj(svm))
	crc.enqueue(svm)
}

func (crc *MigrationRunnerController) enqueue(svm *svmv1beta1.StorageVersionMigration) {
	crc.queue.Add(svm.Spec.Resource)
}

func (crc *MigrationRunnerController) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()

	logger := klog.FromContext(ctx)
	logger.Info("Starting", "controller", MigrationRunnerControllerName)

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down", "controller", MigrationRunnerControllerName)
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

func (crc *MigrationRunnerController) worker(ctx context.Context) {
	for crc.processNext(ctx) {
	}
}

func (crc *MigrationRunnerController) processNext(ctx context.Context) bool {
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

func (crc *MigrationRunnerController) sync(ctx context.Context, resource metav1.GroupResource) error {
	// Get all SVMs for this resource and trigger re-sync
	svms, err := crc.getSVMsForResource(resource)
	if err != nil {
		return err
	}

	if err := crc.manageAdmission(ctx, svms); err != nil {
		return err
	}
	return nil
}

func (crc *MigrationRunnerController) manageAdmission(ctx context.Context, allSVMs []*svmv1beta1.StorageVersionMigration) error {
	nonTerminalSVMs := make([]*svmv1beta1.StorageVersionMigration, 0, len(allSVMs))
	// Clean up all terminal SVMs first, removing them from the list of SVMs to process.
	for _, svm := range allSVMs {
		if isTerminal(svm) {
			if err := crc.cleanupAdmission(ctx, svm); err != nil {
				return err
			}
		} else {
			nonTerminalSVMs = append(nonTerminalSVMs, svm)
		}
	}

	// Check if any other SVM is active
	for _, other := range nonTerminalSVMs {
		if meta.IsStatusConditionTrue(other.Status.Conditions, string(svmv1beta1.MigrationRunning)) {
			return nil
		}
	}

	sort.Slice(nonTerminalSVMs, func(i, j int) bool {
		switch nonTerminalSVMs[i].CreationTimestamp.Compare(nonTerminalSVMs[j].CreationTimestamp.Time) {
		case -1:
			return true
		case 1:
			return false
		case 0:
			return nonTerminalSVMs[i].Name < nonTerminalSVMs[j].Name
		}
		return false
	})

	if len(nonTerminalSVMs) > 0 {
		return crc.markAsActive(ctx, nonTerminalSVMs[0])
	}

	return nil
}

func (crc *MigrationRunnerController) cleanupAdmission(ctx context.Context, svm *svmv1beta1.StorageVersionMigration) error {
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

	return nil
}

func (crc *MigrationRunnerController) markAsActive(ctx context.Context, svm *svmv1beta1.StorageVersionMigration) error {
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
			applyConfig := applyconfigurationapiextensionsv1.CustomResourceDefinition(crd.Name).
				WithStatus(applyconfigurationapiextensionsv1.CustomResourceDefinitionStatus().
					WithConditions(applyconfigurationapiextensionsv1.CustomResourceDefinitionCondition().
						WithType(apiextensionsv1.StorageMigrating).
						WithStatus(apiextensionsv1.ConditionTrue).
						WithLastTransitionTime(metav1.Now()).
						WithReason("MigrationRunning").
						WithMessage(fmt.Sprintf("Migration %s is running", svm.Name)).
						WithObservedGeneration(crd.Generation)))

			_, err = crc.crdClient.ApplyStatus(ctx, applyConfig, metav1.ApplyOptions{FieldManager: MigrationRunnerControllerName, Force: true})
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

func (crc *MigrationRunnerController) getSVMsForResource(resource metav1.GroupResource) ([]*svmv1beta1.StorageVersionMigration, error) {
	svms, err := crc.svmListers.List(labels.Everything())
	if err != nil {
		return nil, err
	}
	var sameResourceSVMs []*svmv1beta1.StorageVersionMigration
	for _, svm := range svms {
		if svm.Spec.Resource == resource {
			sameResourceSVMs = append(sameResourceSVMs, svm.DeepCopy())
		}
	}
	return sameResourceSVMs, nil
}

func isTerminal(svm *svmv1beta1.StorageVersionMigration) bool {
	return meta.IsStatusConditionTrue(svm.Status.Conditions, string(svmv1beta1.MigrationSucceeded)) ||
		meta.IsStatusConditionTrue(svm.Status.Conditions, string(svmv1beta1.MigrationFailed))
}

func (crc *MigrationRunnerController) cleanupCRD(ctx context.Context, toBeProcessedSVM *svmv1beta1.StorageVersionMigration) error {
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

func (crc *MigrationRunnerController) finishCRDMigration(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition, success bool) error {
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

		var reason, msg string
		if success {
			reason = "MigrationSucceeded"
			msg = "The migration has succeeded"
			if canUpdateStoredVersions {
				msg += " and the stored versions have been updated"
			} else {
				msg += " but the stored versions have not been updated due to a generation mismatch"
			}
		} else {
			reason = "MigrationFailed"
			msg = "The migration has failed"
		}

		applyConfig := applyconfigurationapiextensionsv1.CustomResourceDefinition(crd.Name).
			WithStatus(applyconfigurationapiextensionsv1.CustomResourceDefinitionStatus().
				WithConditions(applyconfigurationapiextensionsv1.CustomResourceDefinitionCondition().
					WithType(apiextensionsv1.StorageMigrating).
					WithStatus(apiextensionsv1.ConditionFalse).
					WithLastTransitionTime(metav1.Now()).
					WithReason(reason).
					WithMessage(msg).
					WithObservedGeneration(crd.Generation)))

		if canUpdateStoredVersions {
			// Wipe out all stored versions that we have migrated off of if nothing else
			// has transpired while the migration was in progress.
			var storedVersion string
			for _, version := range crd.Spec.Versions {
				if version.Storage {
					storedVersion = version.Name
				}
			}
			applyConfig.Status.WithStoredVersions(storedVersion)
		}

		_, err = crc.crdClient.ApplyStatus(ctx, applyConfig, metav1.ApplyOptions{FieldManager: MigrationRunnerControllerName, Force: true})
		return err
	})
}

func (crc *MigrationRunnerController) crdForGroupResource(ctx context.Context, gr metav1.GroupResource) (*apiextensionsv1.CustomResourceDefinition, error) {
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
