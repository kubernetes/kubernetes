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
	"sync"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/typed/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/meta"
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

	if err := crc.setInitialCRDCondition(ctx, toBeProcessedSVM); err != nil {
		return err
	}
	if err := crc.updateCRDStoredVersions(ctx, toBeProcessedSVM); err != nil {
		return err
	}
	return nil
}

func (crc *CustomResourceController) setInitialCRDCondition(ctx context.Context, toBeProcessedSVM *svmv1beta1.StorageVersionMigration) error {
	migratingCond := meta.FindStatusCondition(toBeProcessedSVM.Status.Conditions, string(svmv1beta1.MigrationRunning))
	crdCond := meta.FindStatusCondition(toBeProcessedSVM.Status.Conditions, string(svmv1beta1.AssociatedCRD))
	if migratingCond == nil && crdCond == nil {
		crd, err := crc.crdForGroupResource(ctx, toBeProcessedSVM.Spec.Resource)
		if err != nil {
			return err
		}
		if crd == nil {
			toBeProcessedSVM.Status.Conditions = append(toBeProcessedSVM.Status.Conditions,
				metav1.Condition{
					Type:               string(svmv1beta1.AssociatedCRD),
					Status:             metav1.ConditionFalse,
					LastTransitionTime: metav1.Now(),
					Reason:             "NoCRD",
					Message:            "There is no associated crd for this migration",
				})
		} else {
			toBeProcessedSVM.Status.Conditions = append(toBeProcessedSVM.Status.Conditions,
				metav1.Condition{
					Type:               string(svmv1beta1.AssociatedCRD),
					Status:             metav1.ConditionTrue,
					LastTransitionTime: metav1.Now(),
					Reason:             "AssociatedCRD",
					Message:            "There is an associated crd for this migration",
					ObservedGeneration: crd.Generation,
				})
		}
		_, err = crc.kubeClient.StoragemigrationV1beta1().
			StorageVersionMigrations().
			UpdateStatus(
				ctx,
				toBeProcessedSVM,
				metav1.UpdateOptions{},
			)
		if err != nil {
			return err
		}
	}
	return nil
}

func (crc *CustomResourceController) updateCRDStoredVersions(ctx context.Context, toBeProcessedSVM *svmv1beta1.StorageVersionMigration) error {
	migratingSuccessfulCond := meta.FindStatusCondition(toBeProcessedSVM.Status.Conditions, string(svmv1beta1.MigrationSucceeded))
	crdCond := meta.FindStatusCondition(toBeProcessedSVM.Status.Conditions, string(svmv1beta1.AssociatedCRD))
	if migratingSuccessfulCond != nil && migratingSuccessfulCond.Status == metav1.ConditionTrue {
		crd, err := crc.crdForGroupResource(ctx, toBeProcessedSVM.Spec.Resource)
		if err != nil {
			return err
		}
		if crd == nil {
			return nil
		}
		if crdCond != nil {
			obsGen := crdCond.ObservedGeneration
			if err := crc.finishCRDMigration(ctx, crd, obsGen); err != nil {
				return err
			}
		}
	}
	return nil
}

func (crc *CustomResourceController) finishCRDMigration(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition, obsGen int64) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		crd, err := crc.crdClient.Get(ctx, crd.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if crd.Generation != obsGen {
			return nil
		}
		// Wipe out all stored versions that we have migrated off of if nothing else
		// has transpired during the migration.
		var storedVersion string
		for _, version := range crd.Spec.Versions {
			if version.Storage {
				storedVersion = version.Name
			}
		}
		crd.Status.StoredVersions = []string{storedVersion}
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
