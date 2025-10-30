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
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/tools/cache"
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

const (
	// this name is guaranteed to be not present in the cluster as it not a valid namespace name
	fakeSVMNamespaceName          string = "@fake:svm_ns!"
	ResourceVersionControllerName string = "resource-version-controller"
)

var verbsRequiredForMigration = []string{"update", "patch", "list"}

// ResourceVersionController adds the resource version obtained from a randomly nonexistent namespace
// to the SVM status before the migration is initiated. This resource version is utilized for checking
// freshness of GC cache before the migration is initiated.
type ResourceVersionController struct {
	discoveryClient discovery.DiscoveryInterface
	metadataClient  metadata.Interface
	svmListers      svmlisters.StorageVersionMigrationLister
	svmSynced       cache.InformerSynced
	queue           workqueue.TypedRateLimitingInterface[string]
	kubeClient      clientset.Interface
	mapper          meta.RESTMapper
}

func NewResourceVersionController(
	ctx context.Context,
	kubeClient clientset.Interface,
	discoveryClient discovery.DiscoveryInterface,
	metadataClient metadata.Interface,
	svmInformer svminformers.StorageVersionMigrationInformer,
	mapper meta.RESTMapper,
) *ResourceVersionController {
	logger := klog.FromContext(ctx)

	rvController := &ResourceVersionController{
		kubeClient:      kubeClient,
		discoveryClient: discoveryClient,
		metadataClient:  metadataClient,
		svmListers:      svmInformer.Lister(),
		svmSynced:       svmInformer.Informer().HasSynced,
		mapper:          mapper,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: ResourceVersionControllerName},
		),
	}

	_, _ = svmInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			rvController.addSVM(logger, obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			rvController.updateSVM(logger, oldObj, newObj)
		},
	})

	return rvController
}

func (rv *ResourceVersionController) addSVM(logger klog.Logger, obj interface{}) {
	svm := obj.(*svmv1beta1.StorageVersionMigration)
	logger.V(4).Info("Adding", "svm", klog.KObj(svm))
	rv.enqueue(svm)
}

func (rv *ResourceVersionController) updateSVM(logger klog.Logger, oldObj, newObj interface{}) {
	oldSVM := oldObj.(*svmv1beta1.StorageVersionMigration)
	newSVM := newObj.(*svmv1beta1.StorageVersionMigration)
	logger.V(4).Info("Updating", "svm", klog.KObj(oldSVM))
	rv.enqueue(newSVM)
}

func (rv *ResourceVersionController) enqueue(svm *svmv1beta1.StorageVersionMigration) {
	key, err := controller.KeyFunc(svm)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %w", svm, err))
		return
	}

	rv.queue.Add(key)
}

func (rv *ResourceVersionController) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()
	defer rv.queue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting", "controller", ResourceVersionControllerName)
	defer logger.Info("Shutting down", "controller", ResourceVersionControllerName)

	if !cache.WaitForNamedCacheSyncWithContext(ctx, rv.svmSynced) {
		return
	}

	go wait.UntilWithContext(ctx, rv.worker, time.Second)

	<-ctx.Done()
}

func (rv *ResourceVersionController) worker(ctx context.Context) {
	for rv.processNext(ctx) {
	}
}

func (rv *ResourceVersionController) processNext(ctx context.Context) bool {
	key, quit := rv.queue.Get()
	if quit {
		return false
	}
	defer rv.queue.Done(key)

	err := rv.sync(ctx, key)
	if err == nil {
		rv.queue.Forget(key)
		return true
	}

	klog.FromContext(ctx).V(2).Info("Error syncing SVM resource, retrying", "svm", key, "err", err)
	rv.queue.AddRateLimited(key)

	return true
}

func (rv *ResourceVersionController) sync(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	startTime := time.Now()

	// SVM is a cluster scoped resource so we don't care about the namespace
	_, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	svm, err := rv.svmListers.Get(name)
	if apierrors.IsNotFound(err) {
		// no work to do, don't fail and requeue
		return nil
	}
	if err != nil {
		return err
	}
	// working with copy to avoid race condition between this and migration controller
	toBeProcessedSVM := svm.DeepCopy()
	gr := toBeProcessedSVM.Spec.Resource

	if meta.IsStatusConditionTrue(toBeProcessedSVM.Status.Conditions, string(svmv1beta1.MigrationSucceeded)) ||
		meta.IsStatusConditionTrue(toBeProcessedSVM.Status.Conditions, string(svmv1beta1.MigrationFailed)) {
		logger.V(4).Info("Migration has already succeeded or failed previously, skipping", "svm", name)
		return nil
	}

	if len(toBeProcessedSVM.Status.ResourceVersion) != 0 {
		logger.V(4).Info("Resource version is already set", "svm", name)
		return nil
	}

	gvr, exists, err := resourceFor(rv.mapper, gr)
	if err != nil {
		return err
	}
	if !exists {
		// our GC cache could be missing a recently created custom resource, so give it some time to catch up
		// we resync discovery every 30 seconds so twice that should be sufficient
		if toBeProcessedSVM.CreationTimestamp.Add(time.Minute).After(time.Now()) {
			return fmt.Errorf("resource does not exist in our rest mapper, requeuing to attempt again")
		}
		return rv.failMigration(ctx, toBeProcessedSVM, "resource does not exist in discovery")
	}

	isMigratable, err := rv.isResourceMigratable(*gvr)
	if err != nil {
		return err
	}
	// The resource does not support CRUD operations for migration.
	if !isMigratable {
		err := fmt.Errorf("resource %q does not support discovery operations: %v", gvr.String(), verbsRequiredForMigration)
		logger.Error(err, "resource is not able to be migrated, not retrying", "gvr", gvr.String())
		return rv.failMigration(ctx, toBeProcessedSVM, err.Error())
	}

	latestRV, err := rv.getLatestResourceVersion(*gvr, ctx)
	if err != nil {
		return err
	}

	// Compare the resource version against itself, if it fails then it is not a
	// well formed RV and we should fail migration.
	if _, err := resourceversion.CompareResourceVersion(latestRV, latestRV); err != nil {
		err := fmt.Errorf("latest resourceVersion for %s is not valid: %w", gvr.String(), err)
		return rv.failMigration(ctx, toBeProcessedSVM, err.Error())
	}
	toBeProcessedSVM.Status.ResourceVersion = latestRV

	_, err = rv.kubeClient.StoragemigrationV1beta1().
		StorageVersionMigrations().
		UpdateStatus(ctx, toBeProcessedSVM, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("error updating status for %s: %w", toBeProcessedSVM.Name, err)
	}

	logger.V(4).Info("Resource version has been successfully added", "svm", key, "elapsed", time.Since(startTime))
	return nil
}

func (rv *ResourceVersionController) getLatestResourceVersion(gvr schema.GroupVersionResource, ctx context.Context) (string, error) {
	isResourceNamespaceScoped, err := rv.isResourceNamespaceScoped(gvr)
	if err != nil {
		return "", err
	}

	var randomList *metav1.PartialObjectMetadataList
	if isResourceNamespaceScoped {
		// get list resourceVersion from random non-existent namesapce for the given GVR
		randomList, err = rv.metadataClient.Resource(gvr).
			Namespace(fakeSVMNamespaceName).
			List(ctx, metav1.ListOptions{
				Limit: 1,
			})
	} else {
		randomList, err = rv.metadataClient.Resource(gvr).
			List(ctx, metav1.ListOptions{
				Limit: 1,
			})
	}
	if err != nil {
		// error here is very abstract. adding additional context for better debugging
		return "", fmt.Errorf("error getting latest resourceVersion for %s: %w", gvr.String(), err)
	}

	return randomList.GetResourceVersion(), err
}

func resourceFor(mapper meta.RESTMapper, gr metav1.GroupResource) (*schema.GroupVersionResource, bool, error) {
	partial := schema.GroupVersionResource{
		Group:    gr.Group,
		Resource: gr.Resource,
	}
	mapperGVRs, err := mapper.ResourcesFor(partial)
	if meta.IsNoMatchError(err) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}

	for _, mapperGVR := range mapperGVRs {
		if mapperGVR.Group == gr.Group && mapperGVR.Resource == gr.Resource {
			return &mapperGVR, true, nil
		}
	}

	return nil, false, nil
}

func (rv *ResourceVersionController) isResourceNamespaceScoped(gvr schema.GroupVersionResource) (bool, error) {
	resourceList, err := rv.discoveryClient.ServerResourcesForGroupVersion(gvr.GroupVersion().String())
	if err != nil {
		return false, err
	}

	for _, resource := range resourceList.APIResources {
		if resource.Name == gvr.Resource {
			return resource.Namespaced, nil
		}
	}

	return false, fmt.Errorf("resource %q not found", gvr.String())
}

// isResourceMigratable checks if the GVR has the list of verbs required for
// migration. Returns true if all verbs are in the discovery document and false
// otherwise. If there is an error querying the discovery client or we fail to
// get the GVR, return an error.
func (rv *ResourceVersionController) isResourceMigratable(gvr schema.GroupVersionResource) (bool, error) {
	resourceList, err := rv.discoveryClient.ServerResourcesForGroupVersion(gvr.GroupVersion().String())
	if apierrors.IsNotFound(err) {
		return false, nil
	}

	if resourceList != nil {
		// even in case of an error above there might be a partial list for APIs that
		// were already successfully discovered.
		for _, resource := range resourceList.APIResources {
			if resource.Name == gvr.Resource {
				if resource.Verbs != nil && sets.NewString(resource.Verbs...).HasAll(verbsRequiredForMigration...) {
					return true, nil
				}
				return false, nil
			}
		}
	}

	if err != nil {
		return false, err
	}
	return false, fmt.Errorf("resource %q not found in discovery", gvr.String())
}

func (rv *ResourceVersionController) failMigration(ctx context.Context, svm *svmv1beta1.StorageVersionMigration, message string) error {
	_, err := rv.kubeClient.StoragemigrationV1beta1().
		StorageVersionMigrations().
		UpdateStatus(
			ctx,
			setStatusConditions(svm, svmv1beta1.MigrationFailed, migrationFailedStatusReason, message),
			metav1.UpdateOptions{},
		)
	if err != nil {
		return err
	}
	return nil
}
