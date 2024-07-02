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

package storageversion

import (
	"context"
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	apiserverinternalv1alpha1 "k8s.io/api/apiserverinternal/v1alpha1"
	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	crdinformers "k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericstorageversion "k8s.io/apiserver/pkg/storageversion"
)

var (
	// workerFrequency is the frequency at which the worker for updating storageversion for a CRD
	// will be called.
	workerFrequency = 1 * time.Second
)

// Manager maintains necessary structures needed for updating
// StorageVersion for CRDs. It does goroutine management to allow CRD
// storage version updates running in the background and not blocking the caller.
type Manager struct {
	// client is the client interface that manager uses to update
	// StorageVersion objects.
	client genericstorageversion.Client
	// apiserverID is the ID of the apiserver that invokes this manager.
	apiserverID string
	crdInformer crdinformers.CustomResourceDefinitionInformer
	// workqueue to process storageversion updates for this CRD.
	queue *workqueue.Type
	// map from crd.Name -> number of active teardowns for this CRD.
	svUpdateInfoMap sync.Map
}

type svUpdateInfo struct {
	teardownCount              int
	finishedPublishingLatestSV bool
	latestPublishedSV          string
}

// NewManager creates a CRD StorageVersion Manager.
func NewManager(svClient genericstorageversion.Client, apiserverID string,
	crdInformer crdinformers.CustomResourceDefinitionInformer) *Manager {
	return &Manager{
		client:          svClient,
		apiserverID:     apiserverID,
		crdInformer:     crdInformer,
		queue:           workqueue.NewNamed("storageversion-updater"),
		svUpdateInfoMap: sync.Map{},
	}
}

func (m *Manager) CanServeWrite(crd *apiextensionsv1.CustomResourceDefinition) bool {
	val, ok := m.svUpdateInfoMap.Load(crd.Name)
	if !ok {
		return false
	}

	svUpdateInfo := val.(*svUpdateInfo)
	donePublishing := svUpdateInfo.finishedPublishingLatestSV
	svProcessed, err := m.storageVersionAlreadyPublished(crd, svUpdateInfo)
	if err != nil {
		return false
	}

	return donePublishing && svProcessed
}

// UpdateActiveTeardownsCount updates the teardown count of the CRD in sync.Map.
func (m *Manager) UpdateActiveTeardownsCount(crdName string, value int) error {
	val, ok := m.svUpdateInfoMap.Load(crdName)
	if !ok {
		return fmt.Errorf("error while trying to update number of teardowns. No entry found in sync.Map for %v", crdName)
	}
	svUpdateInfo := val.(*svUpdateInfo)
	svUpdateInfo.teardownCount += value

	if svUpdateInfo.teardownCount == 0 {
		svUpdateInfo.finishedPublishingLatestSV = false
	}
	return nil
}

// Enqueue adds the CRD name to the SV upadte queue.
func (m *Manager) Enqueue(crdName string) {
	if crdName == "" {
		return
	}
	klog.V(4).Infof("enqueueing crd %s for storageversion update", crdName)
	m.queue.Add(crdName)
}

// DeleteSVUpdateInfo deletes specified key from the sync.Map.
func (m *Manager) DeleteSVUpdateInfo(crdName string) {
	m.svUpdateInfoMap.Delete(crdName)
}

// Sync runs a goroutine over the SV updater queue to indefinitely
// process any queued storageversion updates unless stopCh is invoked.
func (m *Manager) Sync(stopCh <-chan struct{}, workers int) {
	defer m.queue.ShutDownWithDrain()
	klog.V(2).Infof("starting storageversion sync loop")
	for i := 0; i < workers; i++ {
		go wait.Until(func() {
			m.worker()
		}, workerFrequency, stopCh)
	}

	<-stopCh
}

// storageVersionAlreadyPublished checks whether the storageversion for the provided
// CRD is already published.
func (m *Manager) storageVersionAlreadyPublished(crd *apiextensionsv1.CustomResourceDefinition, svUpdateInfo *svUpdateInfo) (bool, error) {
	svOfCRD, err := apiextensionshelpers.GetCRDStorageVersion(crd)
	if err != nil {
		return false, fmt.Errorf("error while getting storageversion from CRD for %s: %w", crd.Name, err)
	}

	if svOfCRD == svUpdateInfo.latestPublishedSV {
		return true, nil
	}

	return false, nil
}

func (m *Manager) worker() {
	for m.processLatestUpdateFor() {
	}
}

func (m *Manager) processLatestUpdateFor() bool {
	ctx := context.TODO()
	key, quit := m.queue.Get()
	defer m.queue.Done(key)
	if quit {
		return false
	}

	// Fetch the latest CRD from CRD informer cache to update the storageversion for it.
	crdName := key.(string)
	crd, err := m.crdInformer.Lister().Get(crdName)

	if err != nil {
		// No need to update storgaversion if CRD not found.
		if errors.IsNotFound(err) {
			klog.V(4).Infof("crd: %s not found in CRD cache, abandoning storageversion update", crdName)
			return true
		}

		klog.V(4).Infof("error fetching crd: %s from CRD cache, will retry storageversion update", crdName)
		// TODO: enqueue again?
		return true
	}

	skip, reason, err := m.shouldSkipSVUpdate(crd)
	if err != nil {
		klog.V(4).Infof("error while checking if storagversion update can be skipped for crd: %s, err: %v", crd.Name, err)
		// TODO: enqueue again?
		return true
	}

	if skip {
		klog.V(4).Infof("skipping storagversion update for crd: %s, reason: %s ", crd.Name, reason)
		return true
	}

	klog.V(4).Infof("starting storageversion update for crd: %s", crd.Name)
	m.updateSVUpdateInfoMap(crd.Name, &svUpdateInfo{
		finishedPublishingLatestSV: false,
	})

	err = m.updateStorageVersion(ctx, crd)
	if err == nil {
		klog.V(4).Infof("successfully updated storage version for %s", crd.Name)
		return true
	}

	// TODO: indefinitely requeue on error?
	m.Enqueue(crd.Name)
	return true
}

func (m *Manager) shouldSkipSVUpdate(crd *apiextensionsv1.CustomResourceDefinition) (bool, string, error) {
	skip, teardownCount := m.teardownInProgress(crd.Name)
	if skip {
		return true, fmt.Sprintf("%d active teardowns", teardownCount), nil
	}

	return false, "", nil
}

func (m *Manager) teardownInProgress(crdName string) (bool, int) {
	val, _ := m.svUpdateInfoMap.LoadOrStore(crdName, &svUpdateInfo{})
	svUpdateInfo := val.(*svUpdateInfo)
	if svUpdateInfo.teardownCount > 0 {
		return true, svUpdateInfo.teardownCount
	}

	return false, svUpdateInfo.teardownCount
}

func (m *Manager) updateStorageVersion(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition) error {
	gr := schema.GroupResource{
		Group:    crd.Spec.Group,
		Resource: crd.Spec.Names.Plural,
	}
	storageVersion, err := apiextensionshelpers.GetCRDStorageVersion(crd)

	if err != nil {
		// This should never happen if crd is valid, which is true since we
		// only update storage version for CRDs that have been written to the
		// storage.
		return err
	}

	encodingVersion := crd.Spec.Group + "/" + storageVersion
	var servedVersions, decodableVersions []string
	for _, v := range crd.Spec.Versions {
		decodableVersions = append(decodableVersions, crd.Spec.Group+"/"+v.Name)
		if v.Served {
			servedVersions = append(servedVersions, crd.Spec.Group+"/"+v.Name)
		}
	}

	appendOwnerRefFunc := func(sv *apiserverinternalv1alpha1.StorageVersion) error {
		ref := metav1.OwnerReference{
			APIVersion: apiextensionsv1.SchemeGroupVersion.String(),
			Kind:       "CustomResourceDefinition",
			Name:       crd.Name,
			UID:        crd.UID,
		}
		for _, r := range sv.OwnerReferences {
			if r == ref {
				return nil
			}
		}
		sv.OwnerReferences = append(sv.OwnerReferences, ref)
		return nil
	}
	err = genericstorageversion.UpdateStorageVersionFor(
		ctx,
		m.client,
		m.apiserverID,
		gr,
		encodingVersion,
		decodableVersions,
		servedVersions,
		appendOwnerRefFunc)

	if err != nil {
		m.updateSVUpdateInfoMap(crd.Name, &svUpdateInfo{
			finishedPublishingLatestSV: true,
			latestPublishedSV:          storageVersion,
		})
	}
	return err
}

func (m *Manager) updateSVUpdateInfoMap(crdName string, info *svUpdateInfo) {
	val, ok := m.svUpdateInfoMap.LoadOrStore(crdName, info)
	if ok {
		svUpdateInfo := val.(*svUpdateInfo)
		svUpdateInfo.finishedPublishingLatestSV = info.finishedPublishingLatestSV
		if info.latestPublishedSV != "" {
			svUpdateInfo.latestPublishedSV = info.latestPublishedSV
		}
	}
}
