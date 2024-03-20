/*
Copyright 2023 The Kubernetes Authors.

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

	apiserverinternalv1alpha1 "k8s.io/api/apiserverinternal/v1alpha1"
	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	genericstorageversion "k8s.io/apiserver/pkg/storageversion"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

const (
	// If the StorageVersionAPI feature gate is enabled, after a CRD storage
	// (a.k.a. serving info) is created, the API server will block CR write
	// requests that hit this storage until the corresponding storage
	// version update gets processed by the storage version manager.
	// The API server will unblock CR write requests if the storage version
	// update takes longer than storageVersionUpdateTimeout after the
	// storage is created.
	storageVersionUpdateTimeout = 15 * time.Second
)

var (
	// map from crd.UID -> storageVersionUpdateInfo to keep
	// track of the latest storageversion update for a CRD.
	latestStorageVersionUpdateInfo sync.Map

	// workqueues per crd.UID to process storageversion updates.
	updateQueues sync.Map
)

// Manager provides methods for updating StorageVersion for CRDs. It does
// goroutine management to allow CRD storage version updates running in the
// background and not blocking the caller.
type Manager struct {
	// client is the client interface that manager uses to update
	// StorageVersion objects.
	client genericstorageversion.Client
	// apiserverID is the ID of the apiserver that invokes this manager.
	apiserverID string
	// quit if true, shutsdown all updateQueues
	shutdownQueues chan struct{}
}

// storageVersionUpdateInfo holds information about a storage version update,
// indicating whether the update gets processed, or timed-out.
type storageVersionUpdateInfo struct {
	crd *apiextensionsv1.CustomResourceDefinition

	// updateChannels contain the channels that indicate whether
	// a storageversion udpate succeeded or errored out.
	// CR handler will refer to these to know when to unblock
	// or fail a CR request.
	updateChannels *storageVersionUpdateChannels

	// processed is set to true when the storageversion update is done being processed,
	// whether it failed or succeeded. Used to avoid redundant storagversion
	// updates on unchanged CRDs.
	processed bool
}

type storageVersionUpdateChannels struct {
	// processedCh is closed by the storage version manager after the
	// storage version update gets processed successfully.
	// The API server will unblock and allow CR write requests if this
	// channel is closed.
	processedCh chan struct{}

	// errCh is closed by the storage version manager when it
	// encounters an error while trying to update a storage version.
	// The API server will block the serve 503 for CR write requests if
	// this channel is closed.
	errCh chan struct{}

	// teardownFinishedCh is closed when the teardown of a previous
	// storageversion completes. This is used by the storageversion manager
	// to unblock publishing of the latest storageversion.
	teardownFinishedCh <-chan struct{}
}

// NewManager creates a CRD StorageVersion Manager.
func NewManager(client genericstorageversion.Client, apiserverID string) Manager {
	return Manager{
		client:         client,
		apiserverID:    apiserverID,
		shutdownQueues: make(chan struct{}),
	}
}

func (m *Manager) Enqueue(crd *apiextensionsv1.CustomResourceDefinition, tearDownFinishedCh <-chan struct{}) {
	m.recordLatestSVUpdateInfoWithTeardown(crd, tearDownFinishedCh)
	q := m.getOrCreateUpdateQueueFor(crd)
	q.Add(crd.UID)
}

func (m *Manager) WaitForStorageVersionUpdate(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition) error {
	var latestSVUpdateInfo *storageVersionUpdateInfo
	val, found := latestStorageVersionUpdateInfo.Load(crd.UID)
	if !found {
		return nil
	}

	latestSVUpdateInfo = val.(*storageVersionUpdateInfo)

	// NOTE: currently the graceful CRD deletion waits 1s for in-flight requests
	// to register themselves to the wait group. Ideally the storage version update should
	// not cause the requests to miss the 1s window; otherwise the requests may
	// fail ungracefully (e.g. it may happen if the CRD was deleted immediately after the
	// first CR request establishes the underlying storage).
	select {
	case <-latestSVUpdateInfo.updateChannels.errCh:
		return fmt.Errorf("error while waiting for CRD storage version update")
	case <-latestSVUpdateInfo.updateChannels.processedCh:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("aborted waiting for CRD storage version update: %w", ctx.Err())
	case <-time.After(storageVersionUpdateTimeout):
		return fmt.Errorf("timeout waiting for CRD storage version update")
	}

}

func (m *Manager) Shutdown(stopCh <-chan struct{}) {
	<-stopCh
	close(m.shutdownQueues)
}

func (m *Manager) sync(stopCh <-chan struct{}, crdUID types.UID) {
	go wait.Until(func() {
		m.worker(crdUID)
	}, time.Second, stopCh)
	val, _ := updateQueues.Load(crdUID)
	queue := val.(*workqueue.Type)
	defer queue.ShutDown()
	<-stopCh
}

func (m *Manager) worker(crdUID types.UID) {
	for m.processLatestUpdateFor(crdUID) {
	}
}

func (m *Manager) processLatestUpdateFor(crdUID types.UID) bool {
	ctx := context.TODO()
	val, _ := updateQueues.Load(crdUID)
	queue := val.(*workqueue.Type)

	key, quit := queue.Get()
	defer queue.Done(key)
	if quit {
		return false
	}

	val, ok := latestStorageVersionUpdateInfo.Load(key)
	latestSVUpdateInfo := val.(*storageVersionUpdateInfo)
	if !ok || latestSVUpdateInfo.processed {
		klog.V(4).Infof("No pending storageversion update found for crdUID: %s, returning", crdUID)
		return true
	}

	m.updateStorageVersion(ctx, latestSVUpdateInfo.crd, latestSVUpdateInfo.updateChannels.teardownFinishedCh, latestSVUpdateInfo.updateChannels.processedCh, latestSVUpdateInfo.updateChannels.errCh)
	markUpdateAsProcessed(latestSVUpdateInfo)
	return true
}

func (m *Manager) recordLatestSVUpdateInfoWithTeardown(crd *apiextensionsv1.CustomResourceDefinition, tearDownFinishedCh <-chan struct{}) {
	svUpdateInfo := &storageVersionUpdateInfo{
		crd: crd,
		updateChannels: &storageVersionUpdateChannels{
			processedCh:        make(chan struct{}),
			errCh:              make(chan struct{}),
			teardownFinishedCh: tearDownFinishedCh,
		},
	}

	// overwrite existing SVUpdateInfo with latest update event.
	m.updateLatestStorageVersionUpdateInfo(crd.UID, svUpdateInfo)
}

func (m *Manager) updateLatestStorageVersionUpdateInfo(crdUID types.UID, svUpdateInfo *storageVersionUpdateInfo) {
	latestStorageVersionUpdateInfo.Store(crdUID, svUpdateInfo)
}

func (m *Manager) getOrCreateUpdateQueueFor(crd *apiextensionsv1.CustomResourceDefinition) *workqueue.Type {
	val, ok := updateQueues.Load(crd.UID)
	if ok {
		queue := val.(*workqueue.Type)
		return queue
	}

	queue := workqueue.NewNamed(fmt.Sprintf("%s-storageversion-updater", crd.Name))
	updateQueues.Store(crd.UID, queue)
	go m.sync(m.shutdownQueues, crd.UID)
	return queue
}

// updateStorageVersion updates a StorageVersion for the given
// CRD and returns immediately. Optionally, the caller may specify a
// non-nil waitCh and/or a non-nil processedCh.
// A non-nil waitCh will block the StorageVersion update until waitCh is
// closed.
// The manager will close the non-nil processedCh if it finished
// processing the StorageVersion update (note that the update can either
// succeeded or failed).
func (m *Manager) updateStorageVersion(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition,
	waitCh <-chan struct{}, processedCh chan<- struct{}, errCh chan<- struct{}) {
	if waitCh != nil {
		done := false
		for {
			select {
			case <-waitCh:
				done = true
			case <-ctx.Done():
				klog.V(4).Infof("aborting storageversion update for %v", crd)
				if errCh != nil {
					close(errCh)
				}
				return
			case <-time.After(1 * time.Minute):
				klog.V(4).Infof("timeout waiting for waitCh to close before proceeding with storageversion update for %v", crd)
				if errCh != nil {
					close(errCh)
				}
				return
			}
			if done {
				break
			}
		}
	}

	if err := m.updateCRDStorageVersion(ctx, crd); err != nil {
		utilruntime.HandleError(err)
		if errCh != nil {
			klog.Infof("error while updating storage version for crd %v: %v", crd, err)
			close(errCh)
		}
	}

	// close processCh after the update is done
	if processedCh != nil {
		close(processedCh)
	}
}

func (m *Manager) updateCRDStorageVersion(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition) error {
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
	return genericstorageversion.UpdateStorageVersionFor(
		ctx,
		m.client,
		m.apiserverID,
		gr,
		encodingVersion,
		decodableVersions,
		servedVersions,
		appendOwnerRefFunc)
}

func markUpdateAsProcessed(svUpdateInfo *storageVersionUpdateInfo) {
	svUpdateInfo.processed = true
}
