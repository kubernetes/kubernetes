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

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
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
	// If the StorageVersionAPI feature gate is enabled, after a CRD storage
	// (a.k.a. serving info) is created, the API server will block CR write
	// requests that hit this storage until the corresponding storage
	// version update gets processed by the storage version manager.
	// The API server will fail CR write requests if the storage version
	// update takes longer than StorageVersionUpdateTimeout after the
	// storage is created.
	StorageVersionUpdateTimeout = 15 * time.Second
	// TeardownFinishedTimeout is the time to wait for teardown of an old storage
	// to finish while updating storageversion of a CRD. If the teardown of
	// old storage times out, we return an error for the storageversion update,
	// and if there are any CR requests blocked on this storgaversion update,
	// they are served 503.
	TeardownFinishedTimeout = 1 * time.Minute
	// WorkerFrequency is the frequency at which the worker for updating storageversion for a CRD
	// will be called.
	WorkerFrequency = 1 * time.Second
	// storageversionUpdateFailureThreshold is the maximum number of times we will
	// requeue the storageversion after the first failure. Once this threshold is crossed,
	// we will fail the storageversion update and consequently the dependent CR writes.
	storageversionUpdateFailureThreshold = 3
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
	// shutdown if set to true, shuts down all per-crd update queues.
	shutdown chan struct{}
	// map from crd.UID -> storageVersionUpdateInfoMap to keep
	// track of the latest storageversion updates for a CRD.
	storageVersionUpdateInfoMap sync.Map
}

// storageVersionUpdateInfo holds information about a storage version update,
// indicating whether the update gets processed or timed-out.
type storageVersionUpdateInfo struct {
	crd *apiextensionsv1.CustomResourceDefinition

	// workqueue to process storageversion updates for this CRD.
	queue *workqueue.Type

	// updateChannels contain the channels that indicate whether
	// a storageversion udpate succeeded or errored out.
	// CR handler will refer to these to know when to unblock
	// or fail a CR request.
	updateChannels *storageVersionUpdateChannels

	// failures keeps track of how many times the storageversion update for this CRD
	// failed. When past the threshold, we stop requeuing this CRD for storageversion updates
	// and return error.
	failures int
}

type storageVersionUpdateChannels struct {
	// processedCh is closed by the storage version manager after the
	// storage version update gets processed successfully.
	// The API server will unblock and allow CR write requests if this
	// channel is closed.
	processedCh chan struct{}

	// errCh is closed by the storage version manager when it
	// encounters an error while trying to update a storage version.
	// The API server will block the serve (503) for CR write requests if
	// this channel is closed.
	errCh chan struct{}

	// teardownFinishedCh is closed when the teardown of a previous
	// storageversion completes. This is used by the storageversion manager
	// to unblock publishing of the latest storageversion.
	teardownFinishedCh <-chan struct{}
}

// NewManager creates a CRD StorageVersion Manager.
func NewManager(client genericstorageversion.Client, apiserverID string) *Manager {
	return &Manager{
		client:                      client,
		apiserverID:                 apiserverID,
		shutdown:                    make(chan struct{}),
		storageVersionUpdateInfoMap: sync.Map{},
	}
}

func (m *Manager) SyncSVOnStartup(crdInformer crdinformers.CustomResourceDefinitionInformer) bool {
	crds, err := crdInformer.Lister().List(labels.Everything())
	if err != nil {
		klog.Errorf("failed to list CRDs from informer while syncing storageversions on server startup : %v", err)
		return false
	}
	for _, crd := range crds {
		m.Enqueue(crd, nil, 0)
	}

	return true
}

// Enqueue records the latest CRD that was updated to process its storageversion update.
// It will first update the sync.Map with the CRD, prepare updateChannels for its storageversion update.
// And finally add it to the relevant CRD queue.
func (m *Manager) Enqueue(crd *apiextensionsv1.CustomResourceDefinition, tearDownFinishedCh <-chan struct{}, failuresSeenSoFar int) {
	svUpdateInfo := m.recordLatestSVUpdateInfo(crd, tearDownFinishedCh, failuresSeenSoFar)
	klog.V(4).Infof("queueing crd %s for storageversion update, failuresSeenSoFar: %d", crd.Name, failuresSeenSoFar)
	svUpdateInfo.queue.Add(crd.UID)
}

// WaitForStorageVersionUpdate allows CR writes to be blocked till the latest storageversion
// of a CRD is updated. If the storageversion update fails, we fail CR writes.
// If the CRD is being deleted, return err and fail the CR write.
func (m *Manager) WaitForStorageVersionUpdate(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition) error {
	var svUpdateInfo *storageVersionUpdateInfo
	if apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Terminating) {
		return fmt.Errorf("CRD %s is being deleted", crd.Name)
	}

	val, found := m.storageVersionUpdateInfoMap.Load(crd.UID)
	if !found {
		// CR write that invoked this call will be served 503.
		return fmt.Errorf("no entry found in storageVersionUpdateInfo map for CRD: %s", crd.Name)
	}

	svUpdateInfo = val.(*storageVersionUpdateInfo)

	// NOTE: currently the graceful CRD deletion waits 1s for in-flight requests
	// to register themselves to the wait group. Ideally the storage version update should
	// not cause the requests to miss the 1s window; otherwise the requests may
	// fail ungracefully (e.g. it may happen if the CRD was deleted immediately after the
	// first CR request establishes the underlying storage).
	select {
	case <-svUpdateInfo.updateChannels.errCh:
		return fmt.Errorf("error while waiting for CRD storage version update")
	case <-svUpdateInfo.updateChannels.processedCh:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("aborted waiting for CRD storage version update: %w", ctx.Err())
	case <-time.After(StorageVersionUpdateTimeout):
		return fmt.Errorf("timeout waiting for CRD storage version update")
	}

}

// DeleteSVUpdateInfo deletes specified key from the sync.Map.
func (m *Manager) DeleteSVUpdateInfo(crdUID types.UID, waitCh <-chan struct{}) {
	// wait for all pending CR writes to drain.
	if waitCh != nil {
		done := false
		for {
			select {
			case <-waitCh:
				done = true
			case <-time.After(TeardownFinishedTimeout):
				klog.V(4).Infof("timeout waiting for waitCh to close before proceeding with deleting storageversion for crdUID %v", crdUID)
				return
			}
			if done {
				break
			}
		}
	}
	// shutdown storageversion update queue
	val, ok := m.storageVersionUpdateInfoMap.Load(crdUID)
	if !ok {
		klog.V(4).Infof("no entry found in sync.map for crdUID %v", crdUID)
		return
	}

	svUpdateInfo := val.(*storageVersionUpdateInfo)
	svUpdateInfo.queue.ShutDown()
	m.storageVersionUpdateInfoMap.Delete(crdUID)
}

// Shutdown signals the storageversion manager to shutdown all
// per-crd storageversion update queues.
func (m *Manager) Shutdown(stopCh <-chan struct{}) {
	<-stopCh
	close(m.shutdown)
}

// sync runs a goroutine over the provided queue to indefinitely
// process any queued storageversion updates unless stopCh is invoked.
func (m *Manager) sync(shutdownQueue <-chan struct{}, queue *workqueue.Type, workerFrequency, teardownFinishedTimeout time.Duration) {
	defer queue.ShutDownWithDrain()
	go wait.Until(func() {
		m.worker(queue, teardownFinishedTimeout)
	}, workerFrequency, shutdownQueue)

	<-shutdownQueue
}

func (m *Manager) worker(queue *workqueue.Type, teardownFinishedTimeout time.Duration) {
	for m.processLatestUpdateFor(queue, teardownFinishedTimeout) {
	}
}

func (m *Manager) processLatestUpdateFor(queue *workqueue.Type, teardownFinishedTimeout time.Duration) bool {
	ctx := context.TODO()

	key, quit := queue.Get()
	defer queue.Done(key)
	if quit {
		return false
	}

	// Note: since we queue CRDs by UID, if the same CRD is updated multiple times,
	// the same UID is queued multiple times. We take care to only update the latest queued
	// entry for this UID, by referring to the sync.Map which is overwritten everytime a
	// new update is made to the CRD.
	// Ex: if there were rapid consecutive updates made to a crd like v1 -> v2 -> v3
	// such that when processLatestUpdateFor() is called for the v1 update, it observes
	// that the latest entry in the sync.map is for v3 - and updates the version to v3.
	// When we get to processLatestUpdateFor() for v2 - the latest entry in sync.map is still
	// v3. We should not process v3 update again.
	val, ok := m.storageVersionUpdateInfoMap.Load(key)
	if !ok {
		klog.V(4).Infof("No pending storageversion update found for crdUID: %s, returning", key)
		return true
	}
	latestSVUpdateInfo := val.(*storageVersionUpdateInfo)
	select {
	case <-latestSVUpdateInfo.updateChannels.processedCh:
		klog.V(4).Infof("Storageversion is already updated to the latest value for crdUID: %s, returning", key)
		return true
	case <-latestSVUpdateInfo.updateChannels.errCh:
		klog.V(4).Infof("Storageversion is already processed for crdUID: %s, but returned an error.", key)
		return true
	default:
		err := m.updateStorageVersion(ctx, latestSVUpdateInfo.crd, latestSVUpdateInfo.updateChannels.teardownFinishedCh, teardownFinishedTimeout)
		m.processStorageVersionUpdateResponse(err, latestSVUpdateInfo)
		return true
	}
}

func (m *Manager) recordLatestSVUpdateInfo(crd *apiextensionsv1.CustomResourceDefinition, tearDownFinishedCh <-chan struct{}, failuresSeenSoFar int) *storageVersionUpdateInfo {
	latestSVUpdateInfo := &storageVersionUpdateInfo{
		crd: crd,
		updateChannels: &storageVersionUpdateChannels{
			processedCh:        make(chan struct{}),
			errCh:              make(chan struct{}),
			teardownFinishedCh: tearDownFinishedCh,
		},
		failures: failuresSeenSoFar,
	}

	val, ok := m.storageVersionUpdateInfoMap.Load(crd.UID)
	if !ok {
		// if CRD seen for the first time,
		// 1. create new update-queue
		// 2. start the queue
		// 3. store it in latestSVUpdateInfo
		// 4. update sync.map and return it
		queue := workqueue.NewNamed(fmt.Sprintf("%s-storageversion-updater", crd.Name))
		latestSVUpdateInfo.queue = queue
		go m.sync(m.shutdown, queue, WorkerFrequency, TeardownFinishedTimeout)
		m.storageVersionUpdateInfoMap.Store(crd.UID, latestSVUpdateInfo)
		return latestSVUpdateInfo
	}

	// overwrite existing SVUpdateInfo in the map only if
	// 1. there's a new latest update event.
	// 2. the update event is the same as last seen, but it was re-enqueued
	// because of some failure in the SV update.
	existingSVUpdateInfo := val.(*storageVersionUpdateInfo)
	if crd.ObjectMeta.ResourceVersion >= existingSVUpdateInfo.crd.ObjectMeta.ResourceVersion {
		updatedSVInfo := &storageVersionUpdateInfo{
			crd:            latestSVUpdateInfo.crd,
			updateChannels: latestSVUpdateInfo.updateChannels,
			failures:       latestSVUpdateInfo.failures,
			queue:          existingSVUpdateInfo.queue,
		}
		m.storageVersionUpdateInfoMap.Store(crd.UID, updatedSVInfo)
		return updatedSVInfo
	}
	return existingSVUpdateInfo
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
	waitCh <-chan struct{}, teardownFinishedTimeout time.Duration) error {
	if waitCh != nil {
		done := false
		for {
			select {
			case <-waitCh:
				done = true
			case <-ctx.Done():
				return fmt.Errorf("aborting storageversion update for %v, context closed", crd)
			case <-time.After(teardownFinishedTimeout):
				return fmt.Errorf("timeout waiting for waitCh to close before proceeding with storageversion update for %v", crd)
			}
			if done {
				break
			}
		}
	}

	if err := m.updateCRDStorageVersion(ctx, crd); err != nil {
		return fmt.Errorf("error while updating storage version for crd %v: %w", crd, err)
	}

	return nil
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

func (m *Manager) processStorageVersionUpdateResponse(err error, svUpdateInfo *storageVersionUpdateInfo) {
	if err == nil {
		klog.V(4).Infof("successfully updated storage version for %s", svUpdateInfo.crd.Name)
		close(svUpdateInfo.updateChannels.processedCh)
		return
	}

	failuresSeenSoFar := svUpdateInfo.failures + 1
	if failuresSeenSoFar > storageversionUpdateFailureThreshold {
		klog.V(4).Infof("storageversion update for crd %s has exceeded maximum allowed update errors", svUpdateInfo.crd.Name)
		close(svUpdateInfo.updateChannels.errCh)
		return
	}

	m.Enqueue(svUpdateInfo.crd, svUpdateInfo.updateChannels.teardownFinishedCh, failuresSeenSoFar)
}
