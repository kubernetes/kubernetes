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

// Manager provides methods for updating StorageVersion for CRDs. It does
// goroutine management to allow CRD storage version updates running in the
// background and not blocking the caller.
type Manager struct {
	// client is the client interface that manager uses to update
	// StorageVersion objects.
	client genericstorageversion.Client
	// apiserverID is the ID of the apiserver that invokes this manager.
	apiserverID                    string
	queue                          *workqueue.Type
	latestStorageVersionUpdateInfo sync.Map
}

// StorageVersionUpdateInfo holds information about a storage version update,
// indicating whether the update gets processed, or timed-out.
type StorageVersionUpdateInfo struct {
	crd *apiextensionsv1.CustomResourceDefinition

	// updateChannels contain the channels that indicate whether
	// a storageversion udpate succeeded or errored out.
	// CR handler will refer to these to know when to unblock
	// or fail a CR request.
	updateChannels *storageVersionUpdateChannels

	// timeout is the time after which the API server will fail the CR
	// write requests indicating that the storageversion update timed out.
	timeout time.Time
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
		client:      client,
		apiserverID: apiserverID,
		queue:       workqueue.NewNamed("storageversion-updater"),
	}
}

func (m *Manager) Enqueue(crd *apiextensionsv1.CustomResourceDefinition, tearDownFinishedCh <-chan struct{}) {
	m.recordLatestSVUpdateInfoWithTeardown(crd, tearDownFinishedCh)
	m.queue.Add(crd.UID)
}

func (m *Manager) WaitForStorageVersionUpdate(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition) error {
	var latestSVUpdateInfo *StorageVersionUpdateInfo
	val, found := m.latestStorageVersionUpdateInfo.Load(crd.UID)
	if !found {
		return nil
	}

	latestSVUpdateInfo = val.(*StorageVersionUpdateInfo)

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
	// Unblock the requests if the storage version update takes a long time, otherwise
	// CR requests may stack up and overwhelm the API server.
	// TODO(roycaihw): benchmark the storage version update latency to adjust the timeout.
	case <-time.After(time.Until(latestSVUpdateInfo.timeout)):
		return fmt.Errorf("timeout waiting for CRD storage version update")
	}

}

func (m *Manager) RunUpdateLoop(stopCh <-chan struct{}) {
	go wait.Until(m.worker, time.Second, stopCh)
	defer m.queue.ShutDown()
	<-stopCh
}

func (m *Manager) worker() {
	for m.processLatestSVUpdate() {
	}
}

func (m *Manager) processLatestSVUpdate() bool {
	key, quit := m.queue.Get()
	if quit {
		return false
	}
	defer m.queue.Done(key)

	val, ok := m.latestStorageVersionUpdateInfo.Load(key)
	if !ok {
		klog.V(2).Infof("No latest storageversion update found, skipping SV update")
		return true
	}

	latestSVUpdateInfo := val.(*StorageVersionUpdateInfo)
	m.updateStorageVersion(latestSVUpdateInfo.crd, latestSVUpdateInfo.updateChannels.teardownFinishedCh, latestSVUpdateInfo.updateChannels.processedCh, latestSVUpdateInfo.updateChannels.errCh)

	// TODO: avoid doing SV updates on the same latest info retrieved.
	return true
}

func (m *Manager) recordLatestSVUpdateInfoWithTeardown(crd *apiextensionsv1.CustomResourceDefinition, tearDownFinishedCh <-chan struct{}) {
	svUpdateInfo := &StorageVersionUpdateInfo{
		crd:     crd,
		timeout: time.Now().Add(storageVersionUpdateTimeout),
		updateChannels: &storageVersionUpdateChannels{
			processedCh:        make(chan struct{}),
			errCh:              make(chan struct{}),
			teardownFinishedCh: tearDownFinishedCh,
		},
	}

	// overwrite existing SVUpdateInfo with latest update event.
	m.updateLatestStorageVersionUpdateInfo(crd.UID, svUpdateInfo)
}

func (m *Manager) updateLatestStorageVersionUpdateInfo(crdUID types.UID, svUpdateInfo *StorageVersionUpdateInfo) {
	m.latestStorageVersionUpdateInfo.Store(crdUID, svUpdateInfo)
}

// updateStorageVersion updates a StorageVesrion for the given
// CRD and returns immediately. Optionally, the caller may specify a
// non-nil waitCh and/or a non-nil processedCh.
// A non-nil waitCh will block the StorageVersion update until waitCh is
// closed.
// The manager will close the non-nil processedCh if it finished
// processing the StorageVersion update (note that the update can either
// succeeded or failed).
func (m *Manager) updateStorageVersion(crd *apiextensionsv1.CustomResourceDefinition,
	waitCh <-chan struct{}, processedCh chan<- struct{}, errCh chan<- struct{}) {
	// TODO: propagate context correctly
	ctx := context.TODO()
	if waitCh != nil {
		done := false
		for {
			select {
			case <-waitCh:
				done = true
			case <-time.After(1 * time.Minute):
				if errCh != nil {
					klog.Infof("timeout waiting for waitCh to close before proceeding with storageversion update for %v", crd)
					close(errCh)
					return
				}
			}
			if done {
				break
			}
		}
	}

	if err := m.updateCRDStorageVersion(ctx, crd); err != nil {
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
		// This should never happened if crd is valid, which is true since we
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
