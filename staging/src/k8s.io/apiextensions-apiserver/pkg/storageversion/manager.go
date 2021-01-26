/*
Copyright 2021 The Kubernetes Authors.

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

	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"

	apiserverinternalv1alpha1 "k8s.io/api/apiserverinternal/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	genericstorageversion "k8s.io/apiserver/pkg/storageversion"
	"k8s.io/klog/v2"
)

const (
	// updateQueueBufferSize is the channel buffer size for each
	// StorageVersion updateQueue. Since the storage version manager keeps
	// one updateQueue for each CRD UID, this means each CRD may have at
	// most 10 pending storage version updates in the queue. When a queue
	// is full, new storage version updates will be dropped.
	updateQueueBufferSize = 10
)

// Manager provides methods for updating StorageVersion for CRDs. It does
// goroutine management to allow CRD storage version updates running in the
// background and not blocking the caller.
type Manager interface {
	// EnqueueStorageVersionUpdate queues a StorageVesrion update for the given
	// CRD and returns immediately. Optionally, the caller may specify a
	// non-nil waitCh and/or a non-nil processedCh and a non-nil abortedCh.
	// A non-nil waitCh will block the StorageVersion update until waitCh is
	// closed.
	// The manager will close the non-nil processedCh if it finished
	// processing the StorageVersion update (note that the update can either
	// succeeded or failed), or close the non-nil abortedCh if it aborted
	// processing the update.
	EnqueueStorageVersionUpdate(crd *apiextensionsv1.CustomResourceDefinition,
		waitCh <-chan struct{}, processedCh, abortedCh chan<- struct{})
	// TeardownFor aborts all pending updates for the given CRD UID, and
	// stops the corresponding goroutine.
	TeardownFor(uid types.UID)
}

// update represents one CRD StorageVersion update request that needs to be processed.
type update struct {
	crd *apiextensionsv1.CustomResourceDefinition
	// If non-nil, wait for the channel to be closed before processing the update.
	waitCh <-chan struct{}
	// If non-nil, close the channel after the update process is finished.
	processedCh chan<- struct{}
	// If non-nil, close the channel if the update process is aborted.
	abortedCh chan<- struct{}
}

// updateQueue is a queue of StorageVersion updates. Upon creation, a goroutine
// is also created which keeps processing pending updates in the queue, until
// the queue is closed.
type updateQueue struct {
	// q is the actual queue. A user can send StorageVersion updates to the
	// queue. A goroutine runs in the background keeps processing the
	// pending updates and doing the actual work, until the queue is closed.
	q chan<- *update
	// All the updates in the queue share the same context. Calling cancel
	// aborts all the pending updates in the queue. This function is used
	// when a CRD is deleted and we want to release all the associated
	// resources (channel and goroutine). The caller should also close q to
	// stop the background goroutine.
	cancel context.CancelFunc
}

// manager implements the Manager interface.
type manager struct {
	// lock protects updateQueues from concurrent writes, and protects
	// individual queues from concurrent write and close().
	lock sync.Mutex
	// updateQueues holds a CRD UID to updateQueue map. Each CRD has its
	// own queue of StorageVersion updates, and a goroutine which processes
	// the pending updates in the queue. The manager sends update requests
	// to the queue.
	updateQueues map[types.UID]*updateQueue
	// client is the client interface that manager uses to update
	// StorageVersion objects.
	client genericstorageversion.Client
	// apiserverID is the ID of the apiserver that invokes this manager.
	apiserverID string
}

// NewManager creates a CRD StorageVersion Manager.
func NewManager(client genericstorageversion.Client, apiserverID string) Manager {
	return &manager{
		updateQueues: make(map[types.UID]*updateQueue),
		client:       client,
		apiserverID:  apiserverID,
	}
}

// EnqueueStorageVersionUpdate queues a StorageVesrion update for the given
// CRD and returns immediately. Optionally, the caller may specify a
// non-nil waitCh and/or a non-nil processedCh and a non-nil abortedCh.
// A non-nil waitCh will block the StorageVersion update until waitCh is
// closed.
// The manager will close the non-nil processedCh if it finished
// processing the StorageVersion update (note that the update can either
// succeeded or failed), or close the non-nil abortedCh if it aborted
// processing the update.
func (m *manager) EnqueueStorageVersionUpdate(crd *apiextensionsv1.CustomResourceDefinition,
	waitCh <-chan struct{}, processedCh, abortedCh chan<- struct{}) {
	m.lock.Lock()
	defer m.lock.Unlock()
	q := m.getOrCreateUpdateQueueLocked(crd.UID)
	// When the channel buffer is full, writing to the channel becomes
	// blocking. Here we give up updating storage version for this CRD and
	// print a log, so that we can return immediately and not block the
	// caller.
	if len(q) == updateQueueBufferSize {
		// TODO(roycaihw): use warning instead of info when StorageVersionAPI
		// graduates to beta/GA
		klog.V(2).Infof("Skipping the storage version update for CRD with UID %v due to the queue being full (queue size: %v).",
			crd.UID, updateQueueBufferSize)
		if processedCh != nil {
			close(processedCh)
		}
		return
	}
	// m.lock ensures we won't write to a closed queue.
	q <- &update{
		crd:         crd,
		waitCh:      waitCh,
		processedCh: processedCh,
		abortedCh:   abortedCh,
	}
}

// getOrCreateUpdateQueueLocked returns the channel for the given UID, or create a new
// one and a new goroutine if necessary. The goroutine keeps processing updates
// until the channel is closed. The caller should hold the manager's lock.
func (m *manager) getOrCreateUpdateQueueLocked(uid types.UID) chan<- *update {
	if queue, ok := m.updateQueues[uid]; ok {
		return queue.q
	}

	queue := make(chan *update, updateQueueBufferSize)
	ctx, cancel := context.WithCancel(context.TODO())
	m.updateQueues[uid] = &updateQueue{
		q:      queue,
		cancel: cancel,
	}
	go func() {
		defer func() {
			err := recover()
			if err != nil {
				// Log the panic and teardown the queue, so
				// that the manager can restart a new queue.
				utilruntime.HandleError(fmt.Errorf("[SHOULD NOT HAPPEN] observed panic in CRD storage version update queue %v: %v", uid, err))
				m.TeardownFor(uid)
			}
		}()
		for update := range queue {
			select {
			case <-ctx.Done():
				// The queue was cancelled. Abort the update.
				if update.abortedCh != nil {
					close(update.abortedCh)
				}
				continue
			default:
			}

			// TODO(roycaihw): there are two types of updates:
			//   1) the ones with nil processedCh and abortedCh, requested by
			//      watch events handler
			//   2) the ones with non-nil processedCh and abortedCh, requested
			//      by newly-created CRD storage
			// An update of type 1) can be merged with a consecutive update,
			// where the latter update's storage version is honored, and both
			// updates' waitChs get evaluated.
			if update.waitCh != nil {
				<-update.waitCh
			}
			if err := m.updateCRDStorageVersion(ctx, update.crd); err != nil {
				utilruntime.HandleError(err)
			}
			if update.processedCh != nil {
				select {
				case <-ctx.Done():
					// The queue was cancelled. Potentially we didn't finish this
					// storage version update.
					if update.abortedCh != nil {
						close(update.abortedCh)
					}
				default:
					close(update.processedCh)
				}
			}
		}
	}()
	return queue
}

// TeardownFor closes the channel for the given UID. It ensures that we don't
// leak goroutines.
func (m *manager) TeardownFor(uid types.UID) {
	m.lock.Lock()
	defer m.lock.Unlock()
	if queue, ok := m.updateQueues[uid]; ok {
		// Cancel all the pending updates, so that if the CRD is
		// re-created, the old updates won't race with the new updates.
		// We can safely discard the old storage version updates
		// because:
		//   1. if the CRD is deleted forever, all CRs will be GC'ed and
		//      the storage version doesn't matter.
		//   2. if the CRD gets deleted and re-created in a short period
		//      and the old CRs remain, the new CRD will update new
		//      storage version.
		klog.V(4).Infof("Cancelling the storage version update queue for CRD with UID %v.",
			uid)
		queue.cancel()
		// Since writers to the queue acquire the same lock as we do, we
		// make sure no one can write to a closed queue.
		close(queue.q)
		delete(m.updateQueues, uid)
	}
}

func (m *manager) updateCRDStorageVersion(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition) error {
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
	var decodableVersions []string
	for _, v := range crd.Spec.Versions {
		decodableVersions = append(decodableVersions, crd.Spec.Group+"/"+v.Name)
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
		appendOwnerRefFunc)
}
