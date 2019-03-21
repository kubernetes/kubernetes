/*
Copyright 2017 The Kubernetes Authors.

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

package status

import (
	"fmt"
	"sync"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
)

const (
	// LoadError indicates that the Kubelet failed to load the config checkpoint
	LoadError = "failed to load config, see Kubelet log for details"
	// ValidateError indicates that the Kubelet failed to validate the config checkpoint
	ValidateError = "failed to validate config, see Kubelet log for details"
	// AllNilSubfieldsError is used when no subfields are set
	// This could happen in the case that an old client tries to read an object from a newer API server with a set subfield it does not know about
	AllNilSubfieldsError = "invalid NodeConfigSource, exactly one subfield must be non-nil, but all were nil"
	// DownloadError is used when the download fails, e.g. due to network issues
	DownloadError = "failed to download config, see Kubelet log for details"
	// InternalError indicates that some internal error happened while trying to sync config, e.g. filesystem issues
	InternalError = "internal failure, see Kubelet log for details"

	// SyncErrorFmt is used when the system couldn't sync the config, due to a malformed Node.Spec.ConfigSource, a download failure, etc.
	SyncErrorFmt = "failed to sync: %s"
)

// NodeConfigStatus represents Node.Status.Config
type NodeConfigStatus interface {
	// SetActive sets the active source in the status
	SetActive(source *apiv1.NodeConfigSource)
	// SetAssigned sets the assigned source in the status
	SetAssigned(source *apiv1.NodeConfigSource)
	// SetLastKnownGood sets the last-known-good source in the status
	SetLastKnownGood(source *apiv1.NodeConfigSource)
	// SetError sets the error associated with the status
	SetError(err string)
	// SetErrorOverride sets an error that overrides the base error set by SetError.
	// If the override is set to the empty string, the base error is reported in
	// the status, otherwise the override is reported.
	SetErrorOverride(err string)
	// Sync patches the current status into the Node identified by `nodeName` if an update is pending
	Sync(client clientset.Interface, nodeName string)
}

type nodeConfigStatus struct {
	// status is the core NodeConfigStatus that we report
	status apiv1.NodeConfigStatus
	// mux is a mutex on the nodeConfigStatus, alternate between setting and syncing the status
	mux sync.Mutex
	// errorOverride is sent in place of the usual error if it is non-empty
	errorOverride string
	// syncCh; write to this channel to indicate that the status needs to be synced to the API server
	syncCh chan bool
}

// NewNodeConfigStatus returns a new NodeConfigStatus interface
func NewNodeConfigStatus() NodeConfigStatus {
	// channels must have capacity at least 1, since we signal with non-blocking writes
	syncCh := make(chan bool, 1)
	// prime new status managers to sync with the API server on the first call to Sync
	syncCh <- true
	return &nodeConfigStatus{
		syncCh: syncCh,
	}
}

// transact grabs the lock, performs the fn, records the need to sync, and releases the lock
func (s *nodeConfigStatus) transact(fn func()) {
	s.mux.Lock()
	defer s.mux.Unlock()
	fn()
	s.sync()
}

func (s *nodeConfigStatus) SetAssigned(source *apiv1.NodeConfigSource) {
	s.transact(func() {
		s.status.Assigned = source
	})
}

func (s *nodeConfigStatus) SetActive(source *apiv1.NodeConfigSource) {
	s.transact(func() {
		s.status.Active = source
	})
}

func (s *nodeConfigStatus) SetLastKnownGood(source *apiv1.NodeConfigSource) {
	s.transact(func() {
		s.status.LastKnownGood = source
	})
}

func (s *nodeConfigStatus) SetError(err string) {
	s.transact(func() {
		s.status.Error = err
	})
}

func (s *nodeConfigStatus) SetErrorOverride(err string) {
	s.transact(func() {
		s.errorOverride = err
	})
}

// sync notes that the status needs to be synced to the API server
func (s *nodeConfigStatus) sync() {
	select {
	case s.syncCh <- true:
	default:
	}
}

// Sync attempts to sync the status with the Node object for this Kubelet,
// if syncing fails, an error is logged, and work is queued for retry.
func (s *nodeConfigStatus) Sync(client clientset.Interface, nodeName string) {
	select {
	case <-s.syncCh:
	default:
		// no work to be done, return
		return
	}

	utillog.Infof("updating Node.Status.Config")

	// grab the lock
	s.mux.Lock()
	defer s.mux.Unlock()

	// if the sync fails, we want to retry
	var err error
	defer func() {
		if err != nil {
			utillog.Errorf(err.Error())
			s.sync()
		}
	}()

	// get the Node so we can check the current status
	oldNode, err := client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	if err != nil {
		err = fmt.Errorf("could not get Node %q, will not sync status, error: %v", nodeName, err)
		return
	}

	status := &s.status
	// override error, if necessary
	if len(s.errorOverride) > 0 {
		// copy the status, so we don't overwrite the prior error
		// with the override
		status = status.DeepCopy()
		status.Error = s.errorOverride
	}

	// update metrics based on the status we will sync
	metrics.SetConfigError(len(status.Error) > 0)
	err = metrics.SetAssignedConfig(status.Assigned)
	if err != nil {
		err = fmt.Errorf("failed to update Assigned config metric, error: %v", err)
		return
	}
	err = metrics.SetActiveConfig(status.Active)
	if err != nil {
		err = fmt.Errorf("failed to update Active config metric, error: %v", err)
		return
	}
	err = metrics.SetLastKnownGoodConfig(status.LastKnownGood)
	if err != nil {
		err = fmt.Errorf("failed to update LastKnownGood config metric, error: %v", err)
		return
	}

	// apply the status to a copy of the node so we don't modify the object in the informer's store
	newNode := oldNode.DeepCopy()
	newNode.Status.Config = status

	// patch the node with the new status
	if _, _, err := nodeutil.PatchNodeStatus(client.CoreV1(), types.NodeName(nodeName), oldNode, newNode); err != nil {
		utillog.Errorf("failed to patch node status, error: %v", err)
	}
}
