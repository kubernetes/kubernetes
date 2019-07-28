/*
Copyright 2018 The Kubernetes Authors.

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

package cloudresource

import (
	"context"
	"fmt"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	cloudprovider "k8s.io/cloud-provider"

	"k8s.io/klog"
)

// SyncManager is an interface for making requests to a cloud provider
type SyncManager interface {
	Run(stopCh <-chan struct{})
	NodeAddresses() ([]v1.NodeAddress, error)
}

var _ SyncManager = &cloudResourceSyncManager{}

type cloudResourceSyncManager struct {
	// Cloud provider interface.
	cloud cloudprovider.Interface
	// Sync period
	syncPeriod time.Duration

	nodeAddressesMonitor *sync.Cond
	nodeAddressesErr     error
	nodeAddresses        []v1.NodeAddress

	nodeName types.NodeName
}

// NewSyncManager creates a manager responsible for collecting resources from a
// cloud provider through requests that are sensitive to timeouts and hanging
func NewSyncManager(cloud cloudprovider.Interface, nodeName types.NodeName, syncPeriod time.Duration) SyncManager {
	return &cloudResourceSyncManager{
		cloud:      cloud,
		syncPeriod: syncPeriod,
		nodeName:   nodeName,
		// nodeAddressesMonitor is a monitor that guards a result (nodeAddresses,
		// nodeAddressesErr) of the sync loop under the condition that a result has
		// been saved at least once. The semantics here are:
		//
		// * Readers of the result will wait on the monitor until the first result
		//   has been saved.
		// * The sync loop (i.e. the only writer), will signal all waiters every
		//   time it updates the result.
		nodeAddressesMonitor: sync.NewCond(&sync.Mutex{}),
	}
}

// NodeAddresses waits for the first sync loop to run. If no successful syncs
// have run, it will return the most recent error. If node addresses have been
// synced successfully, it will return the list of node addresses from the most
// recent successful sync.
func (m *cloudResourceSyncManager) NodeAddresses() ([]v1.NodeAddress, error) {
	m.nodeAddressesMonitor.L.Lock()
	defer m.nodeAddressesMonitor.L.Unlock()
	// wait until there is something
	for {
		if addrs, err := m.nodeAddresses, m.nodeAddressesErr; len(addrs) > 0 || err != nil {
			return addrs, err
		}
		klog.V(5).Infof("Waiting for cloud provider to provide node addresses")
		m.nodeAddressesMonitor.Wait()
	}
}

// getNodeAddresses calls the cloud provider to get a current list of node addresses.
func (m *cloudResourceSyncManager) getNodeAddresses() ([]v1.NodeAddress, error) {
	// TODO(roberthbailey): Can we do this without having credentials to talk to
	// the cloud provider?
	// TODO(justinsb): We can if CurrentNodeName() was actually CurrentNode() and
	// returned an interface.
	// TODO: If IP addresses couldn't be fetched from the cloud provider, should
	// kubelet fallback on the other methods for getting the IP below?
	instances, ok := m.cloud.Instances()
	if !ok {
		return nil, fmt.Errorf("failed to get instances from cloud provider")
	}
	return instances.NodeAddresses(context.TODO(), m.nodeName)
}

func (m *cloudResourceSyncManager) syncNodeAddresses() {
	klog.V(5).Infof("Requesting node addresses from cloud provider for node %q", m.nodeName)

	addrs, err := m.getNodeAddresses()

	m.nodeAddressesMonitor.L.Lock()
	defer m.nodeAddressesMonitor.L.Unlock()
	defer m.nodeAddressesMonitor.Broadcast()

	if err != nil {
		klog.V(2).Infof("Node addresses from cloud provider for node %q not collected: %v", m.nodeName, err)

		if len(m.nodeAddresses) > 0 {
			// in the event that a sync loop fails when a previous sync had
			// succeeded, continue to use the old addresses.
			return
		}

		m.nodeAddressesErr = fmt.Errorf("failed to get node address from cloud provider: %v", err)
		return
	}

	klog.V(5).Infof("Node addresses from cloud provider for node %q collected", m.nodeName)
	m.nodeAddressesErr = nil
	m.nodeAddresses = addrs
}

// Run starts the cloud resource sync manager's sync loop.
func (m *cloudResourceSyncManager) Run(stopCh <-chan struct{}) {
	wait.Until(m.syncNodeAddresses, m.syncPeriod, stopCh)
}
