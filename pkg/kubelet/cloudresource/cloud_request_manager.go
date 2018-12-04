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

var nodeAddressesRetryPeriod = 5 * time.Second

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

	nodeAddressesMux sync.Mutex
	nodeAddressesErr error
	nodeAddresses    []v1.NodeAddress

	nodeName types.NodeName
}

// NewSyncManager creates a manager responsible for collecting resources
// from a cloud provider through requests that are sensitive to timeouts and hanging
func NewSyncManager(cloud cloudprovider.Interface, nodeName types.NodeName, syncPeriod time.Duration) SyncManager {
	return &cloudResourceSyncManager{
		cloud:      cloud,
		syncPeriod: syncPeriod,
		nodeName:   nodeName,
	}
}

func (m *cloudResourceSyncManager) getNodeAddressSafe() ([]v1.NodeAddress, error) {
	m.nodeAddressesMux.Lock()
	defer m.nodeAddressesMux.Unlock()

	return m.nodeAddresses, m.nodeAddressesErr
}

func (m *cloudResourceSyncManager) setNodeAddressSafe(nodeAddresses []v1.NodeAddress, err error) {
	m.nodeAddressesMux.Lock()
	defer m.nodeAddressesMux.Unlock()

	m.nodeAddresses = nodeAddresses
	m.nodeAddressesErr = err
}

// NodeAddresses does not wait for cloud provider to return a node addresses.
// It always returns node addresses or an error.
func (m *cloudResourceSyncManager) NodeAddresses() ([]v1.NodeAddress, error) {
	// wait until there is something
	for {
		nodeAddresses, err := m.getNodeAddressSafe()
		if len(nodeAddresses) == 0 && err == nil {
			klog.V(5).Infof("Waiting for %v for cloud provider to provide node addresses", nodeAddressesRetryPeriod)
			time.Sleep(nodeAddressesRetryPeriod)
			continue
		}
		return nodeAddresses, err
	}
}

func (m *cloudResourceSyncManager) collectNodeAddresses(ctx context.Context, nodeName types.NodeName) {
	klog.V(5).Infof("Requesting node addresses from cloud provider for node %q", nodeName)

	instances, ok := m.cloud.Instances()
	if !ok {
		m.setNodeAddressSafe(nil, fmt.Errorf("failed to get instances from cloud provider"))
		return
	}

	// TODO(roberthbailey): Can we do this without having credentials to talk
	// to the cloud provider?
	// TODO(justinsb): We can if CurrentNodeName() was actually CurrentNode() and returned an interface
	// TODO: If IP addresses couldn't be fetched from the cloud provider, should kubelet fallback on the other methods for getting the IP below?

	nodeAddresses, err := instances.NodeAddresses(ctx, nodeName)
	if err != nil {
		m.setNodeAddressSafe(nil, fmt.Errorf("failed to get node address from cloud provider: %v", err))
		klog.V(2).Infof("Node addresses from cloud provider for node %q not collected", nodeName)
	} else {
		m.setNodeAddressSafe(nodeAddresses, nil)
		klog.V(5).Infof("Node addresses from cloud provider for node %q collected", nodeName)
	}
}

// Run starts the cloud resource sync manager's sync loop.
func (m *cloudResourceSyncManager) Run(stopCh <-chan struct{}) {
	wait.Until(func() {
		m.collectNodeAddresses(context.TODO(), m.nodeName)
	}, m.syncPeriod, stopCh)
}
