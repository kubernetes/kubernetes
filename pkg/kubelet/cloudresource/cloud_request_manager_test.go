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
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/cloud-provider/fake"
)

func createNodeInternalIPAddress(address string) []v1.NodeAddress {
	return []v1.NodeAddress{
		{
			Type:    v1.NodeInternalIP,
			Address: address,
		},
	}
}

func TestNodeAddressesDelay(t *testing.T) {
	syncPeriod := 100 * time.Millisecond
	cloud := &fake.Cloud{
		Addresses: createNodeInternalIPAddress("10.0.1.12"),
		// Set the request delay so the manager timeouts and collects the node addresses later
		RequestDelay: 200 * time.Millisecond,
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	manager := NewSyncManager(cloud, "defaultNode", syncPeriod).(*cloudResourceSyncManager)
	go manager.Run(stopCh)

	nodeAddresses, err := manager.NodeAddresses()
	if err != nil {
		t.Errorf("Unexpected err: %q\n", err)
	}
	if !reflect.DeepEqual(nodeAddresses, cloud.Addresses) {
		t.Errorf("Unexpected diff of node addresses: %v", cmp.Diff(nodeAddresses, cloud.Addresses))
	}

	// Change the IP address
	cloud.SetNodeAddresses(createNodeInternalIPAddress("10.0.1.13"))

	// Wait until the IP address changes
	maxRetry := 5
	for i := 0; i < maxRetry; i++ {
		nodeAddresses, err := manager.NodeAddresses()
		t.Logf("nodeAddresses: %#v, err: %v", nodeAddresses, err)
		if err != nil {
			t.Errorf("Unexpected err: %q\n", err)
		}
		// It is safe to read cloud.Addresses since no routine is changing the value at the same time
		if err == nil && nodeAddresses[0].Address != cloud.Addresses[0].Address {
			time.Sleep(syncPeriod)
			continue
		}
		if err != nil {
			t.Errorf("Unexpected err: %q\n", err)
		}
		return
	}
	t.Errorf("Timeout waiting for %q address to appear", cloud.Addresses[0].Address)
}

func TestNodeAddressesUsesLastSuccess(t *testing.T) {
	cloud := &fake.Cloud{}
	manager := NewSyncManager(cloud, "defaultNode", 0).(*cloudResourceSyncManager)

	// These tests are stateful and order dependent.
	tests := []struct {
		name                   string
		addrs                  []v1.NodeAddress
		err                    error
		wantAddrs              []v1.NodeAddress
		wantErr                bool
		shouldDisableInstances bool
	}{
		{
			name:    "first sync loop encounters an error",
			err:     errors.New("bad"),
			wantErr: true,
		},
		{
			name:                   "failed to get instances from cloud provider",
			err:                    errors.New("failed to get instances from cloud provider"),
			wantErr:                true,
			shouldDisableInstances: true,
		},
		{
			name:      "subsequent sync loop succeeds",
			addrs:     createNodeInternalIPAddress("10.0.1.12"),
			wantAddrs: createNodeInternalIPAddress("10.0.1.12"),
		},
		{
			name:      "subsequent sync loop encounters an error, last addresses returned",
			err:       errors.New("bad"),
			wantAddrs: createNodeInternalIPAddress("10.0.1.12"),
		},
		{
			name:      "subsequent sync loop succeeds changing addresses",
			addrs:     createNodeInternalIPAddress("10.0.1.13"),
			wantAddrs: createNodeInternalIPAddress("10.0.1.13"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cloud.Addresses = test.addrs
			cloud.Err = test.err

			if test.shouldDisableInstances {
				cloud.DisableInstances = true
				defer func() {
					cloud.DisableInstances = false
				}()
			}

			manager.syncNodeAddresses()
			nodeAddresses, err := manager.NodeAddresses()
			if (err != nil) != test.wantErr {
				t.Errorf("unexpected err: %v", err)
			}
			if got, want := nodeAddresses, test.wantAddrs; !reflect.DeepEqual(got, want) {
				t.Errorf("Unexpected diff of node addresses: %v", cmp.Diff(got, want))
			}
		})
	}
}
