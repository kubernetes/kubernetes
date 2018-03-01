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

package ipamperf

import (
	"context"
	"net"
	"sync"

	beta "google.golang.org/api/compute/v0.beta"
	ga "google.golang.org/api/compute/v1"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam/cidrset"
	"k8s.io/kubernetes/test/integration/util"
)

// implemntation note:
// ------------------
// cloud.go implements hooks and handler functions for the MockGCE cloud in order to meet expectations
// of cloud behavior from the IPAM controllers. The key constraint is that the IPAM code is spread
// across both GA and Beta instances, which are distinct objects in the mock. We need to solve for
//
// 1. When a GET is called on an instance, we lazy create the instance with or without an assigned
//    ip alias as needed by the IPAM controller type
// 2. When we assign an IP alias for an instance, both the GA and Beta instance have to agree on the
//    assigned alias range
//
// We solve both the problems by using a baseInstanceList which maintains a list of known instances,
// and their pre-assigned ip-alias ranges (if needed). We then create GetHook for GA and Beta GetInstance
// calls as closures over this betaInstanceList that can lookup base instance data.
//
// This has the advantage that once the Get hook pouplates the GCEMock with the base data, we then let the
// rest of the mock code run as is.

// baseInstance tracks basic instance data needed by the IPAM controllers
type baseInstance struct {
	name       string
	zone       string
	aliasRange string
}

// baseInstanceList tracks a set of base instances
type baseInstanceList struct {
	allocateCIDR   bool
	clusterCIDR    *net.IPNet
	subnetMaskSize int
	cidrSet        *cidrset.CidrSet

	lock      sync.Mutex // protect access to instances
	instances map[meta.Key]*baseInstance
}

// toGA is an utility method to return the baseInstance data as a GA Instance object
func (bi *baseInstance) toGA() *ga.Instance {
	inst := &ga.Instance{Name: bi.name, Zone: bi.zone, NetworkInterfaces: []*ga.NetworkInterface{{}}}
	if bi.aliasRange != "" {
		inst.NetworkInterfaces[0].AliasIpRanges = []*ga.AliasIpRange{
			{IpCidrRange: bi.aliasRange, SubnetworkRangeName: util.TestSecondaryRangeName},
		}
	}
	return inst
}

// toGA is an utility method to return the baseInstance data as a beta Instance object
func (bi *baseInstance) toBeta() *beta.Instance {
	inst := &beta.Instance{Name: bi.name, Zone: bi.zone, NetworkInterfaces: []*beta.NetworkInterface{{}}}
	if bi.aliasRange != "" {
		inst.NetworkInterfaces[0].AliasIpRanges = []*beta.AliasIpRange{
			{IpCidrRange: bi.aliasRange, SubnetworkRangeName: util.TestSecondaryRangeName},
		}
	}
	return inst
}

// newBaseInstanceList is the baseInstanceList constructor
func newBaseInstanceList(allocateCIDR bool, clusterCIDR *net.IPNet, subnetMaskSize int) *baseInstanceList {
	cidrSet, _ := cidrset.NewCIDRSet(clusterCIDR, subnetMaskSize)
	return &baseInstanceList{
		allocateCIDR:   allocateCIDR,
		clusterCIDR:    clusterCIDR,
		subnetMaskSize: subnetMaskSize,
		cidrSet:        cidrSet,
		instances:      make(map[meta.Key]*baseInstance),
	}
}

// getOrCreateBaseInstance lazily creates a new base instance, assigning if allocateCIDR is true
func (bil *baseInstanceList) getOrCreateBaseInstance(key *meta.Key) *baseInstance {
	bil.lock.Lock()
	defer bil.lock.Unlock()

	inst, found := bil.instances[*key]
	if !found {
		inst = &baseInstance{name: key.Name, zone: key.Zone}
		if bil.allocateCIDR {
			nextRange, _ := bil.cidrSet.AllocateNext()
			inst.aliasRange = nextRange.String()
		}
		bil.instances[*key] = inst
	}
	return inst
}

// newGAGetHook creates a new closure with the current baseInstanceList to be used as a MockInstances.GetHook
func (bil *baseInstanceList) newGAGetHook() func(ctx context.Context, key *meta.Key, m *cloud.MockInstances) (bool, *ga.Instance, error) {
	return func(ctx context.Context, key *meta.Key, m *cloud.MockInstances) (bool, *ga.Instance, error) {
		m.Lock.Lock()
		defer m.Lock.Unlock()

		if _, found := m.Objects[*key]; !found {
			m.Objects[*key] = &cloud.MockInstancesObj{Obj: bil.getOrCreateBaseInstance(key).toGA()}
		}
		return false, nil, nil
	}
}

// newBetaGetHook creates a new closure with the current baseInstanceList to be used as a MockBetaInstances.GetHook
func (bil *baseInstanceList) newBetaGetHook() func(ctx context.Context, key *meta.Key, m *cloud.MockBetaInstances) (bool, *beta.Instance, error) {
	return func(ctx context.Context, key *meta.Key, m *cloud.MockBetaInstances) (bool, *beta.Instance, error) {
		m.Lock.Lock()
		defer m.Lock.Unlock()

		if _, found := m.Objects[*key]; !found {
			m.Objects[*key] = &cloud.MockInstancesObj{Obj: bil.getOrCreateBaseInstance(key).toBeta()}
		}
		return false, nil, nil
	}
}

// newMockCloud returns a mock GCE instance with the appropriate handlers hooks
func (bil *baseInstanceList) newMockCloud() cloud.Cloud {
	c := cloud.NewMockGCE(nil)

	// insert hooks to lazy create a instance when needed
	c.MockInstances.GetHook = bil.newGAGetHook()
	c.MockBetaInstances.GetHook = bil.newBetaGetHook()

	return c
}
