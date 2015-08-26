/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package network

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"
)

func TestNetworkStrategy(t *testing.T) {
	ctx := api.NewDefaultContext()
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("Networks should not allow create on update")
	}
	network := &api.Network{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "10"},
		Status:     api.NetworkStatus{Phase: api.NetworkTerminating},
		Spec: api.NetworkSpec{
			TenantID: "12345",
			Subnets:  map[string]api.Subnet{"s1": {CIDR: "192.168.0.0/24", Gateway: "192.168.0.1"}}},
	}
	Strategy.PrepareForCreate(network)
	if network.Status.Phase != api.NetworkInitializing {
		t.Errorf("Networks do not allow setting phase on create")
	}
	errs := Strategy.Validate(ctx, network)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	invalidNetwork := &api.Network{
		ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "4"},
		Spec:       api.NetworkSpec{Subnets: map[string]api.Subnet{"s1": {CIDR: "10.10.10.0/24", Gateway: "192.168.0.1"}}},
	}
	errs = Strategy.ValidateUpdate(ctx, invalidNetwork, network)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	if invalidNetwork.ResourceVersion != "4" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}

func TestNetworkStatusStrategy(t *testing.T) {
	ctx := api.NewDefaultContext()
	if StatusStrategy.AllowCreateOnUpdate() {
		t.Errorf("Networks should not allow create on update")
	}
	now := util.Now()
	oldNetwork := &api.Network{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "10"},
		Spec: api.NetworkSpec{
			TenantID: "12345",
			Subnets:  map[string]api.Subnet{"s1": {CIDR: "192.168.0.0/24", Gateway: "192.168.0.1"}}},
		Status: api.NetworkStatus{Phase: api.NetworkActive},
	}
	network := &api.Network{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "9", DeletionTimestamp: &now},
		Spec: api.NetworkSpec{
			TenantID: "12345",
			Subnets:  map[string]api.Subnet{"s1": {CIDR: "192.168.0.0/24", Gateway: "192.168.0.1"}}},
		Status: api.NetworkStatus{Phase: api.NetworkTerminating},
	}
	StatusStrategy.PrepareForUpdate(network, oldNetwork)
	if network.Status.Phase != api.NetworkTerminating {
		t.Errorf("Network status updates should allow change of phase: %v", network.Status.Phase)
	}
	errs := StatusStrategy.ValidateUpdate(ctx, network, oldNetwork)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}
	if network.ResourceVersion != "9" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}
