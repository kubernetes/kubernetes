/*
Copyright 2014 The Kubernetes Authors.

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

package endpoint

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
)

func TestSelectableFieldLabelConversions(t *testing.T) {
	_, fieldsSet, err := EndpointsAttributes(&api.Endpoints{})
	if err != nil {
		t.Fatal(err)
	}
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		testapi.Default.GroupVersion().String(),
		"Endpoints",
		fieldsSet,
		nil,
	)
}

func newEndpoint() *api.Endpoints {
	nodeName := "kubernetes-minion-wrong"
	ep := &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Subsets: []api.EndpointSubset{
			{
				NotReadyAddresses: []api.EndpointAddress{},
				Ports:             []api.EndpointPort{},
				Addresses: []api.EndpointAddress{
					{
						IP:       "8.8.8.8",
						Hostname: "blah.com",
						NodeName: &nodeName}}}}}
	return ep
}

func TestEndpointAddressNodeNameCreateRestrictions(t *testing.T) {
	ctx := api.NewDefaultContext()
	endpoint := newEndpoint()
	Strategy.PrepareForCreate(ctx, endpoint)
	// Check that NodeName cannot be specified during create
	if endpoint.Subsets[0].Addresses[0].NodeName != nil {
		t.Error("Endpoint should not allow setting Subset.Addresses.NodeName on create")
	}
}

func TestEndpointAddressNodeNameUpdateRestrictions(t *testing.T) {
	ctx := api.NewDefaultContext()
	oldEndpoint := newEndpoint()
	goodNodeName := "kubernetes-minion-setup-by-backend"
	oldEndpoint.Subsets[0].Addresses[0].NodeName = &goodNodeName
	updatedEndpoint := newEndpoint()
	// Check that NodeName cannot be changed during update (if already set)
	Strategy.PrepareForUpdate(ctx, updatedEndpoint, oldEndpoint)
	if updatedEndpoint.Subsets[0].Addresses[0].NodeName == nil || *updatedEndpoint.Subsets[0].Addresses[0].NodeName != goodNodeName {
		t.Error("Endpoint should not allow changing of Subset.Addresses.NodeName on update")
	}
	// Check that NodeName can be changed during update (if nil)
	oldEndpoint.Subsets[0].Addresses[0].NodeName = nil
	updatedEndpoint = newEndpoint()
	updatedEndpoint.Subsets[0].Addresses[0].NodeName = &goodNodeName
	Strategy.PrepareForUpdate(ctx, updatedEndpoint, oldEndpoint)
	if updatedEndpoint.Subsets[0].Addresses[0].NodeName == nil || *updatedEndpoint.Subsets[0].Addresses[0].NodeName != goodNodeName {
		t.Error("Endpoint should allow changing of Subset.Addresses.NodeName on update if previous value was nil")
	}
}
