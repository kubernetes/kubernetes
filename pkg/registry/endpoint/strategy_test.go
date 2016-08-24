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

func newEndpoint(nodeName string) *api.Endpoints {
	ep := &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "1",
		},
		Subsets: []api.EndpointSubset{
			{
				NotReadyAddresses: []api.EndpointAddress{},
				Ports:             []api.EndpointPort{{Name: "https", Port: 443, Protocol: "TCP"}},
				Addresses: []api.EndpointAddress{
					{
						IP:       "8.8.8.8",
						Hostname: "zookeeper1",
						NodeName: &nodeName}}}}}
	return ep
}

func TestEndpointAddressNodeNameUpdateRestrictions(t *testing.T) {
	ctx := api.NewDefaultContext()
	oldEndpoint := newEndpoint("kubernetes-minion-setup-by-backend")
	updatedEndpoint := newEndpoint("kubernetes-changed-nodename")
	// Check that NodeName cannot be changed during update (if already set)
	errList := Strategy.ValidateUpdate(ctx, updatedEndpoint, oldEndpoint)
	if len(errList) == 0 {
		t.Error("Endpoint should not allow changing of Subset.Addresses.NodeName on update")
	}
}

func TestEndpointAddressNodeNameInvalidDNS1123(t *testing.T) {
	ctx := api.NewDefaultContext()
	// Check NodeName DNS validation
	endpoint := newEndpoint("illegal.nodename")
	errList := Strategy.Validate(ctx, endpoint)
	if len(errList) == 0 {
		t.Error("Endpoint should reject invalid NodeName")
	}
}
