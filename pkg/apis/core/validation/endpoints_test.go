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

package validation

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	_ "k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestValidateEndpoints(t *testing.T) {
	successCases := map[string]core.Endpoints{
		"simple endpoint": {
			ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
			Subsets: []core.EndpointSubset{
				{
					Addresses: []core.EndpointAddress{{IP: "10.10.1.1"}, {IP: "10.10.2.2"}},
					Ports:     []core.EndpointPort{{Name: "a", Port: 8675, Protocol: "TCP"}, {Name: "b", Port: 309, Protocol: "TCP"}},
				},
				{
					Addresses: []core.EndpointAddress{{IP: "10.10.3.3"}},
					Ports:     []core.EndpointPort{{Name: "a", Port: 93, Protocol: "TCP"}, {Name: "b", Port: 76, Protocol: "TCP"}},
				},
			},
		},
		"empty subsets": {
			ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
		},
		"no name required for singleton port": {
			ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
			Subsets: []core.EndpointSubset{
				{
					Addresses: []core.EndpointAddress{{IP: "10.10.1.1"}},
					Ports:     []core.EndpointPort{{Port: 8675, Protocol: "TCP"}},
				},
			},
		},
		"empty ports": {
			ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
			Subsets: []core.EndpointSubset{
				{
					Addresses: []core.EndpointAddress{{IP: "10.10.3.3"}},
				},
			},
		},
	}

	for k, v := range successCases {
		if errs := ValidateEndpoints(&v); len(errs) != 0 {
			t.Errorf("Expected success for %s, got %v", k, errs)
		}
	}

	errorCases := map[string]struct {
		endpoints   core.Endpoints
		errorType   field.ErrorType
		errorDetail string
	}{
		"missing namespace": {
			endpoints: core.Endpoints{ObjectMeta: metav1.ObjectMeta{Name: "mysvc"}},
			errorType: "FieldValueRequired",
		},
		"missing name": {
			endpoints: core.Endpoints{ObjectMeta: metav1.ObjectMeta{Namespace: "namespace"}},
			errorType: "FieldValueRequired",
		},
		"invalid namespace": {
			endpoints:   core.Endpoints{ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "no@#invalid.;chars\"allowed"}},
			errorType:   "FieldValueInvalid",
			errorDetail: dnsLabelErrMsg,
		},
		"invalid name": {
			endpoints:   core.Endpoints{ObjectMeta: metav1.ObjectMeta{Name: "-_Invliad^&Characters", Namespace: "namespace"}},
			errorType:   "FieldValueInvalid",
			errorDetail: dnsSubdomainLabelErrMsg,
		},
		"empty addresses": {
			endpoints: core.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []core.EndpointSubset{
					{
						Ports: []core.EndpointPort{{Name: "a", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType: "FieldValueRequired",
		},
		"invalid IP": {
			endpoints: core.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []core.EndpointSubset{
					{
						Addresses: []core.EndpointAddress{{IP: "[2001:0db8:85a3:0042:1000:8a2e:0370:7334]"}},
						Ports:     []core.EndpointPort{{Name: "a", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "must be a valid IP address",
		},
		"Multiple ports, one without name": {
			endpoints: core.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []core.EndpointSubset{
					{
						Addresses: []core.EndpointAddress{{IP: "10.10.1.1"}},
						Ports:     []core.EndpointPort{{Port: 8675, Protocol: "TCP"}, {Name: "b", Port: 309, Protocol: "TCP"}},
					},
				},
			},
			errorType: "FieldValueRequired",
		},
		"Invalid port number": {
			endpoints: core.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []core.EndpointSubset{
					{
						Addresses: []core.EndpointAddress{{IP: "10.10.1.1"}},
						Ports:     []core.EndpointPort{{Name: "a", Port: 66000, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "between",
		},
		"Invalid protocol": {
			endpoints: core.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []core.EndpointSubset{
					{
						Addresses: []core.EndpointAddress{{IP: "10.10.1.1"}},
						Ports:     []core.EndpointPort{{Name: "a", Port: 93, Protocol: "Protocol"}},
					},
				},
			},
			errorType: "FieldValueNotSupported",
		},
		"Address missing IP": {
			endpoints: core.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []core.EndpointSubset{
					{
						Addresses: []core.EndpointAddress{{}},
						Ports:     []core.EndpointPort{{Name: "a", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "must be a valid IP address",
		},
		"Port missing number": {
			endpoints: core.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []core.EndpointSubset{
					{
						Addresses: []core.EndpointAddress{{IP: "10.10.1.1"}},
						Ports:     []core.EndpointPort{{Name: "a", Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "between",
		},
		"Port missing protocol": {
			endpoints: core.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []core.EndpointSubset{
					{
						Addresses: []core.EndpointAddress{{IP: "10.10.1.1"}},
						Ports:     []core.EndpointPort{{Name: "a", Port: 93}},
					},
				},
			},
			errorType: "FieldValueRequired",
		},
		"Address is loopback": {
			endpoints: core.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []core.EndpointSubset{
					{
						Addresses: []core.EndpointAddress{{IP: "127.0.0.1"}},
						Ports:     []core.EndpointPort{{Name: "p", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "loopback",
		},
		"Address is link-local": {
			endpoints: core.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []core.EndpointSubset{
					{
						Addresses: []core.EndpointAddress{{IP: "169.254.169.254"}},
						Ports:     []core.EndpointPort{{Name: "p", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "link-local",
		},
		"Address is link-local multicast": {
			endpoints: core.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []core.EndpointSubset{
					{
						Addresses: []core.EndpointAddress{{IP: "224.0.0.1"}},
						Ports:     []core.EndpointPort{{Name: "p", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "link-local multicast",
		},
	}

	for k, v := range errorCases {
		if errs := ValidateEndpoints(&v.endpoints); len(errs) == 0 || errs[0].Type != v.errorType || !strings.Contains(errs[0].Detail, v.errorDetail) {
			t.Errorf("[%s] Expected error type %s with detail %q, got %v", k, v.errorType, v.errorDetail, errs)
		}
	}
}

func TestEndpointAddressNodeNameUpdateRestrictions(t *testing.T) {
	oldEndpoint := newNodeNameEndpoint("kubernetes-node-setup-by-backend")
	updatedEndpoint := newNodeNameEndpoint("kubernetes-changed-nodename")
	// Check that NodeName cannot be changed during update (if already set)
	errList := ValidateEndpoints(updatedEndpoint)
	errList = append(errList, ValidateEndpointsUpdate(updatedEndpoint, oldEndpoint)...)
	if len(errList) == 0 {
		t.Error("Endpoint should not allow changing of Subset.Addresses.NodeName on update")
	}
}

func TestEndpointAddressNodeNameInvalidDNSSubdomain(t *testing.T) {
	// Check NodeName DNS validation
	endpoint := newNodeNameEndpoint("illegal*.nodename")
	errList := ValidateEndpoints(endpoint)
	if len(errList) == 0 {
		t.Error("Endpoint should reject invalid NodeName")
	}
}

func TestEndpointAddressNodeNameCanBeAnIPAddress(t *testing.T) {
	endpoint := newNodeNameEndpoint("10.10.1.1")
	errList := ValidateEndpoints(endpoint)
	if len(errList) != 0 {
		t.Error("Endpoint should accept a NodeName that is an IP address")
	}
}

func newNodeNameEndpoint(nodeName string) *core.Endpoints {
	ep := &core.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "1",
		},
		Subsets: []core.EndpointSubset{
			{
				NotReadyAddresses: []core.EndpointAddress{},
				Ports:             []core.EndpointPort{{Name: "https", Port: 443, Protocol: "TCP"}},
				Addresses: []core.EndpointAddress{
					{
						IP:       "8.8.8.8",
						Hostname: "zookeeper1",
						NodeName: &nodeName}}}}}
	return ep
}
