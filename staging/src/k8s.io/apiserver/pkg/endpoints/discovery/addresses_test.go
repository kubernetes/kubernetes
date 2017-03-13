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

package discovery

import (
	"net"
	"net/http"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
)

func TestGetServerAddressByClientCIDRs(t *testing.T) {
	publicAddressCIDRMap := []metav1.ServerAddressByClientCIDR{
		{
			ClientCIDR:    "0.0.0.0/0",
			ServerAddress: "ExternalAddress",
		},
	}
	internalAddressCIDRMap := []metav1.ServerAddressByClientCIDR{
		publicAddressCIDRMap[0],
		{
			ClientCIDR:    "10.0.0.0/24",
			ServerAddress: "serviceIP",
		},
	}
	internalIP := "10.0.0.1"
	publicIP := "1.1.1.1"
	testCases := []struct {
		Request     http.Request
		ExpectedMap []metav1.ServerAddressByClientCIDR
	}{
		{
			Request:     http.Request{},
			ExpectedMap: publicAddressCIDRMap,
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Real-Ip": {internalIP},
				},
			},
			ExpectedMap: internalAddressCIDRMap,
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Real-Ip": {publicIP},
				},
			},
			ExpectedMap: publicAddressCIDRMap,
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {internalIP},
				},
			},
			ExpectedMap: internalAddressCIDRMap,
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {publicIP},
				},
			},
			ExpectedMap: publicAddressCIDRMap,
		},

		{
			Request: http.Request{
				RemoteAddr: internalIP,
			},
			ExpectedMap: internalAddressCIDRMap,
		},
		{
			Request: http.Request{
				RemoteAddr: publicIP,
			},
			ExpectedMap: publicAddressCIDRMap,
		},
		{
			Request: http.Request{
				RemoteAddr: "invalidIP",
			},
			ExpectedMap: publicAddressCIDRMap,
		},
	}

	_, ipRange, _ := net.ParseCIDR("10.0.0.0/24")
	discoveryAddresses := DefaultAddresses{DefaultAddress: "ExternalAddress"}
	discoveryAddresses.CIDRRules = append(discoveryAddresses.CIDRRules,
		CIDRRule{IPRange: *ipRange, Address: "serviceIP"})

	for i, test := range testCases {
		if a, e := discoveryAddresses.ServerAddressByClientCIDRs(utilnet.GetClientIP(&test.Request)), test.ExpectedMap; reflect.DeepEqual(e, a) != true {
			t.Fatalf("test case %d failed. expected: %v, actual: %v", i+1, e, a)
		}
	}
}
