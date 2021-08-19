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

package certificate

import (
	"net"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	netutils "k8s.io/utils/net"
)

func TestAddressesToHostnamesAndIPs(t *testing.T) {
	tests := []struct {
		name         string
		addresses    []v1.NodeAddress
		wantDNSNames []string
		wantIPs      []net.IP
	}{
		{
			name:         "empty",
			addresses:    nil,
			wantDNSNames: nil,
			wantIPs:      nil,
		},
		{
			name:         "ignore empty values",
			addresses:    []v1.NodeAddress{{Type: v1.NodeHostName, Address: ""}},
			wantDNSNames: nil,
			wantIPs:      nil,
		},
		{
			name: "ignore invalid IPs",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.2"},
				{Type: v1.NodeExternalIP, Address: "3.4"},
			},
			wantDNSNames: nil,
			wantIPs:      nil,
		},
		{
			name: "dedupe values",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeHostName, Address: "hostname"},
				{Type: v1.NodeExternalDNS, Address: "hostname"},
				{Type: v1.NodeInternalDNS, Address: "hostname"},
				{Type: v1.NodeInternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
			},
			wantDNSNames: []string{"hostname"},
			wantIPs:      []net.IP{netutils.ParseIPSloppy("1.1.1.1")},
		},
		{
			name: "order values",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeHostName, Address: "hostname-2"},
				{Type: v1.NodeExternalDNS, Address: "hostname-1"},
				{Type: v1.NodeInternalDNS, Address: "hostname-3"},
				{Type: v1.NodeInternalIP, Address: "2.2.2.2"},
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "3.3.3.3"},
			},
			wantDNSNames: []string{"hostname-1", "hostname-2", "hostname-3"},
			wantIPs:      []net.IP{netutils.ParseIPSloppy("1.1.1.1"), netutils.ParseIPSloppy("2.2.2.2"), netutils.ParseIPSloppy("3.3.3.3")},
		},
		{
			name: "handle IP and DNS hostnames",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeHostName, Address: "hostname"},
				{Type: v1.NodeHostName, Address: "1.1.1.1"},
			},
			wantDNSNames: []string{"hostname"},
			wantIPs:      []net.IP{netutils.ParseIPSloppy("1.1.1.1")},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotDNSNames, gotIPs := addressesToHostnamesAndIPs(tt.addresses)
			if !reflect.DeepEqual(gotDNSNames, tt.wantDNSNames) {
				t.Errorf("addressesToHostnamesAndIPs() gotDNSNames = %v, want %v", gotDNSNames, tt.wantDNSNames)
			}
			if !reflect.DeepEqual(gotIPs, tt.wantIPs) {
				t.Errorf("addressesToHostnamesAndIPs() gotIPs = %v, want %v", gotIPs, tt.wantIPs)
			}
		})
	}
}
