/*
Copyright 2019 The Kubernetes Authors.

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

package metaproxier

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"

	v1 "k8s.io/api/core/v1"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
)

func Test_endpointsIPFamily(t *testing.T) {

	ipv4 := v1.IPv4Protocol
	ipv6 := v1.IPv6Protocol

	tests := []struct {
		name      string
		endpoints *v1.Endpoints
		want      *v1.IPFamily
		wantErr   bool
		errorMsg  string
	}{
		{
			name:      "Endpoints No Subsets",
			endpoints: &v1.Endpoints{},
			want:      nil,
			wantErr:   true,
			errorMsg:  "failed to identify ipfamily for endpoints (no subsets)",
		},
		{
			name:      "Endpoints No Addresses",
			endpoints: &v1.Endpoints{Subsets: []v1.EndpointSubset{{NotReadyAddresses: []v1.EndpointAddress{}}}},
			want:      nil,
			wantErr:   true,
			errorMsg:  "failed to identify ipfamily for endpoints (no addresses)",
		},
		{
			name:      "Endpoints Address Has No IP",
			endpoints: &v1.Endpoints{Subsets: []v1.EndpointSubset{{Addresses: []v1.EndpointAddress{{Hostname: "testhost", IP: ""}}}}},
			want:      nil,
			wantErr:   true,
			errorMsg:  "failed to identify ipfamily for endpoints (address has no ip)",
		},
		{
			name:      "Endpoints Address IPv4",
			endpoints: &v1.Endpoints{Subsets: []v1.EndpointSubset{{Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}}}}},
			want:      &ipv4,
			wantErr:   false,
		},
		{
			name:      "Endpoints Address IPv6",
			endpoints: &v1.Endpoints{Subsets: []v1.EndpointSubset{{Addresses: []v1.EndpointAddress{{IP: "2001:db9::2"}}}}},
			want:      &ipv6,
			wantErr:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := endpointsIPFamily(tt.endpoints)
			if (err != nil) != tt.wantErr {
				t.Errorf("endpointsIPFamily() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err != nil && err.Error() != tt.errorMsg {
				t.Errorf("endpointsIPFamily() error = %v, wantErr %v", err, tt.errorMsg)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("endpointsIPFamily() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAltFamilyPortIdentification(t *testing.T) {
	nodeIPv4 := "172.0.0.4"
	nodeIPv6 := "3000::1"

	testCases := []struct {
		name             string
		services         []v1.Service
		expectedPortList []localPortWithFamily
	}{
		{
			name: "service of each type",
			expectedPortList: []localPortWithFamily{
				{
					/* covering old v4 service */
					family: v1.IPv6Protocol,
					localPort: utilproxy.LocalPort{
						IP:       nodeIPv6,
						Port:     4000,
						Protocol: "tcp",
					},
				},

				{
					/* covering old v6 service */
					family: v1.IPv4Protocol,
					localPort: utilproxy.LocalPort{
						IP:       nodeIPv4,
						Port:     6000,
						Protocol: "tcp",
					},
				},

				{
					/* covering v4 service */
					family: v1.IPv6Protocol,
					localPort: utilproxy.LocalPort{
						IP:       nodeIPv6,
						Port:     4100,
						Protocol: "tcp",
					},
				},

				{
					/* covering v6 service */
					family: v1.IPv4Protocol,
					localPort: utilproxy.LocalPort{
						IP:       nodeIPv4,
						Port:     6100,
						Protocol: "tcp",
					},
				},
			},
			services: []v1.Service{
				{
					/* old v4 service */
					Spec: v1.ServiceSpec{
						ClusterIP: "10.0.0.10",
						Ports: []v1.ServicePort{
							{
								Protocol: v1.ProtocolTCP,
								NodePort: 4000,
							},
						},
					},
				},

				{
					/* old v6 service */
					Spec: v1.ServiceSpec{
						ClusterIP: "2000::1",
						Ports: []v1.ServicePort{
							{
								Protocol: v1.ProtocolTCP,
								NodePort: 6000,
							},
						},
					},
				},

				{
					/* v4 service */
					Spec: v1.ServiceSpec{
						IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
						Ports: []v1.ServicePort{
							{
								Protocol: v1.ProtocolTCP,
								NodePort: 4100,
							},
						},
					},
				},

				{
					/* v6 service */
					Spec: v1.ServiceSpec{
						IPFamilies: []v1.IPFamily{v1.IPv6Protocol},
						Ports: []v1.ServicePort{
							{
								Protocol: v1.ProtocolTCP,
								NodePort: 6100,
							},
						},
					},
				},

				{
					/* v4,v6 service */
					Spec: v1.ServiceSpec{
						IPFamilies: []v1.IPFamily{v1.IPv4Protocol, v1.IPv6Protocol},
						Ports: []v1.ServicePort{
							{
								Protocol: v1.ProtocolTCP,
								NodePort: 4161,
							},
						},
					},
				},
			},
		},
		/* add more if needed */
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			nodeIPs := sets.NewString(nodeIPv4, nodeIPv6)

			metaProxy := &metaProxier{}
			metaProxy.nodeAddresses = nodeIPs
			metaProxy.services = make(map[string]*v1.Service)

			// add services
			for i, s := range testCase.services {
				cpy := (&s).DeepCopy()
				metaProxy.services[fmt.Sprintf("idx:%d", i)] = cpy
			}

			// get ports
			ports := metaProxy.getLocalPortsForServices()
			if len(ports) != len(testCase.expectedPortList) {
				t.Fatalf("expected %v ports got %v ports", len(testCase.expectedPortList), len(ports))
			}

			t.Logf("%+v", ports)

			for _, expectedPort := range testCase.expectedPortList {
				found := false
				for _, currentPort := range ports {
					if currentPort.localPort.IP == expectedPort.localPort.IP &&
						currentPort.localPort.Port == expectedPort.localPort.Port &&
						currentPort.localPort.Protocol == expectedPort.localPort.Protocol &&
						currentPort.family == expectedPort.family {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("port %+v was expected but was not found", expectedPort)
				}
			}

		})
	}
}
