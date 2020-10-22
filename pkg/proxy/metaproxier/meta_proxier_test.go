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
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
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
