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

package ipvs

import (
	"net"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
)

// getNodeIPs returns a set of all node IP addresses.
func getNodeIPs(isIPv6 bool) (sets.String, error) {
	res := sets.NewString()

	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return nil, err
	}
	for _, addr := range addrs {
		ip, _, _ := net.ParseCIDR(addr.String())
		if isIPv6 {
			if ip.To4() == nil {
				res.Insert(ip.String())
			}
		} else {
			if ip.To4() != nil {
				res.Insert(ip.String())
			}
		}
	}

	return res, nil
}

func TestGetLocalAddresses(t *testing.T) {
	isIPv6 := false
	h := NewNetLinkHandle(isIPv6)
	nodeIPSet, err := getNodeIPs(isIPv6)
	if err != nil {
		t.Errorf("GetLocalAddresses() error get node ip: %v", err)
	}

	tests := []struct {
		testName  string
		dev       string
		filterDev string
		want      sets.String
		wantErr   bool
	}{
		{
			testName: "invalid dev",
			dev:      "foobar",
			wantErr:  true,
		},
		{
			testName: "invalid filterDev",
			dev:      "foobar",
			wantErr:  true,
		},
		{
			testName: "lo as dev",
			dev:      "lo",
			want:     sets.NewString("127.0.0.1"),
		},
		{
			testName:  "all network interfaces",
			dev:       "",
			filterDev: "",
			want:      nodeIPSet,
		},
		{
			testName:  "all network interfaces except lo",
			dev:       "",
			filterDev: "lo",
			want:      nodeIPSet.Difference(sets.NewString("127.0.0.1")),
		},
		{
			testName:  "dev and filterDev are the same",
			dev:       "lo",
			filterDev: "lo",
		},
	}
	for _, test := range tests {
		t.Run(test.testName, func(t *testing.T) {
			got, err := h.GetLocalAddresses(test.dev, test.filterDev)
			if (err != nil) != test.wantErr {
				t.Errorf("GetLocalAddresses() error = %v, wantErr %v", err, test.wantErr)
				return
			}
			if !got.Equal(test.want) {
				t.Errorf("GetLocalAddresses() got = %v, want %v", got, test.want)
			}
		})
	}
}
