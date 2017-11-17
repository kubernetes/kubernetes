/*
Copyright 2016 The Kubernetes Authors.

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

package master

import (
	"fmt"
	"net"
	"reflect"
	"testing"
)

func TestDefaultServiceIPRange(t *testing.T) {

	_, normalIpNet, err := net.ParseCIDR("192.168.1.0/24")
	if err != nil {
		t.Fatal(err)
	}
	_, badIpNet, err := net.ParseCIDR("192.168.1.0/30")
	if err != nil {
		t.Fatal(err)
	}
	_, defaultIpNet, err := net.ParseCIDR("10.0.0.0/24")
	if err != nil {
		t.Fatal(err)
	}
	emptyIpNet := net.IPNet{}

	testCases := []struct {
		name        string
		ipNet       net.IPNet
		resultIpNet net.IPNet
		resultIp    string
		stopError   error
	}{
		{
			name:        "normal ipNet",
			ipNet:       *normalIpNet,
			resultIpNet: *normalIpNet,
			resultIp:    "192.168.1.1",
			stopError:   nil,
		},
		{
			name:        "empty ipNet",
			ipNet:       emptyIpNet,
			resultIpNet: *defaultIpNet,
			resultIp:    "10.0.0.1",
			stopError:   nil,
		},
		{
			name:        "bad ipNet",
			ipNet:       *badIpNet,
			resultIpNet: net.IPNet{},
			resultIp:    "<nil>",
			stopError:   fmt.Errorf("the service cluster IP range must be at least %d IP addresses", 8),
		},
	}
	for _, testCase := range testCases {
		ipNet, ip, err := DefaultServiceIPRange(testCase.ipNet)

		if testCase.stopError != nil && err != nil && testCase.stopError.Error() == err.Error() {
			continue
		}

		if testCase.stopError != nil && err == nil {
			t.Errorf("case %v, expected error: %v, but get nil", testCase.name, testCase.stopError)
		}

		if testCase.stopError == nil && err != nil {
			t.Errorf("case %v, unexpected error: %v", testCase.name, err)
		}

		if !reflect.DeepEqual(testCase.resultIpNet, ipNet) {
			t.Errorf("want ipNet: %v, get: %v", testCase.resultIpNet, ipNet)
		}

		if !reflect.DeepEqual(testCase.resultIp, ip.String()) {
			t.Errorf("want ip: %v, get: %v", testCase.resultIp, ip.String())
		}
	}
}
