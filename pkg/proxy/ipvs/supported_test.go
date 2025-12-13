//go:build linux
// +build linux

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
	"fmt"
	"testing"

	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestCanUseIPVSProxier(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	testCases := []struct {
		name         string
		scheduler    string
		ipsetVersion string
		ipsetErr     error
		ipvsErr      string
		ok           bool
	}{
		{
			name:         "happy days",
			ipsetVersion: MinIPSetCheckVersion,
			ok:           true,
		},
		{
			name:         "ipset error",
			scheduler:    "",
			ipsetVersion: MinIPSetCheckVersion,
			ipsetErr:     fmt.Errorf("oops"),
			ok:           false,
		},
		{
			name:         "ipset version too low",
			scheduler:    "rr",
			ipsetVersion: "4.3.0",
			ok:           false,
		},
		{
			name:         "GetVirtualServers fail",
			ipsetVersion: MinIPSetCheckVersion,
			ipvsErr:      "GetVirtualServers",
			ok:           false,
		},
		{
			name:         "AddVirtualServer fail",
			ipsetVersion: MinIPSetCheckVersion,
			ipvsErr:      "AddVirtualServer",
			ok:           false,
		},
		{
			name:         "DeleteVirtualServer fail",
			ipsetVersion: MinIPSetCheckVersion,
			ipvsErr:      "DeleteVirtualServer",
			ok:           false,
		},
	}

	for _, tc := range testCases {
		ipvs := &fakeIpvs{tc.ipvsErr, false}
		versioner := &fakeIPSetVersioner{version: tc.ipsetVersion, err: tc.ipsetErr}
		err := CanUseIPVSProxier(ctx, ipvs, versioner, tc.scheduler)
		if (err == nil) != tc.ok {
			t.Errorf("Case [%s], expect %v, got err: %v", tc.name, tc.ok, err)
		}
	}
}
