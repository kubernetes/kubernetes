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

package capabilities

import (
	"reflect"
	"testing"
)

func TestGet(t *testing.T) {
	defer ResetForTest(nil)
	defaultCap := Capabilities{
		AllowPrivileged: false,
		PrivilegedSources: PrivilegedSources{
			HostNetworkSources: []string{},
			HostPIDSources:     []string{},
			HostIPCSources:     []string{},
		},
	}

	res := Get()
	if !reflect.DeepEqual(defaultCap, res) {
		t.Fatalf("expected Capabilities: %#v, got a non-default: %#v", defaultCap, res)
	}

	cap := Capabilities{
		PrivilegedSources: PrivilegedSources{
			HostNetworkSources: []string{"A", "B"},
		},
	}
	ResetForTest(&cap)

	res = Get()
	if !reflect.DeepEqual(cap, res) {
		t.Fatalf("expected Capabilities: %#v , got a different: %#v", cap, res)
	}
}
func TestSetup(t *testing.T) {
	testCases := []struct {
		name                     string
		allowPrivileged          bool
		perConnectionBytesPerSec int64
		expectedCapabilities     Capabilities
	}{
		{
			name:                     "AllowPrivileged true with bandwidth limit",
			allowPrivileged:          true,
			perConnectionBytesPerSec: 1024,
			expectedCapabilities: Capabilities{
				AllowPrivileged:                        true,
				PerConnectionBandwidthLimitBytesPerSec: 1024,
			},
		},
		{
			name:                     "AllowPrivileged false with higher bandwidth limit",
			allowPrivileged:          false,
			perConnectionBytesPerSec: 2048,
			expectedCapabilities: Capabilities{
				AllowPrivileged:                        false,
				PerConnectionBandwidthLimitBytesPerSec: 2048,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer ResetForTest(nil)

			Setup(tc.allowPrivileged, tc.perConnectionBytesPerSec)
			res := Get()
			if !compareCapabilities(tc.expectedCapabilities, res) {
				t.Fatalf("expected Capabilities: %#v, got: %#v", tc.expectedCapabilities, res)
			}
		})
	}
}

// compareCapabilities compares two Capabilities instances
func compareCapabilities(a, b Capabilities) bool {
	if a.AllowPrivileged != b.AllowPrivileged {
		return false
	}
	return a.PerConnectionBandwidthLimitBytesPerSec == b.PerConnectionBandwidthLimitBytesPerSec
}
