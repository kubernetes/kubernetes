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

package azure

import (
	"fmt"
	"testing"

	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/stretchr/testify/assert"
)

func TestFindProbe(t *testing.T) {
	tests := []struct {
		msg           string
		existingProbe []network.Probe
		curProbe      network.Probe
		expected      bool
	}{
		{
			msg:      "empty existing probes should return false",
			expected: false,
		},
		{
			msg: "probe names match while ports unmatch should return false",
			existingProbe: []network.Probe{
				{
					Name: to.StringPtr("httpProbe"),
					ProbePropertiesFormat: &network.ProbePropertiesFormat{
						Port: to.Int32Ptr(1),
					},
				},
			},
			curProbe: network.Probe{
				Name: to.StringPtr("httpProbe"),
				ProbePropertiesFormat: &network.ProbePropertiesFormat{
					Port: to.Int32Ptr(2),
				},
			},
			expected: false,
		},
		{
			msg: "probe ports match while names unmatch should return false",
			existingProbe: []network.Probe{
				{
					Name: to.StringPtr("probe1"),
					ProbePropertiesFormat: &network.ProbePropertiesFormat{
						Port: to.Int32Ptr(1),
					},
				},
			},
			curProbe: network.Probe{
				Name: to.StringPtr("probe2"),
				ProbePropertiesFormat: &network.ProbePropertiesFormat{
					Port: to.Int32Ptr(1),
				},
			},
			expected: false,
		},
		{
			msg: "both probe ports and names match should return true",
			existingProbe: []network.Probe{
				{
					Name: to.StringPtr("matchName"),
					ProbePropertiesFormat: &network.ProbePropertiesFormat{
						Port: to.Int32Ptr(1),
					},
				},
			},
			curProbe: network.Probe{
				Name: to.StringPtr("matchName"),
				ProbePropertiesFormat: &network.ProbePropertiesFormat{
					Port: to.Int32Ptr(1),
				},
			},
			expected: true,
		},
	}

	for i, test := range tests {
		findResult := findProbe(test.existingProbe, test.curProbe)
		assert.Equal(t, test.expected, findResult, fmt.Sprintf("TestCase[%d]: %s", i, test.msg))
	}
}
