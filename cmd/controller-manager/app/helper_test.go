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

package app

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/component-base/featuregate/testing"
	kubefeatures "k8s.io/kubernetes/pkg/features"

	"github.com/stretchr/testify/assert"
)

func TestIsControllerEnabled(t *testing.T) {
	tcs := []struct {
		name                         string
		controllerName               string
		controllers                  []string
		disabledByDefaultControllers []string
		expected                     bool
	}{
		{
			name:                         "on by name",
			controllerName:               "bravo",
			controllers:                  []string{"alpha", "bravo", "-charlie"},
			disabledByDefaultControllers: []string{"delta", "echo"},
			expected:                     true,
		},
		{
			name:                         "off by name",
			controllerName:               "charlie",
			controllers:                  []string{"alpha", "bravo", "-charlie"},
			disabledByDefaultControllers: []string{"delta", "echo"},
			expected:                     false,
		},
		{
			name:                         "on by default",
			controllerName:               "alpha",
			controllers:                  []string{"*"},
			disabledByDefaultControllers: []string{"delta", "echo"},
			expected:                     true,
		},
		{
			name:                         "off by default",
			controllerName:               "delta",
			controllers:                  []string{"*"},
			disabledByDefaultControllers: []string{"delta", "echo"},
			expected:                     false,
		},
		{
			name:                         "off by default implicit, no star",
			controllerName:               "foxtrot",
			controllers:                  []string{"alpha", "bravo", "-charlie"},
			disabledByDefaultControllers: []string{"delta", "echo"},
			expected:                     false,
		},
	}

	for _, tc := range tcs {
		actual := IsControllerEnabled(tc.controllerName, sets.NewString(tc.disabledByDefaultControllers...), tc.controllers)
		assert.Equal(t, tc.expected, actual, "%v: expected %v, got %v", tc.name, tc.expected, actual)
	}

}

func TestParseClusterCIDRs(t *testing.T) {
	tcs := []struct {
		name       string
		cidrs      string
		nets       []string
		ipv6Enable bool
		valid      bool
	}{
		{
			name:  "invalid cidr",
			cidrs: "127.0.0.1",
			nets:  []string{},
			valid: false,
		},
		{
			name:  "single IPv4 cidr",
			cidrs: "192.168.0.0/24",
			nets:  []string{"192.168.0.0/24"},
			valid: true,
		},
		{
			name:  "dual cidrs without feature gate",
			cidrs: "172.10.0.0/24,2000::/3",
			nets:  []string{},
			valid: false,
		},
		// set the feature gate
		{
			name:       "dual IPv4 cidrs",
			cidrs:      "192.168.0.0/24,172.10.0.0/24",
			nets:       []string{},
			ipv6Enable: true,
			valid:      false,
		},
		{
			name:       "dual cidrs",
			cidrs:      "172.10.0.0/24,2000::/3",
			nets:       []string{"172.10.0.0/24", "2000::/3"},
			ipv6Enable: true,
			valid:      true,
		},
		{
			name:       "trinal cidrs",
			cidrs:      "192.168.0.0/24,172.10.0.0/24,2000::/3",
			nets:       []string{},
			ipv6Enable: true,
			valid:      false,
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			if tc.ipv6Enable {
				defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.IPv6DualStack, true)()
			}
			nets, err := ParseClusterCIDRs(tc.cidrs)
			valid := err == nil
			marshaledNets := make([]string, 0)
			for _, n := range nets {
				nb := fmt.Sprintf("%v", n)
				marshaledNets = append(marshaledNets, nb)
			}
			assert.Equal(t, tc.valid, valid, "validation should be %v, but got %v", tc.valid, valid)
			assert.Equal(t, tc.nets, marshaledNets, "nets should be %v, but got %v", tc.nets, marshaledNets)
		})
	}
}
