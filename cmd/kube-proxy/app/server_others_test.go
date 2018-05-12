// +build !windows

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

package app

import (
	"fmt"
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/proxy/ipvs"
	"k8s.io/kubernetes/pkg/util/iptables"
)

func Test_getProxyMode(t *testing.T) {
	var cases = []struct {
		flag            string
		annotationKey   string
		annotationVal   string
		iptablesVersion string
		ipsetVersion    string
		kmods           []string
		kernelCompat    bool
		iptablesError   error
		ipsetError      error
		expected        string
	}{
		{ // flag says userspace
			flag:     "userspace",
			expected: proxyModeUserspace,
		},
		{ // flag says iptables, error detecting version
			flag:          "iptables",
			iptablesError: fmt.Errorf("oops!"),
			expected:      proxyModeUserspace,
		},
		{ // flag says iptables, version too low
			flag:            "iptables",
			iptablesVersion: "0.0.0",
			expected:        proxyModeUserspace,
		},
		{ // flag says iptables, version ok, kernel not compatible
			flag:            "iptables",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    false,
			expected:        proxyModeUserspace,
		},
		{ // flag says iptables, version ok, kernel is compatible
			flag:            "iptables",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    true,
			expected:        proxyModeIPTables,
		},
		{ // detect, error
			flag:          "",
			iptablesError: fmt.Errorf("oops!"),
			expected:      proxyModeUserspace,
		},
		{ // detect, version too low
			flag:            "",
			iptablesVersion: "0.0.0",
			expected:        proxyModeUserspace,
		},
		{ // detect, version ok, kernel not compatible
			flag:            "",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    false,
			expected:        proxyModeUserspace,
		},
		{ // detect, version ok, kernel is compatible
			flag:            "",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    true,
			expected:        proxyModeIPTables,
		},
		{ // specify ipvs, feature gateway disabled, iptables version ok, kernel is compatible
			flag:            "ipvs",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    true,
			expected:        proxyModeIPTables,
		},
		{ // specify ipvs, feature gateway disabled, iptables version too low
			flag:            "ipvs",
			iptablesVersion: "0.0.0",
			expected:        proxyModeUserspace,
		},
		{ // specify ipvs, feature gateway disabled, iptables version ok, kernel is not compatible
			flag:            "ipvs",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    false,
			expected:        proxyModeUserspace,
		},
	}
	for i, c := range cases {
		versioner := &fakeIPTablesVersioner{c.iptablesVersion, c.iptablesError}
		kcompater := &fakeKernelCompatTester{c.kernelCompat}
		ipsetver := &fakeIPSetVersioner{c.ipsetVersion, c.ipsetError}
		khandler := &fakeKernelHandler{c.kmods}
		r := getProxyMode(c.flag, versioner, khandler, ipsetver, kcompater)
		if r != c.expected {
			t.Errorf("Case[%d] Expected %q, got %q", i, c.expected, r)
		}
	}
}

// This is a coarse test, but it offers some modicum of confidence as the code is evolved.
func Test_getProxyModeEnableFeatureGateway(t *testing.T) {
	// enable IPVS feature gateway
	utilfeature.DefaultFeatureGate.Set("SupportIPVSProxyMode=true")

	var cases = []struct {
		flag            string
		iptablesVersion string
		ipsetVersion    string
		kernelCompat    bool
		iptablesError   error
		ipsetError      error
		mods            []string
		expected        string
	}{
		{ // flag says userspace
			flag:     "userspace",
			expected: proxyModeUserspace,
		},
		{ // flag says iptables, error detecting version
			flag:          "iptables",
			iptablesError: fmt.Errorf("oops!"),
			expected:      proxyModeUserspace,
		},
		{ // flag says iptables, version too low
			flag:            "iptables",
			iptablesVersion: "0.0.0",
			expected:        proxyModeUserspace,
		},
		{ // flag says iptables, version ok, kernel not compatible
			flag:            "iptables",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    false,
			expected:        proxyModeUserspace,
		},
		{ // flag says iptables, version ok, kernel is compatible
			flag:            "iptables",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    true,
			expected:        proxyModeIPTables,
		},
		{ // detect, error
			flag:          "",
			iptablesError: fmt.Errorf("oops!"),
			expected:      proxyModeUserspace,
		},
		{ // detect, version too low
			flag:            "",
			iptablesVersion: "0.0.0",
			expected:        proxyModeUserspace,
		},
		{ // detect, version ok, kernel not compatible
			flag:            "",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    false,
			expected:        proxyModeUserspace,
		},
		{ // detect, version ok, kernel is compatible
			flag:            "",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    true,
			expected:        proxyModeIPTables,
		},
		{ // detect, version ok, kernel is compatible
			flag:            "",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    true,
			expected:        proxyModeIPTables,
		},
		{ // flag says ipvs, ipset version ok, kernel modules installed
			flag:         "ipvs",
			mods:         []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack_ipv4"},
			ipsetVersion: ipvs.MinIPSetCheckVersion,
			expected:     proxyModeIPVS,
		},
		{ // flag says ipvs, ipset version too low, fallback on iptables mode
			flag:            "ipvs",
			mods:            []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack_ipv4"},
			ipsetVersion:    "0.0",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    true,
			expected:        proxyModeIPTables,
		},
		{ // flag says ipvs, bad ipset version, fallback on iptables mode
			flag:            "ipvs",
			mods:            []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack_ipv4"},
			ipsetVersion:    "a.b.c",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    true,
			expected:        proxyModeIPTables,
		},
		{ // flag says ipvs, required kernel modules are not installed, fallback on iptables mode
			flag:            "ipvs",
			mods:            []string{"foo", "bar", "baz"},
			ipsetVersion:    ipvs.MinIPSetCheckVersion,
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    true,
			expected:        proxyModeIPTables,
		},
		{ // flag says ipvs, required kernel modules are not installed, iptables version too old, fallback on userspace mode
			flag:            "ipvs",
			mods:            []string{"foo", "bar", "baz"},
			ipsetVersion:    ipvs.MinIPSetCheckVersion,
			iptablesVersion: "0.0.0",
			kernelCompat:    true,
			expected:        proxyModeUserspace,
		},
		{ // flag says ipvs, ipset version too low, iptables version too old, kernel not compatible, fallback on userspace mode
			flag:            "ipvs",
			mods:            []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack_ipv4"},
			ipsetVersion:    "0.0",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    false,
			expected:        proxyModeUserspace,
		},
	}
	for i, c := range cases {
		versioner := &fakeIPTablesVersioner{c.iptablesVersion, c.iptablesError}
		kcompater := &fakeKernelCompatTester{c.kernelCompat}
		ipsetver := &fakeIPSetVersioner{c.ipsetVersion, c.ipsetError}
		khandle := &fakeKernelHandler{c.mods}
		r := getProxyMode(c.flag, versioner, khandle, ipsetver, kcompater)
		if r != c.expected {
			t.Errorf("Case[%d] Expected %q, got %q", i, c.expected, r)
		}
	}
}
