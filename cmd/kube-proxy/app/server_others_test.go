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

	"k8s.io/kubernetes/pkg/proxy/ipvs"
)

type fakeIPSetVersioner struct {
	version string // what to return
	err     error  // what to return
}

func (fake *fakeIPSetVersioner) GetVersion() (string, error) {
	return fake.version, fake.err
}

type fakeKernelCompatTester struct {
	ok bool
}

func (fake *fakeKernelCompatTester) IsCompatible() error {
	if !fake.ok {
		return fmt.Errorf("error")
	}
	return nil
}

// fakeKernelHandler implements KernelHandler.
type fakeKernelHandler struct {
	modules       []string
	kernelVersion string
}

func (fake *fakeKernelHandler) GetModules() ([]string, error) {
	return fake.modules, nil
}

func (fake *fakeKernelHandler) GetKernelVersion() (string, error) {
	return fake.kernelVersion, nil
}

func Test_getProxyMode(t *testing.T) {
	var cases = []struct {
		flag          string
		ipsetVersion  string
		kmods         []string
		kernelVersion string
		kernelCompat  bool
		ipsetError    error
		expected      string
	}{
		{ // flag says userspace
			flag:     "userspace",
			expected: proxyModeUserspace,
		},
		{ // flag says iptables, kernel not compatible
			flag:         "iptables",
			kernelCompat: false,
			expected:     proxyModeUserspace,
		},
		{ // flag says iptables, kernel is compatible
			flag:         "iptables",
			kernelCompat: true,
			expected:     proxyModeIPTables,
		},
		{ // detect, kernel not compatible
			flag:         "",
			kernelCompat: false,
			expected:     proxyModeUserspace,
		},
		{ // detect, kernel is compatible
			flag:         "",
			kernelCompat: true,
			expected:     proxyModeIPTables,
		},
		{ // flag says ipvs, ipset version ok, kernel modules installed for linux kernel before 4.19
			flag:          "ipvs",
			kmods:         []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack_ipv4"},
			kernelVersion: "4.18",
			ipsetVersion:  ipvs.MinIPSetCheckVersion,
			expected:      proxyModeIPVS,
		},
		{ // flag says ipvs, ipset version ok, kernel modules installed for linux kernel 4.19
			flag:          "ipvs",
			kmods:         []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack"},
			kernelVersion: "4.19",
			ipsetVersion:  ipvs.MinIPSetCheckVersion,
			expected:      proxyModeIPVS,
		},
		{ // flag says ipvs, ipset version too low, fallback on iptables mode
			flag:          "ipvs",
			kmods:         []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack"},
			kernelVersion: "4.19",
			ipsetVersion:  "0.0",
			kernelCompat:  true,
			expected:      proxyModeIPTables,
		},
		{ // flag says ipvs, bad ipset version, fallback on iptables mode
			flag:          "ipvs",
			kmods:         []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack"},
			kernelVersion: "4.19",
			ipsetVersion:  "a.b.c",
			kernelCompat:  true,
			expected:      proxyModeIPTables,
		},
		{ // flag says ipvs, required kernel modules are not installed, fallback on iptables mode
			flag:          "ipvs",
			kmods:         []string{"foo", "bar", "baz"},
			kernelVersion: "4.19",
			ipsetVersion:  ipvs.MinIPSetCheckVersion,
			kernelCompat:  true,
			expected:      proxyModeIPTables,
		},
	}
	for i, c := range cases {
		kcompater := &fakeKernelCompatTester{c.kernelCompat}
		ipsetver := &fakeIPSetVersioner{c.ipsetVersion, c.ipsetError}
		khandler := &fakeKernelHandler{
			modules:       c.kmods,
			kernelVersion: c.kernelVersion,
		}
		r := getProxyMode(c.flag, khandler, ipsetver, kcompater)
		if r != c.expected {
			t.Errorf("Case[%d] Expected %q, got %q", i, c.expected, r)
		}
	}
}
