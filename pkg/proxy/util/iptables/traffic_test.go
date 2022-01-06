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

package iptables

import (
	"reflect"
	"testing"

	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	iptablestest "k8s.io/kubernetes/pkg/util/iptables/testing"
)

func TestNoOpLocalDetector(t *testing.T) {
	localDetector := NewNoOpLocalDetector()
	if localDetector.IsImplemented() {
		t.Error("NoOpLocalDetector returns true for IsImplemented")
	}

	ifLocal := localDetector.IfLocal()
	if len(ifLocal) != 0 {
		t.Errorf("NoOpLocalDetector returns %v for IsLocal (expected nil)", ifLocal)
	}

	ifNotLocal := localDetector.IfNotLocal()
	if len(ifNotLocal) != 0 {
		t.Errorf("NoOpLocalDetector returns %v for IsNotLocal (expected nil)", ifNotLocal)
	}
}

func TestNewDetectLocalByCIDR(t *testing.T) {
	cases := []struct {
		cidr        string
		ipt         utiliptables.Interface
		errExpected bool
	}{
		{
			cidr:        "10.0.0.0/14",
			ipt:         iptablestest.NewFake(),
			errExpected: false,
		},
		{
			cidr:        "2002::1234:abcd:ffff:c0a8:101/64",
			ipt:         iptablestest.NewIPv6Fake(),
			errExpected: false,
		},
		{
			cidr:        "10.0.0.0/14",
			ipt:         iptablestest.NewIPv6Fake(),
			errExpected: true,
		},
		{
			cidr:        "2002::1234:abcd:ffff:c0a8:101/64",
			ipt:         iptablestest.NewFake(),
			errExpected: true,
		},
		{
			cidr:        "10.0.0.0",
			ipt:         iptablestest.NewFake(),
			errExpected: true,
		},
		{
			cidr:        "2002::1234:abcd:ffff:c0a8:101",
			ipt:         iptablestest.NewIPv6Fake(),
			errExpected: true,
		},
		{
			cidr:        "",
			ipt:         iptablestest.NewFake(),
			errExpected: true,
		},
		{
			cidr:        "",
			ipt:         iptablestest.NewIPv6Fake(),
			errExpected: true,
		},
	}
	for i, c := range cases {
		r, err := NewDetectLocalByCIDR(c.cidr, c.ipt)
		if c.errExpected {
			if err == nil {
				t.Errorf("Case[%d] expected error, but succeeded with: %q", i, r)
			}
			continue
		}
		if err != nil {
			t.Errorf("Case[%d] failed with error: %v", i, err)
		}
	}
}

func TestDetectLocalByCIDR(t *testing.T) {
	cases := []struct {
		cidr                     string
		ipt                      utiliptables.Interface
		expectedIfLocalOutput    []string
		expectedIfNotLocalOutput []string
	}{
		{
			cidr:                     "10.0.0.0/14",
			ipt:                      iptablestest.NewFake(),
			expectedIfLocalOutput:    []string{"-s", "10.0.0.0/14"},
			expectedIfNotLocalOutput: []string{"!", "-s", "10.0.0.0/14"},
		},
		{
			cidr:                     "2002::1234:abcd:ffff:c0a8:101/64",
			ipt:                      iptablestest.NewIPv6Fake(),
			expectedIfLocalOutput:    []string{"-s", "2002::1234:abcd:ffff:c0a8:101/64"},
			expectedIfNotLocalOutput: []string{"!", "-s", "2002::1234:abcd:ffff:c0a8:101/64"},
		},
	}
	for _, c := range cases {
		localDetector, err := NewDetectLocalByCIDR(c.cidr, c.ipt)
		if err != nil {
			t.Errorf("Error initializing localDetector: %v", err)
			continue
		}
		if !localDetector.IsImplemented() {
			t.Error("DetectLocalByCIDR returns false for IsImplemented")
		}

		ifLocal := localDetector.IfLocal()
		ifNotLocal := localDetector.IfNotLocal()

		if !reflect.DeepEqual(ifLocal, c.expectedIfLocalOutput) {
			t.Errorf("IfLocal, expected: '%v', but got: '%v'", c.expectedIfLocalOutput, ifLocal)
		}

		if !reflect.DeepEqual(ifNotLocal, c.expectedIfNotLocalOutput) {
			t.Errorf("IfNotLocal, expected: '%v', but got: '%v'", c.expectedIfNotLocalOutput, ifNotLocal)
		}
	}
}
