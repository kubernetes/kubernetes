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
	cases := []struct {
		chain                   string
		args                    []string
		expectedJumpIfOutput    []string
		expectedJumpIfNotOutput []string
	}{
		{
			chain:                   "TEST",
			args:                    []string{"arg1", "arg2"},
			expectedJumpIfOutput:    []string{"arg1", "arg2"},
			expectedJumpIfNotOutput: []string{"arg1", "arg2"},
		},
	}
	for _, c := range cases {
		localDetector := NewNoOpLocalDetector()
		if localDetector.IsImplemented() {
			t.Error("DetectLocalByCIDR returns true for IsImplemented")
		}

		jumpIf := localDetector.JumpIfLocal(c.args, c.chain)
		jumpIfNot := localDetector.JumpIfNotLocal(c.args, c.chain)

		if !reflect.DeepEqual(jumpIf, c.expectedJumpIfOutput) {
			t.Errorf("JumpIf, expected: '%v', but got: '%v'", c.expectedJumpIfOutput, jumpIf)
		}

		if !reflect.DeepEqual(jumpIfNot, c.expectedJumpIfNotOutput) {
			t.Errorf("JumpIfNot, expected: '%v', but got: '%v'", c.expectedJumpIfNotOutput, jumpIfNot)
		}
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
		cidr                    string
		ipt                     utiliptables.Interface
		chain                   string
		args                    []string
		expectedJumpIfOutput    []string
		expectedJumpIfNotOutput []string
	}{
		{
			cidr:                    "10.0.0.0/14",
			ipt:                     iptablestest.NewFake(),
			chain:                   "TEST",
			args:                    []string{"arg1", "arg2"},
			expectedJumpIfOutput:    []string{"arg1", "arg2", "-s", "10.0.0.0/14", "-j", "TEST"},
			expectedJumpIfNotOutput: []string{"arg1", "arg2", "!", "-s", "10.0.0.0/14", "-j", "TEST"},
		},
		{
			cidr:                    "2002::1234:abcd:ffff:c0a8:101/64",
			ipt:                     iptablestest.NewIPv6Fake(),
			chain:                   "TEST",
			args:                    []string{"arg1", "arg2"},
			expectedJumpIfOutput:    []string{"arg1", "arg2", "-s", "2002::1234:abcd:ffff:c0a8:101/64", "-j", "TEST"},
			expectedJumpIfNotOutput: []string{"arg1", "arg2", "!", "-s", "2002::1234:abcd:ffff:c0a8:101/64", "-j", "TEST"},
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

		jumpIf := localDetector.JumpIfLocal(c.args, c.chain)
		jumpIfNot := localDetector.JumpIfNotLocal(c.args, c.chain)

		if !reflect.DeepEqual(jumpIf, c.expectedJumpIfOutput) {
			t.Errorf("JumpIf, expected: '%v', but got: '%v'", c.expectedJumpIfOutput, jumpIf)
		}

		if !reflect.DeepEqual(jumpIfNot, c.expectedJumpIfNotOutput) {
			t.Errorf("JumpIfNot, expected: '%v', but got: '%v'", c.expectedJumpIfNotOutput, jumpIfNot)
		}
	}
}
