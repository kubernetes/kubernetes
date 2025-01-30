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

package util

import (
	"reflect"
	"testing"
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

func TestDetectLocalByCIDR(t *testing.T) {
	cases := []struct {
		cidr                     string
		expectedIfLocalOutput    []string
		expectedIfNotLocalOutput []string
	}{
		{
			cidr:                     "10.0.0.0/14",
			expectedIfLocalOutput:    []string{"-s", "10.0.0.0/14"},
			expectedIfNotLocalOutput: []string{"!", "-s", "10.0.0.0/14"},
		},
		{
			cidr:                     "2002:0:0:1234::/64",
			expectedIfLocalOutput:    []string{"-s", "2002:0:0:1234::/64"},
			expectedIfNotLocalOutput: []string{"!", "-s", "2002:0:0:1234::/64"},
		},
	}
	for _, c := range cases {
		localDetector := NewDetectLocalByCIDR(c.cidr)
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

func TestDetectLocalByBridgeInterface(t *testing.T) {
	cases := []struct {
		ifaceName               string
		expectedJumpIfOutput    []string
		expectedJumpIfNotOutput []string
	}{
		{
			ifaceName:               "eth0",
			expectedJumpIfOutput:    []string{"-i", "eth0"},
			expectedJumpIfNotOutput: []string{"!", "-i", "eth0"},
		},
	}
	for _, c := range cases {
		localDetector := NewDetectLocalByBridgeInterface(c.ifaceName)
		if !localDetector.IsImplemented() {
			t.Error("DetectLocalByBridgeInterface returns false for IsImplemented")
		}

		ifLocal := localDetector.IfLocal()
		ifNotLocal := localDetector.IfNotLocal()

		if !reflect.DeepEqual(ifLocal, c.expectedJumpIfOutput) {
			t.Errorf("IfLocal, expected: '%v', but got: '%v'", c.expectedJumpIfOutput, ifLocal)
		}

		if !reflect.DeepEqual(ifNotLocal, c.expectedJumpIfNotOutput) {
			t.Errorf("IfNotLocal, expected: '%v', but got: '%v'", c.expectedJumpIfNotOutput, ifNotLocal)
		}
	}
}

func TestDetectLocalByInterfaceNamePrefix(t *testing.T) {
	cases := []struct {
		ifacePrefix             string
		chain                   string
		args                    []string
		expectedJumpIfOutput    []string
		expectedJumpIfNotOutput []string
	}{
		{
			ifacePrefix:             "eth0",
			expectedJumpIfOutput:    []string{"-i", "eth0+"},
			expectedJumpIfNotOutput: []string{"!", "-i", "eth0+"},
		},
	}
	for _, c := range cases {
		localDetector := NewDetectLocalByInterfaceNamePrefix(c.ifacePrefix)
		if !localDetector.IsImplemented() {
			t.Error("DetectLocalByInterfaceNamePrefix returns false for IsImplemented")
		}

		ifLocal := localDetector.IfLocal()
		ifNotLocal := localDetector.IfNotLocal()

		if !reflect.DeepEqual(ifLocal, c.expectedJumpIfOutput) {
			t.Errorf("IfLocal, expected: '%v', but got: '%v'", c.expectedJumpIfOutput, ifLocal)
		}

		if !reflect.DeepEqual(ifNotLocal, c.expectedJumpIfNotOutput) {
			t.Errorf("IfNotLocal, expected: '%v', but got: '%v'", c.expectedJumpIfNotOutput, ifNotLocal)
		}
	}
}
