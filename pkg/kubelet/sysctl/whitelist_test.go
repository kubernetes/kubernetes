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

package sysctl

import (
	"testing"

	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/sysctl"
)

func TestNewWhitelist(t *testing.T) {
	type Test struct {
		sysctls []string
		err     bool
	}
	for _, test := range []Test{
		{sysctls: []string{"kernel.msg*", "kernel.sem"}},
		{sysctls: []string{" kernel.msg*"}, err: true},
		{sysctls: []string{"kernel.msg* "}, err: true},
		{sysctls: []string{"net.-"}, err: true},
		{sysctls: []string{"net.*.foo"}, err: true},
		{sysctls: []string{"foo"}, err: true},
	} {
		_, err := NewWhitelist(append(sysctl.SafeSysctlWhitelist(), test.sysctls...))
		if test.err && err == nil {
			t.Errorf("expected an error creating a whitelist for %v", test.sysctls)
		} else if !test.err && err != nil {
			t.Errorf("got unexpected error creating a whitelist for %v: %v", test.sysctls, err)
		}
	}
}

func TestWhitelist(t *testing.T) {
	type Test struct {
		sysctl           string
		hostNet, hostIPC bool
	}
	valid := []Test{
		{sysctl: "kernel.shm_rmid_forced"},
		{sysctl: "net.ipv4.ip_local_port_range"},
		{sysctl: "kernel.msgmax"},
		{sysctl: "kernel.sem"},
	}
	invalid := []Test{
		{sysctl: "kernel.shm_rmid_forced", hostIPC: true},
		{sysctl: "net.ipv4.ip_local_port_range", hostNet: true},
		{sysctl: "foo"},
		{sysctl: "net.a.b.c", hostNet: false},
		{sysctl: "net.ipv4.ip_local_port_range.a.b.c", hostNet: false},
		{sysctl: "kernel.msgmax", hostIPC: true},
		{sysctl: "kernel.sem", hostIPC: true},
	}

	w, err := NewWhitelist(append(sysctl.SafeSysctlWhitelist(), "kernel.msg*", "kernel.sem"))
	if err != nil {
		t.Fatalf("failed to create whitelist: %v", err)
	}

	for _, test := range valid {
		if err := w.validateSysctl(test.sysctl, test.hostNet, test.hostIPC); err != nil {
			t.Errorf("expected to be whitelisted: %+v, got: %v", test, err)
		}
	}

	for _, test := range invalid {
		if err := w.validateSysctl(test.sysctl, test.hostNet, test.hostIPC); err == nil {
			t.Errorf("expected to be rejected: %+v", test)
		}
	}
}
