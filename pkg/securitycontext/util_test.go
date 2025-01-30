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

package securitycontext

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	utilptr "k8s.io/utils/pointer"
)

func TestAddNoNewPrivileges(t *testing.T) {
	pfalse := false
	ptrue := true

	tests := map[string]struct {
		sc     *v1.SecurityContext
		expect bool
	}{
		"allowPrivilegeEscalation nil security context nil": {
			sc:     nil,
			expect: false,
		},
		"allowPrivilegeEscalation nil": {
			sc: &v1.SecurityContext{
				AllowPrivilegeEscalation: nil,
			},
			expect: false,
		},
		"allowPrivilegeEscalation false": {
			sc: &v1.SecurityContext{
				AllowPrivilegeEscalation: &pfalse,
			},
			expect: true,
		},
		"allowPrivilegeEscalation true": {
			sc: &v1.SecurityContext{
				AllowPrivilegeEscalation: &ptrue,
			},
			expect: false,
		},
	}

	for k, v := range tests {
		actual := AddNoNewPrivileges(v.sc)
		if actual != v.expect {
			t.Errorf("%s failed, expected %t but received %t", k, v.expect, actual)
		}
	}
}

func TestConvertToRuntimeMaskedPaths(t *testing.T) {
	dPM := v1.DefaultProcMount
	uPM := v1.UnmaskedProcMount
	tests := map[string]struct {
		pm     *v1.ProcMountType
		expect []string
	}{
		"procMount nil": {
			pm:     nil,
			expect: defaultMaskedPaths,
		},
		"procMount default": {
			pm:     &dPM,
			expect: defaultMaskedPaths,
		},
		"procMount unmasked": {
			pm:     &uPM,
			expect: []string{},
		},
	}

	for k, v := range tests {
		actual := ConvertToRuntimeMaskedPaths(v.pm)
		if !reflect.DeepEqual(actual, v.expect) {
			t.Errorf("%s failed, expected %#v but received %#v", k, v.expect, actual)
		}
	}
}

func TestConvertToRuntimeReadonlyPaths(t *testing.T) {
	dPM := v1.DefaultProcMount
	uPM := v1.UnmaskedProcMount
	tests := map[string]struct {
		pm     *v1.ProcMountType
		expect []string
	}{
		"procMount nil": {
			pm:     nil,
			expect: defaultReadonlyPaths,
		},
		"procMount default": {
			pm:     &dPM,
			expect: defaultReadonlyPaths,
		},
		"procMount unmasked": {
			pm:     &uPM,
			expect: []string{},
		},
	}

	for k, v := range tests {
		actual := ConvertToRuntimeReadonlyPaths(v.pm)
		if !reflect.DeepEqual(actual, v.expect) {
			t.Errorf("%s failed, expected %#v but received %#v", k, v.expect, actual)
		}
	}
}

func TestDetermineEffectiveRunAsUser(t *testing.T) {
	tests := []struct {
		desc          string
		pod           *v1.Pod
		container     *v1.Container
		wantRunAsUser *int64
	}{
		{
			desc: "no securityContext in pod, no securityContext in container",
			pod: &v1.Pod{
				Spec: v1.PodSpec{},
			},
			container:     &v1.Container{},
			wantRunAsUser: nil,
		},
		{
			desc: "no runAsUser in pod, no runAsUser in container",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{},
				},
			},
			container: &v1.Container{
				SecurityContext: &v1.SecurityContext{},
			},
			wantRunAsUser: nil,
		},
		{
			desc: "runAsUser in pod, no runAsUser in container",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{
						RunAsUser: new(int64),
					},
				},
			},
			container: &v1.Container{
				SecurityContext: &v1.SecurityContext{},
			},
			wantRunAsUser: new(int64),
		},
		{
			desc: "no runAsUser in pod, runAsUser in container",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{},
				},
			},
			container: &v1.Container{
				SecurityContext: &v1.SecurityContext{
					RunAsUser: new(int64),
				},
			},
			wantRunAsUser: new(int64),
		},
		{
			desc: "no runAsUser in pod, runAsUser in container",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{
						RunAsUser: new(int64),
					},
				},
			},
			container: &v1.Container{
				SecurityContext: &v1.SecurityContext{
					RunAsUser: utilptr.Int64Ptr(1),
				},
			},
			wantRunAsUser: utilptr.Int64Ptr(1),
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			runAsUser, ok := DetermineEffectiveRunAsUser(test.pod, test.container)
			if !ok && test.wantRunAsUser != nil {
				t.Errorf("DetermineEffectiveRunAsUser(%v, %v) = %v, want %d", test.pod, test.container, runAsUser, *test.wantRunAsUser)
			}
			if ok && test.wantRunAsUser == nil {
				t.Errorf("DetermineEffectiveRunAsUser(%v, %v) = %d, want %v", test.pod, test.container, *runAsUser, test.wantRunAsUser)
			}
			if ok && test.wantRunAsUser != nil && *runAsUser != *test.wantRunAsUser {
				t.Errorf("DetermineEffectiveRunAsUser(%v, %v) = %d, want %d", test.pod, test.container, *runAsUser, *test.wantRunAsUser)
			}
		})
	}
}
