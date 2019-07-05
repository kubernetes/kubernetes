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

package securitycontext

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestPodSecurityContextAccessor(t *testing.T) {
	fsGroup := int64(2)
	runAsUser := int64(1)
	runAsGroup := int64(1)
	runAsNonRoot := true

	testcases := []*api.PodSecurityContext{
		nil,
		{},
		{FSGroup: &fsGroup},
		{HostIPC: true},
		{HostNetwork: true},
		{HostPID: true},
		{RunAsNonRoot: &runAsNonRoot},
		{RunAsUser: &runAsUser},
		{RunAsGroup: &runAsGroup},
		{SELinuxOptions: &api.SELinuxOptions{User: "bob"}},
		{SupplementalGroups: []int64{1, 2, 3}},
	}

	for i, tc := range testcases {
		expected := tc
		if expected == nil {
			expected = &api.PodSecurityContext{}
		}

		a := NewPodSecurityContextAccessor(tc)

		if v := a.FSGroup(); !reflect.DeepEqual(expected.FSGroup, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.FSGroup, v)
		}
		if v := a.HostIPC(); !reflect.DeepEqual(expected.HostIPC, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.HostIPC, v)
		}
		if v := a.HostNetwork(); !reflect.DeepEqual(expected.HostNetwork, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.HostNetwork, v)
		}
		if v := a.HostPID(); !reflect.DeepEqual(expected.HostPID, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.HostPID, v)
		}
		if v := a.RunAsNonRoot(); !reflect.DeepEqual(expected.RunAsNonRoot, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.RunAsNonRoot, v)
		}
		if v := a.RunAsUser(); !reflect.DeepEqual(expected.RunAsUser, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.RunAsUser, v)
		}
		if v := a.RunAsGroup(); !reflect.DeepEqual(expected.RunAsGroup, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.RunAsGroup, v)
		}
		if v := a.SELinuxOptions(); !reflect.DeepEqual(expected.SELinuxOptions, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.SELinuxOptions, v)
		}
		if v := a.SupplementalGroups(); !reflect.DeepEqual(expected.SupplementalGroups, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.SupplementalGroups, v)
		}
	}
}

func TestPodSecurityContextMutator(t *testing.T) {
	testcases := map[string]struct {
		newSC func() *api.PodSecurityContext
	}{
		"nil": {
			newSC: func() *api.PodSecurityContext { return nil },
		},
		"zero": {
			newSC: func() *api.PodSecurityContext { return &api.PodSecurityContext{} },
		},
		"populated": {
			newSC: func() *api.PodSecurityContext {
				return &api.PodSecurityContext{
					HostNetwork:        true,
					HostIPC:            true,
					HostPID:            true,
					SELinuxOptions:     &api.SELinuxOptions{},
					RunAsUser:          nil,
					RunAsGroup:         nil,
					RunAsNonRoot:       nil,
					SupplementalGroups: nil,
					FSGroup:            nil,
				}
			},
		},
	}

	nonNilSC := func(sc *api.PodSecurityContext) *api.PodSecurityContext {
		if sc == nil {
			return &api.PodSecurityContext{}
		}
		return sc
	}

	for k, tc := range testcases {
		{
			sc := tc.newSC()
			originalSC := tc.newSC()
			m := NewPodSecurityContextMutator(sc)

			// no-op sets should not modify the object
			m.SetFSGroup(m.FSGroup())
			m.SetHostNetwork(m.HostNetwork())
			m.SetHostIPC(m.HostIPC())
			m.SetHostPID(m.HostPID())
			m.SetRunAsNonRoot(m.RunAsNonRoot())
			m.SetRunAsUser(m.RunAsUser())
			m.SetRunAsGroup(m.RunAsGroup())
			m.SetSELinuxOptions(m.SELinuxOptions())
			m.SetSupplementalGroups(m.SupplementalGroups())
			if !reflect.DeepEqual(sc, originalSC) {
				t.Errorf("%s: unexpected mutation: %#v, %#v", k, sc, originalSC)
			}
			if !reflect.DeepEqual(m.PodSecurityContext(), originalSC) {
				t.Errorf("%s: unexpected mutation: %#v, %#v", k, m.PodSecurityContext(), originalSC)
			}
		}

		// FSGroup
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewPodSecurityContextMutator(tc.newSC())
			i := int64(1123)
			modifiedSC.FSGroup = &i
			m.SetFSGroup(&i)
			if !reflect.DeepEqual(m.PodSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.PodSecurityContext()))
				continue
			}
		}

		// HostNetwork
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewPodSecurityContextMutator(tc.newSC())
			modifiedSC.HostNetwork = !modifiedSC.HostNetwork
			m.SetHostNetwork(!m.HostNetwork())
			if !reflect.DeepEqual(m.PodSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.PodSecurityContext()))
				continue
			}
		}

		// HostIPC
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewPodSecurityContextMutator(tc.newSC())
			modifiedSC.HostIPC = !modifiedSC.HostIPC
			m.SetHostIPC(!m.HostIPC())
			if !reflect.DeepEqual(m.PodSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.PodSecurityContext()))
				continue
			}
		}

		// HostPID
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewPodSecurityContextMutator(tc.newSC())
			modifiedSC.HostPID = !modifiedSC.HostPID
			m.SetHostPID(!m.HostPID())
			if !reflect.DeepEqual(m.PodSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.PodSecurityContext()))
				continue
			}
		}

		// RunAsNonRoot
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewPodSecurityContextMutator(tc.newSC())
			b := true
			modifiedSC.RunAsNonRoot = &b
			m.SetRunAsNonRoot(&b)
			if !reflect.DeepEqual(m.PodSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.PodSecurityContext()))
				continue
			}
		}

		// RunAsUser
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewPodSecurityContextMutator(tc.newSC())
			i := int64(1123)
			modifiedSC.RunAsUser = &i
			m.SetRunAsUser(&i)
			if !reflect.DeepEqual(m.PodSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.PodSecurityContext()))
				continue
			}
		}

		// RunAsGroup
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewPodSecurityContextMutator(tc.newSC())
			i := int64(1123)
			modifiedSC.RunAsGroup = &i
			m.SetRunAsGroup(&i)
			if !reflect.DeepEqual(m.PodSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.PodSecurityContext()))
				continue
			}
		}

		// SELinuxOptions
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewPodSecurityContextMutator(tc.newSC())
			modifiedSC.SELinuxOptions = &api.SELinuxOptions{User: "bob"}
			m.SetSELinuxOptions(&api.SELinuxOptions{User: "bob"})
			if !reflect.DeepEqual(m.PodSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.PodSecurityContext()))
				continue
			}
		}

		// SupplementalGroups
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewPodSecurityContextMutator(tc.newSC())
			modifiedSC.SupplementalGroups = []int64{1, 1, 2, 3}
			m.SetSupplementalGroups([]int64{1, 1, 2, 3})
			if !reflect.DeepEqual(m.PodSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.PodSecurityContext()))
				continue
			}
		}
	}
}

func TestContainerSecurityContextAccessor(t *testing.T) {
	privileged := true
	runAsUser := int64(1)
	runAsNonRoot := true
	readOnlyRootFilesystem := true
	allowPrivilegeEscalation := true

	testcases := []*api.SecurityContext{
		nil,
		{},
		{Capabilities: &api.Capabilities{Drop: []api.Capability{"test"}}},
		{Privileged: &privileged},
		{SELinuxOptions: &api.SELinuxOptions{User: "bob"}},
		{RunAsUser: &runAsUser},
		{RunAsNonRoot: &runAsNonRoot},
		{ReadOnlyRootFilesystem: &readOnlyRootFilesystem},
		{AllowPrivilegeEscalation: &allowPrivilegeEscalation},
	}

	for i, tc := range testcases {
		expected := tc
		if expected == nil {
			expected = &api.SecurityContext{}
		}

		a := NewContainerSecurityContextAccessor(tc)

		if v := a.Capabilities(); !reflect.DeepEqual(expected.Capabilities, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.Capabilities, v)
		}
		if v := a.Privileged(); !reflect.DeepEqual(expected.Privileged, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.Privileged, v)
		}
		if v := a.RunAsNonRoot(); !reflect.DeepEqual(expected.RunAsNonRoot, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.RunAsNonRoot, v)
		}
		if v := a.RunAsUser(); !reflect.DeepEqual(expected.RunAsUser, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.RunAsUser, v)
		}
		if v := a.SELinuxOptions(); !reflect.DeepEqual(expected.SELinuxOptions, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.SELinuxOptions, v)
		}
		if v := a.ReadOnlyRootFilesystem(); !reflect.DeepEqual(expected.ReadOnlyRootFilesystem, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.ReadOnlyRootFilesystem, v)
		}
		if v := a.AllowPrivilegeEscalation(); !reflect.DeepEqual(expected.AllowPrivilegeEscalation, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.AllowPrivilegeEscalation, v)
		}
	}
}

func TestContainerSecurityContextMutator(t *testing.T) {
	testcases := map[string]struct {
		newSC func() *api.SecurityContext
	}{
		"nil": {
			newSC: func() *api.SecurityContext { return nil },
		},
		"zero": {
			newSC: func() *api.SecurityContext { return &api.SecurityContext{} },
		},
		"populated": {
			newSC: func() *api.SecurityContext {
				return &api.SecurityContext{
					Capabilities:   &api.Capabilities{Drop: []api.Capability{"test"}},
					SELinuxOptions: &api.SELinuxOptions{},
				}
			},
		},
	}

	nonNilSC := func(sc *api.SecurityContext) *api.SecurityContext {
		if sc == nil {
			return &api.SecurityContext{}
		}
		return sc
	}

	for k, tc := range testcases {
		{
			sc := tc.newSC()
			originalSC := tc.newSC()
			m := NewContainerSecurityContextMutator(sc)

			// no-op sets should not modify the object
			m.SetAllowPrivilegeEscalation(m.AllowPrivilegeEscalation())
			m.SetCapabilities(m.Capabilities())
			m.SetPrivileged(m.Privileged())
			m.SetReadOnlyRootFilesystem(m.ReadOnlyRootFilesystem())
			m.SetRunAsNonRoot(m.RunAsNonRoot())
			m.SetRunAsUser(m.RunAsUser())
			m.SetSELinuxOptions(m.SELinuxOptions())
			if !reflect.DeepEqual(sc, originalSC) {
				t.Errorf("%s: unexpected mutation: %#v, %#v", k, sc, originalSC)
			}
			if !reflect.DeepEqual(m.ContainerSecurityContext(), originalSC) {
				t.Errorf("%s: unexpected mutation: %#v, %#v", k, m.ContainerSecurityContext(), originalSC)
			}
		}

		// AllowPrivilegeEscalation
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewContainerSecurityContextMutator(tc.newSC())
			b := true
			modifiedSC.AllowPrivilegeEscalation = &b
			m.SetAllowPrivilegeEscalation(&b)
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}

		// Capabilities
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewContainerSecurityContextMutator(tc.newSC())
			modifiedSC.Capabilities = &api.Capabilities{Drop: []api.Capability{"test2"}}
			m.SetCapabilities(&api.Capabilities{Drop: []api.Capability{"test2"}})
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}

		// Privileged
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewContainerSecurityContextMutator(tc.newSC())
			b := true
			modifiedSC.Privileged = &b
			m.SetPrivileged(&b)
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}

		// ReadOnlyRootFilesystem
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewContainerSecurityContextMutator(tc.newSC())
			b := true
			modifiedSC.ReadOnlyRootFilesystem = &b
			m.SetReadOnlyRootFilesystem(&b)
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}

		// RunAsNonRoot
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewContainerSecurityContextMutator(tc.newSC())
			b := true
			modifiedSC.RunAsNonRoot = &b
			m.SetRunAsNonRoot(&b)
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}

		// RunAsUser
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewContainerSecurityContextMutator(tc.newSC())
			i := int64(1123)
			modifiedSC.RunAsUser = &i
			m.SetRunAsUser(&i)
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}

		// SELinuxOptions
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewContainerSecurityContextMutator(tc.newSC())
			modifiedSC.SELinuxOptions = &api.SELinuxOptions{User: "bob"}
			m.SetSELinuxOptions(&api.SELinuxOptions{User: "bob"})
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}
	}
}

func TestEffectiveContainerSecurityContextAccessor(t *testing.T) {
	privileged := true
	runAsUser := int64(1)
	runAsUserPod := int64(12)
	runAsGroup := int64(1)
	runAsGroupPod := int64(12)
	runAsNonRoot := true
	runAsNonRootPod := false
	readOnlyRootFilesystem := true
	allowPrivilegeEscalation := true

	testcases := []struct {
		PodSC     *api.PodSecurityContext
		SC        *api.SecurityContext
		Effective *api.SecurityContext
	}{
		{
			PodSC:     nil,
			SC:        nil,
			Effective: nil,
		},
		{
			PodSC:     &api.PodSecurityContext{},
			SC:        &api.SecurityContext{},
			Effective: &api.SecurityContext{},
		},
		{
			PodSC: &api.PodSecurityContext{
				SELinuxOptions: &api.SELinuxOptions{User: "bob"},
				RunAsUser:      &runAsUser,
				RunAsNonRoot:   &runAsNonRoot,
			},
			SC: nil,
			Effective: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{User: "bob"},
				RunAsUser:      &runAsUser,
				RunAsNonRoot:   &runAsNonRoot,
			},
		},
		{
			PodSC: &api.PodSecurityContext{
				SELinuxOptions: &api.SELinuxOptions{User: "bob"},
				RunAsUser:      &runAsUserPod,
				RunAsNonRoot:   &runAsNonRootPod,
			},
			SC: &api.SecurityContext{},
			Effective: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{User: "bob"},
				RunAsUser:      &runAsUserPod,
				RunAsNonRoot:   &runAsNonRootPod,
			},
		},
		{
			PodSC: &api.PodSecurityContext{
				SELinuxOptions: &api.SELinuxOptions{User: "bob"},
				RunAsUser:      &runAsUserPod,
				RunAsNonRoot:   &runAsNonRootPod,
			},
			SC: &api.SecurityContext{
				AllowPrivilegeEscalation: &allowPrivilegeEscalation,
				Capabilities:             &api.Capabilities{Drop: []api.Capability{"test"}},
				Privileged:               &privileged,
				ReadOnlyRootFilesystem:   &readOnlyRootFilesystem,
				RunAsUser:                &runAsUser,
				RunAsNonRoot:             &runAsNonRoot,
				SELinuxOptions:           &api.SELinuxOptions{User: "bob"},
			},
			Effective: &api.SecurityContext{
				AllowPrivilegeEscalation: &allowPrivilegeEscalation,
				Capabilities:             &api.Capabilities{Drop: []api.Capability{"test"}},
				Privileged:               &privileged,
				ReadOnlyRootFilesystem:   &readOnlyRootFilesystem,
				RunAsUser:                &runAsUser,
				RunAsNonRoot:             &runAsNonRoot,
				SELinuxOptions:           &api.SELinuxOptions{User: "bob"},
			},
		},
		{
			PodSC: &api.PodSecurityContext{
				RunAsGroup: &runAsGroup,
			},
			SC: nil,
			Effective: &api.SecurityContext{
				RunAsGroup: &runAsGroup,
			},
		},
		{
			PodSC: &api.PodSecurityContext{
				RunAsGroup: &runAsGroupPod,
			},
			SC: &api.SecurityContext{
				RunAsGroup: &runAsGroup,
			},
			Effective: &api.SecurityContext{
				RunAsGroup: &runAsGroup,
			},
		},
	}

	for i, tc := range testcases {
		expected := tc.Effective
		if expected == nil {
			expected = &api.SecurityContext{}
		}

		a := NewEffectiveContainerSecurityContextAccessor(
			NewPodSecurityContextAccessor(tc.PodSC),
			NewContainerSecurityContextMutator(tc.SC),
		)

		if v := a.Capabilities(); !reflect.DeepEqual(expected.Capabilities, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.Capabilities, v)
		}
		if v := a.Privileged(); !reflect.DeepEqual(expected.Privileged, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.Privileged, v)
		}
		if v := a.RunAsNonRoot(); !reflect.DeepEqual(expected.RunAsNonRoot, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.RunAsNonRoot, v)
		}
		if v := a.RunAsUser(); !reflect.DeepEqual(expected.RunAsUser, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.RunAsUser, v)
		}
		if v := a.SELinuxOptions(); !reflect.DeepEqual(expected.SELinuxOptions, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.SELinuxOptions, v)
		}
		if v := a.ReadOnlyRootFilesystem(); !reflect.DeepEqual(expected.ReadOnlyRootFilesystem, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.ReadOnlyRootFilesystem, v)
		}
		if v := a.AllowPrivilegeEscalation(); !reflect.DeepEqual(expected.AllowPrivilegeEscalation, v) {
			t.Errorf("%d: expected %#v, got %#v", i, expected.AllowPrivilegeEscalation, v)
		}
	}
}

func TestEffectiveContainerSecurityContextMutator(t *testing.T) {
	runAsNonRootPod := false
	runAsUserPod := int64(12)

	testcases := map[string]struct {
		newPodSC func() *api.PodSecurityContext
		newSC    func() *api.SecurityContext
	}{
		"nil": {
			newPodSC: func() *api.PodSecurityContext { return nil },
			newSC:    func() *api.SecurityContext { return nil },
		},
		"zero": {
			newPodSC: func() *api.PodSecurityContext { return &api.PodSecurityContext{} },
			newSC:    func() *api.SecurityContext { return &api.SecurityContext{} },
		},
		"populated pod sc": {
			newPodSC: func() *api.PodSecurityContext {
				return &api.PodSecurityContext{
					SELinuxOptions: &api.SELinuxOptions{User: "poduser"},
					RunAsNonRoot:   &runAsNonRootPod,
					RunAsUser:      &runAsUserPod,
				}
			},
			newSC: func() *api.SecurityContext {
				return &api.SecurityContext{}
			},
		},
		"populated sc": {
			newPodSC: func() *api.PodSecurityContext { return nil },
			newSC: func() *api.SecurityContext {
				return &api.SecurityContext{
					Capabilities:   &api.Capabilities{Drop: []api.Capability{"test"}},
					SELinuxOptions: &api.SELinuxOptions{},
				}
			},
		},
	}

	nonNilSC := func(sc *api.SecurityContext) *api.SecurityContext {
		if sc == nil {
			return &api.SecurityContext{}
		}
		return sc
	}

	for k, tc := range testcases {
		{
			podSC := tc.newPodSC()
			sc := tc.newSC()
			originalPodSC := tc.newPodSC()
			originalSC := tc.newSC()
			m := NewEffectiveContainerSecurityContextMutator(
				NewPodSecurityContextAccessor(podSC),
				NewContainerSecurityContextMutator(sc),
			)

			// no-op sets should not modify the object
			m.SetAllowPrivilegeEscalation(m.AllowPrivilegeEscalation())
			m.SetCapabilities(m.Capabilities())
			m.SetPrivileged(m.Privileged())
			m.SetReadOnlyRootFilesystem(m.ReadOnlyRootFilesystem())
			m.SetRunAsNonRoot(m.RunAsNonRoot())
			m.SetRunAsUser(m.RunAsUser())
			m.SetSELinuxOptions(m.SELinuxOptions())
			if !reflect.DeepEqual(podSC, originalPodSC) {
				t.Errorf("%s: unexpected mutation: %#v, %#v", k, podSC, originalPodSC)
			}
			if !reflect.DeepEqual(sc, originalSC) {
				t.Errorf("%s: unexpected mutation: %#v, %#v", k, sc, originalSC)
			}
			if !reflect.DeepEqual(m.ContainerSecurityContext(), originalSC) {
				t.Errorf("%s: unexpected mutation: %#v, %#v", k, m.ContainerSecurityContext(), originalSC)
			}
		}

		// AllowPrivilegeEscalation
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewEffectiveContainerSecurityContextMutator(
				NewPodSecurityContextAccessor(tc.newPodSC()),
				NewContainerSecurityContextMutator(tc.newSC()),
			)
			b := true
			modifiedSC.AllowPrivilegeEscalation = &b
			m.SetAllowPrivilegeEscalation(&b)
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}

		// Capabilities
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewEffectiveContainerSecurityContextMutator(
				NewPodSecurityContextAccessor(tc.newPodSC()),
				NewContainerSecurityContextMutator(tc.newSC()),
			)
			modifiedSC.Capabilities = &api.Capabilities{Drop: []api.Capability{"test2"}}
			m.SetCapabilities(&api.Capabilities{Drop: []api.Capability{"test2"}})
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}

		// Privileged
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewEffectiveContainerSecurityContextMutator(
				NewPodSecurityContextAccessor(tc.newPodSC()),
				NewContainerSecurityContextMutator(tc.newSC()),
			)
			b := true
			modifiedSC.Privileged = &b
			m.SetPrivileged(&b)
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}

		// ReadOnlyRootFilesystem
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewEffectiveContainerSecurityContextMutator(
				NewPodSecurityContextAccessor(tc.newPodSC()),
				NewContainerSecurityContextMutator(tc.newSC()),
			)
			b := true
			modifiedSC.ReadOnlyRootFilesystem = &b
			m.SetReadOnlyRootFilesystem(&b)
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}

		// RunAsNonRoot
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewEffectiveContainerSecurityContextMutator(
				NewPodSecurityContextAccessor(tc.newPodSC()),
				NewContainerSecurityContextMutator(tc.newSC()),
			)
			b := true
			modifiedSC.RunAsNonRoot = &b
			m.SetRunAsNonRoot(&b)
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}

		// RunAsUser
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewEffectiveContainerSecurityContextMutator(
				NewPodSecurityContextAccessor(tc.newPodSC()),
				NewContainerSecurityContextMutator(tc.newSC()),
			)
			i := int64(1123)
			modifiedSC.RunAsUser = &i
			m.SetRunAsUser(&i)
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}

		// SELinuxOptions
		{
			modifiedSC := nonNilSC(tc.newSC())
			m := NewEffectiveContainerSecurityContextMutator(
				NewPodSecurityContextAccessor(tc.newPodSC()),
				NewContainerSecurityContextMutator(tc.newSC()),
			)
			modifiedSC.SELinuxOptions = &api.SELinuxOptions{User: "bob"}
			m.SetSELinuxOptions(&api.SELinuxOptions{User: "bob"})
			if !reflect.DeepEqual(m.ContainerSecurityContext(), modifiedSC) {
				t.Errorf("%s: unexpected object:\n%s", k, diff.ObjectGoPrintSideBySide(modifiedSC, m.ContainerSecurityContext()))
				continue
			}
		}
	}
}
