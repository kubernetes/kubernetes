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

package securitycontextconstraints

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	sccutil "k8s.io/kubernetes/pkg/securitycontextconstraints/util"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

func TestCreatePodSecurityContextNonmutating(t *testing.T) {
	// Create a pod with a security context that needs filling in
	createPod := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{},
			},
		}
	}

	// Create an SCC with strategies that will populate a blank psc
	createSCC := func() *api.SecurityContextConstraints {
		return &api.SecurityContextConstraints{
			ObjectMeta: api.ObjectMeta{
				Name: "scc-sa",
			},
			SeccompProfiles:          []string{"foo"},
			DefaultAddCapabilities:   []api.Capability{"foo"},
			RequiredDropCapabilities: []api.Capability{"bar"},
			RunAsUser: api.RunAsUserStrategyOptions{
				Type: api.RunAsUserStrategyRunAsAny,
			},
			SELinuxContext: api.SELinuxContextStrategyOptions{
				Type: api.SELinuxStrategyRunAsAny,
			},
			// these are pod mutating strategies that are tested above
			FSGroup: api.FSGroupStrategyOptions{
				Type: api.FSGroupStrategyMustRunAs,
				Ranges: []api.IDRange{
					{Min: 1, Max: 1},
				},
			},
			SupplementalGroups: api.SupplementalGroupsStrategyOptions{
				Type: api.SupplementalGroupsStrategyMustRunAs,
				Ranges: []api.IDRange{
					{Min: 1, Max: 1},
				},
			},
		}
	}

	pod := createPod()
	scc := createSCC()

	provider, err := NewSimpleProvider(scc)
	if err != nil {
		t.Fatalf("unable to create provider %v", err)
	}
	sc, annotations, err := provider.CreatePodSecurityContext(pod)
	if err != nil {
		t.Fatalf("unable to create psc %v", err)
	}

	// The generated security context should have filled in missing options, so they should differ
	if reflect.DeepEqual(sc, &pod.Spec.SecurityContext) {
		t.Error("expected created security context to be different than container's, but they were identical")
	}

	if reflect.DeepEqual(annotations, pod.Annotations) {
		t.Error("expected created annotations to be different than container's, but they were identical")
	}

	// Creating the provider or the security context should not have mutated the scc or pod
	if !reflect.DeepEqual(createPod(), pod) {
		diff := diff.ObjectDiff(createPod(), pod)
		t.Errorf("pod was mutated by CreatePodSecurityContext. diff:\n%s", diff)
	}
	if !reflect.DeepEqual(createSCC(), scc) {
		t.Error("scc was mutated by CreatePodSecurityContext")
	}
}

func TestCreateContainerSecurityContextNonmutating(t *testing.T) {
	// Create a pod with a security context that needs filling in
	createPod := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{{
					SecurityContext: &api.SecurityContext{},
				}},
			},
		}
	}

	// Create an SCC with strategies that will populate a blank security context
	createSCC := func() *api.SecurityContextConstraints {
		var uid int64 = 1
		return &api.SecurityContextConstraints{
			ObjectMeta: api.ObjectMeta{
				Name: "scc-sa",
			},
			DefaultAddCapabilities:   []api.Capability{"foo"},
			RequiredDropCapabilities: []api.Capability{"bar"},
			RunAsUser: api.RunAsUserStrategyOptions{
				Type: api.RunAsUserStrategyMustRunAs,
				UID:  &uid,
			},
			SELinuxContext: api.SELinuxContextStrategyOptions{
				Type:           api.SELinuxStrategyMustRunAs,
				SELinuxOptions: &api.SELinuxOptions{User: "you"},
			},
			// these are pod mutating strategies that are tested above
			FSGroup: api.FSGroupStrategyOptions{
				Type: api.FSGroupStrategyRunAsAny,
			},
			SupplementalGroups: api.SupplementalGroupsStrategyOptions{
				Type: api.SupplementalGroupsStrategyRunAsAny,
			},
			// mutates the container SC by defaulting to true if container sets nil
			ReadOnlyRootFilesystem: true,
		}
	}

	pod := createPod()
	scc := createSCC()

	provider, err := NewSimpleProvider(scc)
	if err != nil {
		t.Fatalf("unable to create provider %v", err)
	}
	sc, err := provider.CreateContainerSecurityContext(pod, &pod.Spec.Containers[0])
	if err != nil {
		t.Fatalf("unable to create container security context %v", err)
	}

	// The generated security context should have filled in missing options, so they should differ
	if reflect.DeepEqual(sc, &pod.Spec.Containers[0].SecurityContext) {
		t.Error("expected created security context to be different than container's, but they were identical")
	}

	// Creating the provider or the security context should not have mutated the scc or pod
	if !reflect.DeepEqual(createPod(), pod) {
		diff := diff.ObjectDiff(createPod(), pod)
		t.Errorf("pod was mutated by CreateContainerSecurityContext. diff:\n%s", diff)
	}
	if !reflect.DeepEqual(createSCC(), scc) {
		t.Error("scc was mutated by CreateContainerSecurityContext")
	}
}

func TestValidatePodSecurityContextFailures(t *testing.T) {
	failHostNetworkPod := defaultPod()
	failHostNetworkPod.Spec.SecurityContext.HostNetwork = true

	failHostPIDPod := defaultPod()
	failHostPIDPod.Spec.SecurityContext.HostPID = true

	failHostIPCPod := defaultPod()
	failHostIPCPod.Spec.SecurityContext.HostIPC = true

	failSupplementalGroupPod := defaultPod()
	failSupplementalGroupPod.Spec.SecurityContext.SupplementalGroups = []int64{999}
	failSupplementalGroupSCC := defaultSCC()
	failSupplementalGroupSCC.SupplementalGroups = api.SupplementalGroupsStrategyOptions{
		Type: api.SupplementalGroupsStrategyMustRunAs,
		Ranges: []api.IDRange{
			{Min: 1, Max: 1},
		},
	}

	failFSGroupPod := defaultPod()
	fsGroup := int64(999)
	failFSGroupPod.Spec.SecurityContext.FSGroup = &fsGroup
	failFSGroupSCC := defaultSCC()
	failFSGroupSCC.FSGroup = api.FSGroupStrategyOptions{
		Type: api.FSGroupStrategyMustRunAs,
		Ranges: []api.IDRange{
			{Min: 1, Max: 1},
		},
	}

	failNilSELinuxPod := defaultPod()
	failSELinuxSCC := defaultSCC()
	failSELinuxSCC.SELinuxContext.Type = api.SELinuxStrategyMustRunAs
	failSELinuxSCC.SELinuxContext.SELinuxOptions = &api.SELinuxOptions{
		Level: "foo",
	}

	failInvalidSELinuxPod := defaultPod()
	failInvalidSELinuxPod.Spec.SecurityContext.SELinuxOptions = &api.SELinuxOptions{
		Level: "bar",
	}

	failNoSeccompAllowed := defaultPod()
	failNoSeccompAllowed.Annotations[api.SeccompPodAnnotationKey] = "bar"

	failInvalidSeccompProfile := defaultPod()
	failInvalidSeccompProfile.Annotations[api.SeccompPodAnnotationKey] = "bar"

	failInvalidSeccompProfileSCC := defaultSCC()
	failInvalidSeccompProfileSCC.SeccompProfiles = []string{"foo"}

	failOtherSysctlsAllowedSCC := defaultSCC()
	failOtherSysctlsAllowedSCC.Annotations[extensions.SysctlsPodSecurityPolicyAnnotationKey] = "bar,abc"

	failNoSysctlAllowedSCC := defaultSCC()
	failNoSysctlAllowedSCC.Annotations[extensions.SysctlsPodSecurityPolicyAnnotationKey] = ""

	failSafeSysctlFooPod := defaultPod()
	failSafeSysctlFooPod.Annotations[api.SysctlsPodAnnotationKey] = "foo=1"

	failUnsafeSysctlFooPod := defaultPod()
	failUnsafeSysctlFooPod.Annotations[api.UnsafeSysctlsPodAnnotationKey] = "foo=1"

	errorCases := map[string]struct {
		pod           *api.Pod
		scc           *api.SecurityContextConstraints
		expectedError string
	}{
		"failHostNetworkSCC": {
			pod:           failHostNetworkPod,
			scc:           defaultSCC(),
			expectedError: "Host network is not allowed to be used",
		},
		"failHostPIDSCC": {
			pod:           failHostPIDPod,
			scc:           defaultSCC(),
			expectedError: "Host PID is not allowed to be used",
		},
		"failHostIPCSCC": {
			pod:           failHostIPCPod,
			scc:           defaultSCC(),
			expectedError: "Host IPC is not allowed to be used",
		},
		"failSupplementalGroupOutOfRange": {
			pod:           failSupplementalGroupPod,
			scc:           failSupplementalGroupSCC,
			expectedError: "999 is not an allowed group",
		},
		"failSupplementalGroupEmpty": {
			pod:           defaultPod(),
			scc:           failSupplementalGroupSCC,
			expectedError: "unable to validate empty groups against required ranges",
		},
		"failFSGroupOutOfRange": {
			pod:           failFSGroupPod,
			scc:           failFSGroupSCC,
			expectedError: "999 is not an allowed group",
		},
		"failFSGroupEmpty": {
			pod:           defaultPod(),
			scc:           failFSGroupSCC,
			expectedError: "unable to validate empty groups against required ranges",
		},
		"failNilSELinux": {
			pod:           failNilSELinuxPod,
			scc:           failSELinuxSCC,
			expectedError: "unable to validate nil seLinuxOptions",
		},
		"failInvalidSELinux": {
			pod:           failInvalidSELinuxPod,
			scc:           failSELinuxSCC,
			expectedError: "does not match required level.  Found bar, wanted foo",
		},
		"failNoSeccomp": {
			pod:           failNoSeccompAllowed,
			scc:           defaultSCC(),
			expectedError: "seccomp may not be set",
		},
		"failInvalidSeccompPod": {
			pod:           failInvalidSeccompProfile,
			scc:           failInvalidSeccompProfileSCC,
			expectedError: "bar is not a valid seccomp profile",
		},
		"failSafeSysctlFooPod with failNoSysctlAllowedSCC": {
			pod:           failSafeSysctlFooPod,
			scc:           failNoSysctlAllowedSCC,
			expectedError: "sysctls are not allowed",
		},
		"failUnsafeSysctlFooPod with failNoSysctlAllowedSCC": {
			pod:           failUnsafeSysctlFooPod,
			scc:           failNoSysctlAllowedSCC,
			expectedError: "sysctls are not allowed",
		},
		"failSafeSysctlFooPod with failOtherSysctlsAllowedSCC": {
			pod:           failSafeSysctlFooPod,
			scc:           failOtherSysctlsAllowedSCC,
			expectedError: "sysctl \"foo\" is not allowed",
		},
		"failUnsafeSysctlFooPod with failOtherSysctlsAllowedSCC": {
			pod:           failUnsafeSysctlFooPod,
			scc:           failOtherSysctlsAllowedSCC,
			expectedError: "sysctl \"foo\" is not allowed",
		},
	}
	for k, v := range errorCases {
		provider, err := NewSimpleProvider(v.scc)
		if err != nil {
			t.Fatalf("unable to create provider %v", err)
		}
		errs := provider.ValidatePodSecurityContext(v.pod, field.NewPath(""))
		if len(errs) == 0 {
			t.Errorf("%s expected validation failure but did not receive errors", k)
			continue
		}
		if !strings.Contains(errs[0].Error(), v.expectedError) {
			t.Errorf("%s received unexpected error %v", k, errs)
		}
	}
}

func TestValidateContainerSecurityContextFailures(t *testing.T) {
	// fail user strat
	failUserSCC := defaultSCC()
	var uid int64 = 999
	var badUID int64 = 1
	failUserSCC.RunAsUser = api.RunAsUserStrategyOptions{
		Type: api.RunAsUserStrategyMustRunAs,
		UID:  &uid,
	}
	failUserPod := defaultPod()
	failUserPod.Spec.Containers[0].SecurityContext.RunAsUser = &badUID

	// fail selinux strat
	failSELinuxSCC := defaultSCC()
	failSELinuxSCC.SELinuxContext = api.SELinuxContextStrategyOptions{
		Type: api.SELinuxStrategyMustRunAs,
		SELinuxOptions: &api.SELinuxOptions{
			Level: "foo",
		},
	}
	failSELinuxPod := defaultPod()
	failSELinuxPod.Spec.Containers[0].SecurityContext.SELinuxOptions = &api.SELinuxOptions{
		Level: "bar",
	}

	failPrivPod := defaultPod()
	var priv bool = true
	failPrivPod.Spec.Containers[0].SecurityContext.Privileged = &priv

	failCapsPod := defaultPod()
	failCapsPod.Spec.Containers[0].SecurityContext.Capabilities = &api.Capabilities{
		Add: []api.Capability{"foo"},
	}

	failHostDirPod := defaultPod()
	failHostDirPod.Spec.Volumes = []api.Volume{
		{
			Name: "bad volume",
			VolumeSource: api.VolumeSource{
				HostPath: &api.HostPathVolumeSource{},
			},
		},
	}

	failHostPortPod := defaultPod()
	failHostPortPod.Spec.Containers[0].Ports = []api.ContainerPort{{HostPort: 1}}

	readOnlyRootFSSCC := defaultSCC()
	readOnlyRootFSSCC.ReadOnlyRootFilesystem = true

	readOnlyRootFSPodFalse := defaultPod()
	readOnlyRootFS := false
	readOnlyRootFSPodFalse.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem = &readOnlyRootFS

	failNoSeccompAllowed := defaultPod()
	failNoSeccompAllowed.Annotations[api.SeccompContainerAnnotationKeyPrefix+failNoSeccompAllowed.Spec.Containers[0].Name] = "bar"
	failNoSeccompAllowedSCC := defaultSCC()
	failNoSeccompAllowedSCC.SeccompProfiles = nil

	failInvalidSeccompProfile := defaultPod()
	failInvalidSeccompProfile.Annotations[api.SeccompContainerAnnotationKeyPrefix+failNoSeccompAllowed.Spec.Containers[0].Name] = "bar"
	failInvalidSeccompProfileSCC := defaultSCC()
	failInvalidSeccompProfileSCC.SeccompProfiles = []string{"foo"}

	errorCases := map[string]struct {
		pod           *api.Pod
		scc           *api.SecurityContextConstraints
		expectedError string
	}{
		"failUserSCC": {
			pod:           failUserPod,
			scc:           failUserSCC,
			expectedError: "does not match required UID",
		},
		"failSELinuxSCC": {
			pod:           failSELinuxPod,
			scc:           failSELinuxSCC,
			expectedError: "does not match required level",
		},
		"failPrivSCC": {
			pod:           failPrivPod,
			scc:           defaultSCC(),
			expectedError: "Privileged containers are not allowed",
		},
		"failCapsSCC": {
			pod:           failCapsPod,
			scc:           defaultSCC(),
			expectedError: "capability may not be added",
		},
		"failHostDirSCC": {
			pod:           failHostDirPod,
			scc:           defaultSCC(),
			expectedError: "hostPath volumes are not allowed to be used",
		},
		"failHostPortSCC": {
			pod:           failHostPortPod,
			scc:           defaultSCC(),
			expectedError: "Host ports are not allowed to be used",
		},
		"failReadOnlyRootFS - nil": {
			pod:           defaultPod(),
			scc:           readOnlyRootFSSCC,
			expectedError: "ReadOnlyRootFilesystem may not be nil and must be set to true",
		},
		"failReadOnlyRootFS - false": {
			pod:           readOnlyRootFSPodFalse,
			scc:           readOnlyRootFSSCC,
			expectedError: "ReadOnlyRootFilesystem must be set to true",
		},
		"failNoSeccompAllowed": {
			pod:           failNoSeccompAllowed,
			scc:           failNoSeccompAllowedSCC,
			expectedError: "seccomp may not be set",
		},
		"failInvalidSeccompPod": {
			pod:           failInvalidSeccompProfile,
			scc:           failInvalidSeccompProfileSCC,
			expectedError: "bar is not a valid seccomp profile",
		},
	}

	for k, v := range errorCases {
		provider, err := NewSimpleProvider(v.scc)
		if err != nil {
			t.Fatalf("unable to create provider %v", err)
		}
		errs := provider.ValidateContainerSecurityContext(v.pod, &v.pod.Spec.Containers[0], field.NewPath(""))
		if len(errs) == 0 {
			t.Errorf("%s expected validation failure but did not receive errors", k)
			continue
		}
		if !strings.Contains(errs[0].Error(), v.expectedError) {
			t.Errorf("%s received unexpected error %v", k, errs)
		}
	}
}

func TestValidatePodSecurityContextSuccess(t *testing.T) {
	hostNetworkSCC := defaultSCC()
	hostNetworkSCC.AllowHostNetwork = true
	hostNetworkPod := defaultPod()
	hostNetworkPod.Spec.SecurityContext.HostNetwork = true

	hostPIDSCC := defaultSCC()
	hostPIDSCC.AllowHostPID = true
	hostPIDPod := defaultPod()
	hostPIDPod.Spec.SecurityContext.HostPID = true

	hostIPCSCC := defaultSCC()
	hostIPCSCC.AllowHostIPC = true
	hostIPCPod := defaultPod()
	hostIPCPod.Spec.SecurityContext.HostIPC = true

	supGroupSCC := defaultSCC()
	supGroupSCC.SupplementalGroups = api.SupplementalGroupsStrategyOptions{
		Type: api.SupplementalGroupsStrategyMustRunAs,
		Ranges: []api.IDRange{
			{Min: 1, Max: 5},
		},
	}
	supGroupPod := defaultPod()
	supGroupPod.Spec.SecurityContext.SupplementalGroups = []int64{3}

	fsGroupSCC := defaultSCC()
	fsGroupSCC.FSGroup = api.FSGroupStrategyOptions{
		Type: api.FSGroupStrategyMustRunAs,
		Ranges: []api.IDRange{
			{Min: 1, Max: 5},
		},
	}
	fsGroupPod := defaultPod()
	fsGroup := int64(3)
	fsGroupPod.Spec.SecurityContext.FSGroup = &fsGroup

	seLinuxPod := defaultPod()
	seLinuxPod.Spec.SecurityContext.SELinuxOptions = &api.SELinuxOptions{
		User:  "user",
		Role:  "role",
		Type:  "type",
		Level: "level",
	}
	seLinuxSCC := defaultSCC()
	seLinuxSCC.SELinuxContext.Type = api.SELinuxStrategyMustRunAs
	seLinuxSCC.SELinuxContext.SELinuxOptions = &api.SELinuxOptions{
		User:  "user",
		Role:  "role",
		Type:  "type",
		Level: "level",
	}

	seccompNilWithNoProfiles := defaultPod()
	seccompNilWithNoProfilesSCC := defaultSCC()
	seccompNilWithNoProfilesSCC.SeccompProfiles = nil

	seccompEmpty := defaultPod()
	seccompEmpty.Annotations[api.SeccompPodAnnotationKey] = ""

	seccompAllowAnySCC := defaultSCC()
	seccompAllowAnySCC.SeccompProfiles = []string{"*"}

	seccompAllowFooSCC := defaultSCC()
	seccompAllowFooSCC.SeccompProfiles = []string{"foo"}

	seccompFooPod := defaultPod()
	seccompFooPod.Annotations[api.SeccompPodAnnotationKey] = "foo"

	sysctlAllowFooSCC := defaultSCC()
	sysctlAllowFooSCC.Annotations[extensions.SysctlsPodSecurityPolicyAnnotationKey] = "foo"

	safeSysctlFooPod := defaultPod()
	safeSysctlFooPod.Annotations[api.SysctlsPodAnnotationKey] = "foo=1"

	unsafeSysctlFooPod := defaultPod()
	unsafeSysctlFooPod.Annotations[api.UnsafeSysctlsPodAnnotationKey] = "foo=1"

	errorCases := map[string]struct {
		pod *api.Pod
		scc *api.SecurityContextConstraints
	}{
		"pass hostNetwork validating SCC": {
			pod: hostNetworkPod,
			scc: hostNetworkSCC,
		},
		"pass hostPID validating SCC": {
			pod: hostPIDPod,
			scc: hostPIDSCC,
		},
		"pass hostIPC validating SCC": {
			pod: hostIPCPod,
			scc: hostIPCSCC,
		},
		"pass supplemental group validating SCC": {
			pod: supGroupPod,
			scc: supGroupSCC,
		},
		"pass fs group validating SCC": {
			pod: fsGroupPod,
			scc: fsGroupSCC,
		},
		"pass selinux validating SCC": {
			pod: seLinuxPod,
			scc: seLinuxSCC,
		},
		"pass seccomp nil with no profiles": {
			pod: seccompNilWithNoProfiles,
			scc: seccompNilWithNoProfilesSCC,
		},
		"pass seccomp empty with no profiles": {
			pod: seccompEmpty,
			scc: seccompNilWithNoProfilesSCC,
		},
		"pass seccomp wild card": {
			pod: seccompFooPod,
			scc: seccompAllowAnySCC,
		},
		"pass seccomp specific profile": {
			pod: seccompFooPod,
			scc: seccompAllowFooSCC,
		},
		"pass sysctl specific profile with safe sysctl": {
			pod: safeSysctlFooPod,
			scc: sysctlAllowFooSCC,
		},
		"pass sysctl specific profile with unsafe sysctl": {
			pod: unsafeSysctlFooPod,
			scc: sysctlAllowFooSCC,
		},
		"pass empty profile with safe sysctl": {
			pod: safeSysctlFooPod,
			scc: defaultSCC(),
		},
		"pass empty profile with unsafe sysctl": {
			pod: unsafeSysctlFooPod,
			scc: defaultSCC(),
		},
	}

	for k, v := range errorCases {
		provider, err := NewSimpleProvider(v.scc)
		if err != nil {
			t.Fatalf("unable to create provider %v", err)
		}
		errs := provider.ValidatePodSecurityContext(v.pod, field.NewPath(""))
		if len(errs) != 0 {
			t.Errorf("%s expected validation pass but received errors %v", k, errs)
			continue
		}
	}
}

func TestValidateContainerSecurityContextSuccess(t *testing.T) {
	var notPriv bool = false
	defaultPod := func() *api.Pod {
		return &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Annotations: map[string]string{},
			},
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{},
				Containers: []api.Container{
					{
						SecurityContext: &api.SecurityContext{
							// expected to be set by defaulting mechanisms
							Privileged: &notPriv,
							// fill in the rest for test cases
						},
					},
				},
			},
		}
	}

	// fail user strat
	userSCC := defaultSCC()
	var uid int64 = 999
	userSCC.RunAsUser = api.RunAsUserStrategyOptions{
		Type: api.RunAsUserStrategyMustRunAs,
		UID:  &uid,
	}
	userPod := defaultPod()
	userPod.Spec.Containers[0].SecurityContext.RunAsUser = &uid

	// fail selinux strat
	seLinuxSCC := defaultSCC()
	seLinuxSCC.SELinuxContext = api.SELinuxContextStrategyOptions{
		Type: api.SELinuxStrategyMustRunAs,
		SELinuxOptions: &api.SELinuxOptions{
			Level: "foo",
		},
	}
	seLinuxPod := defaultPod()
	seLinuxPod.Spec.Containers[0].SecurityContext.SELinuxOptions = &api.SELinuxOptions{
		Level: "foo",
	}

	privSCC := defaultSCC()
	privSCC.AllowPrivilegedContainer = true
	privPod := defaultPod()
	var priv bool = true
	privPod.Spec.Containers[0].SecurityContext.Privileged = &priv

	capsSCC := defaultSCC()
	capsSCC.AllowedCapabilities = []api.Capability{"foo"}
	capsPod := defaultPod()
	capsPod.Spec.Containers[0].SecurityContext.Capabilities = &api.Capabilities{
		Add: []api.Capability{"foo"},
	}

	// pod should be able to request caps that are in the required set even if not specified in the allowed set
	requiredCapsSCC := defaultSCC()
	requiredCapsSCC.DefaultAddCapabilities = []api.Capability{"foo"}
	requiredCapsPod := defaultPod()
	requiredCapsPod.Spec.Containers[0].SecurityContext.Capabilities = &api.Capabilities{
		Add: []api.Capability{"foo"},
	}

	hostDirSCC := defaultSCC()
	hostDirSCC.Volumes = []api.FSType{api.FSTypeHostPath}
	hostDirPod := defaultPod()
	hostDirPod.Spec.Volumes = []api.Volume{
		{
			Name: "bad volume",
			VolumeSource: api.VolumeSource{
				HostPath: &api.HostPathVolumeSource{},
			},
		},
	}

	hostPortSCC := defaultSCC()
	hostPortSCC.AllowHostPorts = true
	hostPortPod := defaultPod()
	hostPortPod.Spec.Containers[0].Ports = []api.ContainerPort{{HostPort: 1}}

	readOnlyRootFSPodFalse := defaultPod()
	readOnlyRootFSFalse := false
	readOnlyRootFSPodFalse.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem = &readOnlyRootFSFalse

	readOnlyRootFSPodTrue := defaultPod()
	readOnlyRootFSTrue := true
	readOnlyRootFSPodTrue.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem = &readOnlyRootFSTrue

	seccompNilWithNoProfiles := defaultPod()
	seccompNilWithNoProfilesSCC := defaultSCC()
	seccompNilWithNoProfilesSCC.SeccompProfiles = nil

	seccompEmptyWithNoProfiles := defaultPod()
	seccompEmptyWithNoProfiles.Annotations[api.SeccompContainerAnnotationKeyPrefix+seccompEmptyWithNoProfiles.Spec.Containers[0].Name] = ""

	seccompAllowAnySCC := defaultSCC()
	seccompAllowAnySCC.SeccompProfiles = []string{"*"}

	seccompAllowFooSCC := defaultSCC()
	seccompAllowFooSCC.SeccompProfiles = []string{"foo"}

	seccompFooPod := defaultPod()
	seccompFooPod.Annotations[api.SeccompContainerAnnotationKeyPrefix+seccompFooPod.Spec.Containers[0].Name] = "foo"

	errorCases := map[string]struct {
		pod *api.Pod
		scc *api.SecurityContextConstraints
	}{
		"pass user must run as SCC": {
			pod: userPod,
			scc: userSCC,
		},
		"pass seLinux must run as SCC": {
			pod: seLinuxPod,
			scc: seLinuxSCC,
		},
		"pass priv validating SCC": {
			pod: privPod,
			scc: privSCC,
		},
		"pass allowed caps validating SCC": {
			pod: capsPod,
			scc: capsSCC,
		},
		"pass required caps validating SCC": {
			pod: requiredCapsPod,
			scc: requiredCapsSCC,
		},
		"pass hostDir validating SCC": {
			pod: hostDirPod,
			scc: hostDirSCC,
		},
		"pass hostPort validating SCC": {
			pod: hostPortPod,
			scc: hostPortSCC,
		},
		"pass read only root fs - nil": {
			pod: defaultPod(),
			scc: defaultSCC(),
		},
		"pass read only root fs - false": {
			pod: readOnlyRootFSPodFalse,
			scc: defaultSCC(),
		},
		"pass read only root fs - true": {
			pod: readOnlyRootFSPodTrue,
			scc: defaultSCC(),
		},
		"pass seccomp nil with no profiles": {
			pod: seccompNilWithNoProfiles,
			scc: seccompNilWithNoProfilesSCC,
		},
		"pass seccomp empty with no profiles": {
			pod: seccompEmptyWithNoProfiles,
			scc: seccompNilWithNoProfilesSCC,
		},
		"pass seccomp wild card": {
			pod: seccompFooPod,
			scc: seccompAllowAnySCC,
		},
		"pass seccomp specific profile": {
			pod: seccompFooPod,
			scc: seccompAllowFooSCC,
		},
	}

	for k, v := range errorCases {
		provider, err := NewSimpleProvider(v.scc)
		if err != nil {
			t.Fatalf("unable to create provider %v", err)
		}
		errs := provider.ValidateContainerSecurityContext(v.pod, &v.pod.Spec.Containers[0], field.NewPath(""))
		if len(errs) != 0 {
			t.Errorf("%s expected validation pass but received errors %v", k, errs)
			continue
		}
	}
}

func TestGenerateContainerSecurityContextReadOnlyRootFS(t *testing.T) {
	trueSCC := defaultSCC()
	trueSCC.ReadOnlyRootFilesystem = true

	trueVal := true
	expectTrue := &trueVal
	falseVal := false
	expectFalse := &falseVal

	falsePod := defaultPod()
	falsePod.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem = expectFalse

	truePod := defaultPod()
	truePod.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem = expectTrue

	tests := map[string]struct {
		pod      *api.Pod
		scc      *api.SecurityContextConstraints
		expected *bool
	}{
		"false scc, nil sc": {
			scc:      defaultSCC(),
			pod:      defaultPod(),
			expected: nil,
		},
		"false scc, false sc": {
			scc:      defaultSCC(),
			pod:      falsePod,
			expected: expectFalse,
		},
		"false scc, true sc": {
			scc:      defaultSCC(),
			pod:      truePod,
			expected: expectTrue,
		},
		"true scc, nil sc": {
			scc:      trueSCC,
			pod:      defaultPod(),
			expected: expectTrue,
		},
		"true scc, false sc": {
			scc: trueSCC,
			pod: falsePod,
			// expect false even though it defaults to true to ensure it doesn't change set values
			// validation catches the mismatch, not generation
			expected: expectFalse,
		},
		"true scc, true sc": {
			scc:      trueSCC,
			pod:      truePod,
			expected: expectTrue,
		},
	}

	for k, v := range tests {
		provider, err := NewSimpleProvider(v.scc)
		if err != nil {
			t.Errorf("%s unable to create provider %v", k, err)
			continue
		}
		sc, err := provider.CreateContainerSecurityContext(v.pod, &v.pod.Spec.Containers[0])
		if err != nil {
			t.Errorf("%s unable to create container security context %v", k, err)
			continue
		}

		if v.expected == nil && sc.ReadOnlyRootFilesystem != nil {
			t.Errorf("%s expected a nil ReadOnlyRootFilesystem but got %t", k, *sc.ReadOnlyRootFilesystem)
		}
		if v.expected != nil && sc.ReadOnlyRootFilesystem == nil {
			t.Errorf("%s expected a non nil ReadOnlyRootFilesystem but recieved nil", k)
		}
		if v.expected != nil && sc.ReadOnlyRootFilesystem != nil && (*v.expected != *sc.ReadOnlyRootFilesystem) {
			t.Errorf("%s expected a non nil ReadOnlyRootFilesystem set to %t but got %t", k, *v.expected, *sc.ReadOnlyRootFilesystem)
		}

	}
}

func defaultSCC() *api.SecurityContextConstraints {
	return &api.SecurityContextConstraints{
		ObjectMeta: api.ObjectMeta{
			Name:        "scc-sa",
			Annotations: map[string]string{},
		},
		RunAsUser: api.RunAsUserStrategyOptions{
			Type: api.RunAsUserStrategyRunAsAny,
		},
		SELinuxContext: api.SELinuxContextStrategyOptions{
			Type: api.SELinuxStrategyRunAsAny,
		},
		FSGroup: api.FSGroupStrategyOptions{
			Type: api.FSGroupStrategyRunAsAny,
		},
		SupplementalGroups: api.SupplementalGroupsStrategyOptions{
			Type: api.SupplementalGroupsStrategyRunAsAny,
		},
	}
}

func defaultPod() *api.Pod {
	var notPriv bool = false
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{Annotations: map[string]string{}},
		Spec: api.PodSpec{
			SecurityContext: &api.PodSecurityContext{
			// fill in for test cases
			},
			Containers: []api.Container{
				{
					SecurityContext: &api.SecurityContext{
						// expected to be set by defaulting mechanisms
						Privileged: &notPriv,
						// fill in the rest for test cases
					},
				},
			},
		},
	}
}

// TestValidateAllowedVolumes will test that for every field of VolumeSource we can create
// a pod with that type of volume and deny it, accept it explicitly, or accept it with
// the FSTypeAll wildcard.
func TestValidateAllowedVolumes(t *testing.T) {
	val := reflect.ValueOf(api.VolumeSource{})

	for i := 0; i < val.NumField(); i++ {
		// reflectively create the volume source
		fieldVal := val.Type().Field(i)

		volumeSource := api.VolumeSource{}
		volumeSourceVolume := reflect.New(fieldVal.Type.Elem())

		reflect.ValueOf(&volumeSource).Elem().FieldByName(fieldVal.Name).Set(volumeSourceVolume)
		volume := api.Volume{VolumeSource: volumeSource}

		// sanity check before moving on
		fsType, err := sccutil.GetVolumeFSType(volume)
		if err != nil {
			t.Errorf("error getting FSType for %s: %s", fieldVal.Name, err.Error())
			continue
		}

		// add the volume to the pod
		pod := defaultPod()
		pod.Spec.Volumes = []api.Volume{volume}

		// create an SCC that allows no volumes
		scc := defaultSCC()

		provider, err := NewSimpleProvider(scc)
		if err != nil {
			t.Errorf("error creating provider for %s: %s", fieldVal.Name, err.Error())
			continue
		}

		// expect a denial for this SCC and test the error message to ensure it's related to the volumesource
		errs := provider.ValidateContainerSecurityContext(pod, &pod.Spec.Containers[0], field.NewPath(""))
		if len(errs) != 1 {
			t.Errorf("expected exactly 1 error for %s but got %v", fieldVal.Name, errs)
		} else {
			if !strings.Contains(errs.ToAggregate().Error(), fmt.Sprintf("%s volumes are not allowed to be used", fsType)) {
				t.Errorf("did not find the expected error, received: %v", errs)
			}
		}

		// now add the fstype directly to the scc and it should validate
		scc.Volumes = []api.FSType{fsType}
		errs = provider.ValidateContainerSecurityContext(pod, &pod.Spec.Containers[0], field.NewPath(""))
		if len(errs) != 0 {
			t.Errorf("directly allowing volume expected no errors for %s but got %v", fieldVal.Name, errs)
		}

		// now change the scc to allow any volumes and the pod should still validate
		scc.Volumes = []api.FSType{api.FSTypeAll}
		errs = provider.ValidateContainerSecurityContext(pod, &pod.Spec.Containers[0], field.NewPath(""))
		if len(errs) != 0 {
			t.Errorf("wildcard volume expected no errors for %s but got %v", fieldVal.Name, errs)
		}
	}
}
