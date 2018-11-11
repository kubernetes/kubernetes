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

package podsecuritypolicy

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/security/apparmor"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/seccomp"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
)

const defaultContainerName = "test-c"

func TestDefaultPodSecurityContextNonmutating(t *testing.T) {
	// Create a pod with a security context that needs filling in
	createPod := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{},
			},
		}
	}

	// Create a PSP with strategies that will populate a blank psc
	allowPrivilegeEscalation := true
	createPSP := func() *policy.PodSecurityPolicy {
		return &policy.PodSecurityPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "psp-sa",
				Annotations: map[string]string{
					seccomp.AllowedProfilesAnnotationKey: "*",
				},
			},
			Spec: policy.PodSecurityPolicySpec{
				AllowPrivilegeEscalation: &allowPrivilegeEscalation,
				RunAsUser: policy.RunAsUserStrategyOptions{
					Rule: policy.RunAsUserStrategyRunAsAny,
				},
				RunAsGroup: &policy.RunAsGroupStrategyOptions{
					Rule: policy.RunAsGroupStrategyRunAsAny,
				},
				SELinux: policy.SELinuxStrategyOptions{
					Rule: policy.SELinuxStrategyRunAsAny,
				},
				FSGroup: policy.FSGroupStrategyOptions{
					Rule: policy.FSGroupStrategyRunAsAny,
				},
				SupplementalGroups: policy.SupplementalGroupsStrategyOptions{
					Rule: policy.SupplementalGroupsStrategyRunAsAny,
				},
			},
		}
	}

	pod := createPod()
	psp := createPSP()

	provider, err := NewSimpleProvider(psp, "namespace", NewSimpleStrategyFactory())
	if err != nil {
		t.Fatalf("unable to create provider %v", err)
	}
	err = provider.DefaultPodSecurityContext(pod)
	if err != nil {
		t.Fatalf("unable to create psc %v", err)
	}

	// Creating the provider or the security context should not have mutated the psp or pod
	// since all the strategies were permissive
	if !reflect.DeepEqual(createPod(), pod) {
		diffs := diff.ObjectDiff(createPod(), pod)
		t.Errorf("pod was mutated by DefaultPodSecurityContext. diff:\n%s", diffs)
	}
	if !reflect.DeepEqual(createPSP(), psp) {
		t.Error("psp was mutated by DefaultPodSecurityContext")
	}
}

func TestDefaultContainerSecurityContextNonmutating(t *testing.T) {
	untrue := false
	tests := []struct {
		security *api.SecurityContext
	}{
		{nil},
		{&api.SecurityContext{RunAsNonRoot: &untrue}},
	}

	for _, tc := range tests {
		// Create a pod with a security context that needs filling in
		createPod := func() *api.Pod {
			return &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						SecurityContext: tc.security,
					}},
				},
			}
		}

		// Create a PSP with strategies that will populate a blank security context
		allowPrivilegeEscalation := true
		createPSP := func() *policy.PodSecurityPolicy {
			return &policy.PodSecurityPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "psp-sa",
					Annotations: map[string]string{
						seccomp.AllowedProfilesAnnotationKey: "*",
						seccomp.DefaultProfileAnnotationKey:  "foo",
					},
				},
				Spec: policy.PodSecurityPolicySpec{
					AllowPrivilegeEscalation: &allowPrivilegeEscalation,
					RunAsUser: policy.RunAsUserStrategyOptions{
						Rule: policy.RunAsUserStrategyRunAsAny,
					},
					RunAsGroup: &policy.RunAsGroupStrategyOptions{
						Rule: policy.RunAsGroupStrategyRunAsAny,
					},
					SELinux: policy.SELinuxStrategyOptions{
						Rule: policy.SELinuxStrategyRunAsAny,
					},
					FSGroup: policy.FSGroupStrategyOptions{
						Rule: policy.FSGroupStrategyRunAsAny,
					},
					SupplementalGroups: policy.SupplementalGroupsStrategyOptions{
						Rule: policy.SupplementalGroupsStrategyRunAsAny,
					},
				},
			}
		}

		pod := createPod()
		psp := createPSP()

		provider, err := NewSimpleProvider(psp, "namespace", NewSimpleStrategyFactory())
		if err != nil {
			t.Fatalf("unable to create provider %v", err)
		}
		err = provider.DefaultContainerSecurityContext(pod, &pod.Spec.Containers[0])
		if err != nil {
			t.Fatalf("unable to create container security context %v", err)
		}

		// Creating the provider or the security context should not have mutated the psp or pod
		// since all the strategies were permissive
		if !reflect.DeepEqual(createPod(), pod) {
			diffs := diff.ObjectDiff(createPod(), pod)
			t.Errorf("pod was mutated by DefaultContainerSecurityContext. diff:\n%s", diffs)
		}
		if !reflect.DeepEqual(createPSP(), psp) {
			t.Error("psp was mutated by DefaultContainerSecurityContext")
		}
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
	failSupplementalGroupMustPSP := defaultPSP()
	failSupplementalGroupMustPSP.Spec.SupplementalGroups = policy.SupplementalGroupsStrategyOptions{
		Rule: policy.SupplementalGroupsStrategyMustRunAs,
		Ranges: []policy.IDRange{
			{Min: 1, Max: 1},
		},
	}
	failSupplementalGroupMayPSP := defaultPSP()
	failSupplementalGroupMayPSP.Spec.SupplementalGroups = policy.SupplementalGroupsStrategyOptions{
		Rule: policy.SupplementalGroupsStrategyMayRunAs,
		Ranges: []policy.IDRange{
			{Min: 50, Max: 50},
			{Min: 55, Max: 998},
			{Min: 1000, Max: 1000},
		},
	}

	failFSGroupPod := defaultPod()
	fsGroup := int64(999)
	failFSGroupPod.Spec.SecurityContext.FSGroup = &fsGroup
	failFSGroupMustPSP := defaultPSP()
	failFSGroupMustPSP.Spec.FSGroup = policy.FSGroupStrategyOptions{
		Rule: policy.FSGroupStrategyMustRunAs,
		Ranges: []policy.IDRange{
			{Min: 1, Max: 1},
		},
	}
	failFSGroupMayPSP := defaultPSP()
	failFSGroupMayPSP.Spec.FSGroup = policy.FSGroupStrategyOptions{
		Rule: policy.FSGroupStrategyMayRunAs,
		Ranges: []policy.IDRange{
			{Min: 10, Max: 20},
			{Min: 1000, Max: 1001},
		},
	}

	failNilSELinuxPod := defaultPod()
	failSELinuxPSP := defaultPSP()
	failSELinuxPSP.Spec.SELinux.Rule = policy.SELinuxStrategyMustRunAs
	failSELinuxPSP.Spec.SELinux.SELinuxOptions = &v1.SELinuxOptions{
		Level: "foo",
	}

	failInvalidSELinuxPod := defaultPod()
	failInvalidSELinuxPod.Spec.SecurityContext.SELinuxOptions = &api.SELinuxOptions{
		Level: "bar",
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

	failHostPathDirPod := defaultPod()
	failHostPathDirPod.Spec.Volumes = []api.Volume{
		{
			Name: "bad volume",
			VolumeSource: api.VolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: "/fail",
				},
			},
		},
	}
	failHostPathDirPSP := defaultPSP()
	failHostPathDirPSP.Spec.Volumes = []policy.FSType{policy.HostPath}
	failHostPathDirPSP.Spec.AllowedHostPaths = []policy.AllowedHostPath{
		{PathPrefix: "/foo/bar"},
	}

	failHostPathReadOnlyPod := defaultPod()
	failHostPathReadOnlyPod.Spec.Containers[0].VolumeMounts = []api.VolumeMount{
		{
			Name:     "bad volume",
			ReadOnly: false,
		},
	}
	failHostPathReadOnlyPod.Spec.Volumes = []api.Volume{
		{
			Name: "bad volume",
			VolumeSource: api.VolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: "/foo",
				},
			},
		},
	}
	failHostPathReadOnlyPSP := defaultPSP()
	failHostPathReadOnlyPSP.Spec.Volumes = []policy.FSType{policy.HostPath}
	failHostPathReadOnlyPSP.Spec.AllowedHostPaths = []policy.AllowedHostPath{
		{
			PathPrefix: "/foo",
			ReadOnly:   true,
		},
	}

	failSysctlDisallowedPSP := defaultPSP()
	failSysctlDisallowedPSP.Spec.ForbiddenSysctls = []string{"kernel.shm_rmid_forced"}

	failNoSafeSysctlAllowedPSP := defaultPSP()
	failNoSafeSysctlAllowedPSP.Spec.ForbiddenSysctls = []string{"*"}

	failAllUnsafeSysctlsPSP := defaultPSP()
	failAllUnsafeSysctlsPSP.Spec.AllowedUnsafeSysctls = []string{}

	failSafeSysctlKernelPod := defaultPod()
	failSafeSysctlKernelPod.Spec.SecurityContext = &api.PodSecurityContext{
		Sysctls: []api.Sysctl{
			{
				Name:  "kernel.shm_rmid_forced",
				Value: "1",
			},
		},
	}

	failUnsafeSysctlPod := defaultPod()
	failUnsafeSysctlPod.Spec.SecurityContext = &api.PodSecurityContext{
		Sysctls: []api.Sysctl{
			{
				Name:  "kernel.sem",
				Value: "32000",
			},
		},
	}

	failSeccompProfilePod := defaultPod()
	failSeccompProfilePod.Annotations = map[string]string{api.SeccompPodAnnotationKey: "foo"}

	podWithInvalidFlexVolumeDriver := defaultPod()
	podWithInvalidFlexVolumeDriver.Spec.Volumes = []api.Volume{
		{
			Name: "flex-volume",
			VolumeSource: api.VolumeSource{
				FlexVolume: &api.FlexVolumeSource{
					Driver: "example/unknown",
				},
			},
		},
	}

	errorCases := map[string]struct {
		pod           *api.Pod
		psp           *policy.PodSecurityPolicy
		expectedError string
	}{
		"failHostNetwork": {
			pod:           failHostNetworkPod,
			psp:           defaultPSP(),
			expectedError: "Host network is not allowed to be used",
		},
		"failHostPID": {
			pod:           failHostPIDPod,
			psp:           defaultPSP(),
			expectedError: "Host PID is not allowed to be used",
		},
		"failHostIPC": {
			pod:           failHostIPCPod,
			psp:           defaultPSP(),
			expectedError: "Host IPC is not allowed to be used",
		},
		"failSupplementalGroupOutOfMustRange": {
			pod:           failSupplementalGroupPod,
			psp:           failSupplementalGroupMustPSP,
			expectedError: "group 999 must be in the ranges: [{1 1}]",
		},
		"failSupplementalGroupOutOfMayRange": {
			pod:           failSupplementalGroupPod,
			psp:           failSupplementalGroupMayPSP,
			expectedError: "group 999 must be in the ranges: [{50 50} {55 998} {1000 1000}]",
		},
		"failSupplementalGroupMustEmpty": {
			pod:           defaultPod(),
			psp:           failSupplementalGroupMustPSP,
			expectedError: "unable to validate empty groups against required ranges",
		},
		"failFSGroupOutOfMustRange": {
			pod:           failFSGroupPod,
			psp:           failFSGroupMustPSP,
			expectedError: "group 999 must be in the ranges: [{1 1}]",
		},
		"failFSGroupOutOfMayRange": {
			pod:           failFSGroupPod,
			psp:           failFSGroupMayPSP,
			expectedError: "group 999 must be in the ranges: [{10 20} {1000 1001}]",
		},
		"failFSGroupMustEmpty": {
			pod:           defaultPod(),
			psp:           failFSGroupMustPSP,
			expectedError: "unable to validate empty groups against required ranges",
		},
		"failNilSELinux": {
			pod:           failNilSELinuxPod,
			psp:           failSELinuxPSP,
			expectedError: "seLinuxOptions: Required",
		},
		"failInvalidSELinux": {
			pod:           failInvalidSELinuxPod,
			psp:           failSELinuxPSP,
			expectedError: "seLinuxOptions.level: Invalid value",
		},
		"failHostDirPSP": {
			pod:           failHostDirPod,
			psp:           defaultPSP(),
			expectedError: "hostPath volumes are not allowed to be used",
		},
		"failHostPathDirPSP": {
			pod:           failHostPathDirPod,
			psp:           failHostPathDirPSP,
			expectedError: "is not allowed to be used",
		},
		"failHostPathReadOnlyPSP": {
			pod:           failHostPathReadOnlyPod,
			psp:           failHostPathReadOnlyPSP,
			expectedError: "must be read-only",
		},
		"failSafeSysctlKernelPod with failNoSafeSysctlAllowedPSP": {
			pod:           failSafeSysctlKernelPod,
			psp:           failNoSafeSysctlAllowedPSP,
			expectedError: "sysctl \"kernel.shm_rmid_forced\" is not allowed",
		},
		"failSafeSysctlKernelPod with failSysctlDisallowedPSP": {
			pod:           failSafeSysctlKernelPod,
			psp:           failSysctlDisallowedPSP,
			expectedError: "sysctl \"kernel.shm_rmid_forced\" is not allowed",
		},
		"failUnsafeSysctlPod with failAllUnsafeSysctlsPSP": {
			pod:           failUnsafeSysctlPod,
			psp:           failAllUnsafeSysctlsPSP,
			expectedError: "unsafe sysctl \"kernel.sem\" is not allowed",
		},
		"failInvalidSeccomp": {
			pod:           failSeccompProfilePod,
			psp:           defaultPSP(),
			expectedError: "Forbidden: seccomp may not be set",
		},
		"fail pod with disallowed flexVolume when flex volumes are allowed": {
			pod:           podWithInvalidFlexVolumeDriver,
			psp:           allowFlexVolumesPSP(false, false),
			expectedError: "Flexvolume driver is not allowed to be used",
		},
		"fail pod with disallowed flexVolume when all volumes are allowed": {
			pod:           podWithInvalidFlexVolumeDriver,
			psp:           allowFlexVolumesPSP(false, true),
			expectedError: "Flexvolume driver is not allowed to be used",
		},
	}
	for k, v := range errorCases {
		provider, err := NewSimpleProvider(v.psp, "namespace", NewSimpleStrategyFactory())
		if err != nil {
			t.Fatalf("unable to create provider %v", err)
		}
		errs := provider.ValidatePod(v.pod)
		if len(errs) == 0 {
			t.Errorf("%s expected validation failure but did not receive errors", k)
			continue
		}
		if !strings.Contains(errs[0].Error(), v.expectedError) {
			t.Errorf("%s received unexpected error %v", k, errs)
		}
	}
}

func allowFlexVolumesPSP(allowAllFlexVolumes, allowAllVolumes bool) *policy.PodSecurityPolicy {
	psp := defaultPSP()

	allowedVolumes := []policy.AllowedFlexVolume{
		{Driver: "example/foo"},
		{Driver: "example/bar"},
	}
	if allowAllFlexVolumes {
		allowedVolumes = []policy.AllowedFlexVolume{}
	}

	allowedVolumeType := policy.FlexVolume
	if allowAllVolumes {
		allowedVolumeType = policy.All
	}

	psp.Spec.AllowedFlexVolumes = allowedVolumes
	psp.Spec.Volumes = []policy.FSType{allowedVolumeType}

	return psp
}

func TestValidateContainerFailures(t *testing.T) {
	// fail user strategy
	failUserPSP := defaultPSP()
	uid := int64(999)
	badUID := int64(1)
	failUserPSP.Spec.RunAsUser = policy.RunAsUserStrategyOptions{
		Rule:   policy.RunAsUserStrategyMustRunAs,
		Ranges: []policy.IDRange{{Min: uid, Max: uid}},
	}
	failUserPod := defaultPod()
	failUserPod.Spec.Containers[0].SecurityContext.RunAsUser = &badUID

	// fail selinux strategy
	failSELinuxPSP := defaultPSP()
	failSELinuxPSP.Spec.SELinux = policy.SELinuxStrategyOptions{
		Rule: policy.SELinuxStrategyMustRunAs,
		SELinuxOptions: &v1.SELinuxOptions{
			Level: "foo",
		},
	}
	failSELinuxPod := defaultPod()
	failSELinuxPod.Spec.Containers[0].SecurityContext.SELinuxOptions = &api.SELinuxOptions{
		Level: "bar",
	}

	failNilAppArmorPod := defaultPod()
	v1FailInvalidAppArmorPod := defaultV1Pod()
	apparmor.SetProfileName(v1FailInvalidAppArmorPod, defaultContainerName, apparmor.ProfileNamePrefix+"foo")
	failInvalidAppArmorPod := &api.Pod{}
	k8s_api_v1.Convert_v1_Pod_To_core_Pod(v1FailInvalidAppArmorPod, failInvalidAppArmorPod, nil)

	failAppArmorPSP := defaultPSP()
	failAppArmorPSP.Annotations = map[string]string{
		apparmor.AllowedProfilesAnnotationKey: apparmor.ProfileRuntimeDefault,
	}

	failPrivPod := defaultPod()
	var priv bool = true
	failPrivPod.Spec.Containers[0].SecurityContext.Privileged = &priv

	failProcMountPod := defaultPod()
	failProcMountPod.Spec.Containers[0].SecurityContext.ProcMount = new(api.ProcMountType)
	*failProcMountPod.Spec.Containers[0].SecurityContext.ProcMount = api.UnmaskedProcMount

	failCapsPod := defaultPod()
	failCapsPod.Spec.Containers[0].SecurityContext.Capabilities = &api.Capabilities{
		Add: []api.Capability{"foo"},
	}

	failHostPortPod := defaultPod()
	failHostPortPod.Spec.Containers[0].Ports = []api.ContainerPort{{HostPort: 1}}

	readOnlyRootFSPSP := defaultPSP()
	readOnlyRootFSPSP.Spec.ReadOnlyRootFilesystem = true

	readOnlyRootFSPodFalse := defaultPod()
	readOnlyRootFS := false
	readOnlyRootFSPodFalse.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem = &readOnlyRootFS

	failSeccompPod := defaultPod()
	failSeccompPod.Annotations = map[string]string{
		api.SeccompContainerAnnotationKeyPrefix + failSeccompPod.Spec.Containers[0].Name: "foo",
	}

	failSeccompPodInheritPodAnnotation := defaultPod()
	failSeccompPodInheritPodAnnotation.Annotations = map[string]string{
		api.SeccompPodAnnotationKey: "foo",
	}

	errorCases := map[string]struct {
		pod           *api.Pod
		psp           *policy.PodSecurityPolicy
		expectedError string
	}{
		"failUserPSP": {
			pod:           failUserPod,
			psp:           failUserPSP,
			expectedError: "runAsUser: Invalid value",
		},
		"failSELinuxPSP": {
			pod:           failSELinuxPod,
			psp:           failSELinuxPSP,
			expectedError: "seLinuxOptions.level: Invalid value",
		},
		"failNilAppArmor": {
			pod:           failNilAppArmorPod,
			psp:           failAppArmorPSP,
			expectedError: "AppArmor profile must be set",
		},
		"failInvalidAppArmor": {
			pod:           failInvalidAppArmorPod,
			psp:           failAppArmorPSP,
			expectedError: "localhost/foo is not an allowed profile. Allowed values: \"runtime/default\"",
		},
		"failPrivPSP": {
			pod:           failPrivPod,
			psp:           defaultPSP(),
			expectedError: "Privileged containers are not allowed",
		},
		"failProcMountPSP": {
			pod:           failProcMountPod,
			psp:           defaultPSP(),
			expectedError: "ProcMountType is not allowed",
		},
		"failCapsPSP": {
			pod:           failCapsPod,
			psp:           defaultPSP(),
			expectedError: "capability may not be added",
		},
		"failHostPortPSP": {
			pod:           failHostPortPod,
			psp:           defaultPSP(),
			expectedError: "Host port 1 is not allowed to be used. Allowed ports: []",
		},
		"failReadOnlyRootFS - nil": {
			pod:           defaultPod(),
			psp:           readOnlyRootFSPSP,
			expectedError: "ReadOnlyRootFilesystem may not be nil and must be set to true",
		},
		"failReadOnlyRootFS - false": {
			pod:           readOnlyRootFSPodFalse,
			psp:           readOnlyRootFSPSP,
			expectedError: "ReadOnlyRootFilesystem must be set to true",
		},
		"failSeccompContainerAnnotation": {
			pod:           failSeccompPod,
			psp:           defaultPSP(),
			expectedError: "Forbidden: seccomp may not be set",
		},
		"failSeccompContainerPodAnnotation": {
			pod:           failSeccompPodInheritPodAnnotation,
			psp:           defaultPSP(),
			expectedError: "Forbidden: seccomp may not be set",
		},
	}

	for k, v := range errorCases {
		provider, err := NewSimpleProvider(v.psp, "namespace", NewSimpleStrategyFactory())
		if err != nil {
			t.Fatalf("unable to create provider %v", err)
		}
		errs := provider.ValidateContainer(v.pod, &v.pod.Spec.Containers[0], field.NewPath(""))
		if len(errs) == 0 {
			t.Errorf("%s expected validation failure but did not receive errors", k)
			continue
		}
		if !strings.Contains(errs[0].Error(), v.expectedError) {
			t.Errorf("%s received unexpected error %v\nexpected: %s", k, errs, v.expectedError)
		}
	}
}

func TestValidatePodSecurityContextSuccess(t *testing.T) {
	hostNetworkPSP := defaultPSP()
	hostNetworkPSP.Spec.HostNetwork = true
	hostNetworkPod := defaultPod()
	hostNetworkPod.Spec.SecurityContext.HostNetwork = true

	hostPIDPSP := defaultPSP()
	hostPIDPSP.Spec.HostPID = true
	hostPIDPod := defaultPod()
	hostPIDPod.Spec.SecurityContext.HostPID = true

	hostIPCPSP := defaultPSP()
	hostIPCPSP.Spec.HostIPC = true
	hostIPCPod := defaultPod()
	hostIPCPod.Spec.SecurityContext.HostIPC = true

	supGroupMustPSP := defaultPSP()
	supGroupMustPSP.Spec.SupplementalGroups = policy.SupplementalGroupsStrategyOptions{
		Rule: policy.SupplementalGroupsStrategyMustRunAs,
		Ranges: []policy.IDRange{
			{Min: 1, Max: 5},
		},
	}
	supGroupMayPSP := defaultPSP()
	supGroupMayPSP.Spec.SupplementalGroups = policy.SupplementalGroupsStrategyOptions{
		Rule: policy.SupplementalGroupsStrategyMayRunAs,
		Ranges: []policy.IDRange{
			{Min: 1, Max: 5},
		},
	}
	supGroupPod := defaultPod()
	supGroupPod.Spec.SecurityContext.SupplementalGroups = []int64{3}

	fsGroupMustPSP := defaultPSP()
	fsGroupMustPSP.Spec.FSGroup = policy.FSGroupStrategyOptions{
		Rule: policy.FSGroupStrategyMustRunAs,
		Ranges: []policy.IDRange{
			{Min: 1, Max: 5},
		},
	}
	fsGroupMayPSP := defaultPSP()
	fsGroupMayPSP.Spec.FSGroup = policy.FSGroupStrategyOptions{
		Rule: policy.FSGroupStrategyMayRunAs,
		Ranges: []policy.IDRange{
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
	seLinuxPSP := defaultPSP()
	seLinuxPSP.Spec.SELinux.Rule = policy.SELinuxStrategyMustRunAs
	seLinuxPSP.Spec.SELinux.SELinuxOptions = &v1.SELinuxOptions{
		User:  "user",
		Role:  "role",
		Type:  "type",
		Level: "level",
	}

	hostPathDirPodVolumeMounts := []api.VolumeMount{
		{
			Name:     "writeable /foo/bar",
			ReadOnly: false,
		},
		{
			Name:     "read only /foo/bar/baz",
			ReadOnly: true,
		},
		{
			Name:     "parent read only volume",
			ReadOnly: true,
		},
		{
			Name:     "read only child volume",
			ReadOnly: true,
		},
	}

	hostPathDirPod := defaultPod()
	hostPathDirPod.Spec.InitContainers = []api.Container{
		{
			Name:         defaultContainerName,
			VolumeMounts: hostPathDirPodVolumeMounts,
		},
	}

	hostPathDirPod.Spec.Containers[0].VolumeMounts = hostPathDirPodVolumeMounts
	hostPathDirPod.Spec.Volumes = []api.Volume{
		{
			Name: "writeable /foo/bar",
			VolumeSource: api.VolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: "/foo/bar",
				},
			},
		},
		{
			Name: "read only /foo/bar/baz",
			VolumeSource: api.VolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: "/foo/bar/baz",
				},
			},
		},
		{
			Name: "parent read only volume",
			VolumeSource: api.VolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: "/foo/",
				},
			},
		},
		{
			Name: "read only child volume",
			VolumeSource: api.VolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: "/foo/readonly/child",
				},
			},
		},
	}

	hostPathDirPSP := defaultPSP()
	hostPathDirPSP.Spec.Volumes = []policy.FSType{policy.HostPath}
	hostPathDirPSP.Spec.AllowedHostPaths = []policy.AllowedHostPath{
		// overlapping test case where child is different than parent directory.
		{PathPrefix: "/foo/bar/baz", ReadOnly: true},
		{PathPrefix: "/foo", ReadOnly: true},
		{PathPrefix: "/foo/bar", ReadOnly: false},
	}

	hostPathDirAsterisksPSP := defaultPSP()
	hostPathDirAsterisksPSP.Spec.Volumes = []policy.FSType{policy.All}
	hostPathDirAsterisksPSP.Spec.AllowedHostPaths = []policy.AllowedHostPath{
		{PathPrefix: "/foo"},
	}

	sysctlAllowAllPSP := defaultPSP()
	sysctlAllowAllPSP.Spec.ForbiddenSysctls = []string{}
	sysctlAllowAllPSP.Spec.AllowedUnsafeSysctls = []string{"*"}

	safeSysctlKernelPod := defaultPod()
	safeSysctlKernelPod.Spec.SecurityContext = &api.PodSecurityContext{
		Sysctls: []api.Sysctl{
			{
				Name:  "kernel.shm_rmid_forced",
				Value: "1",
			},
		},
	}

	unsafeSysctlKernelPod := defaultPod()
	unsafeSysctlKernelPod.Spec.SecurityContext = &api.PodSecurityContext{
		Sysctls: []api.Sysctl{
			{
				Name:  "kernel.sem",
				Value: "32000",
			},
		},
	}

	seccompPSP := defaultPSP()
	seccompPSP.Annotations = map[string]string{
		seccomp.AllowedProfilesAnnotationKey: "foo",
	}

	seccompPod := defaultPod()
	seccompPod.Annotations = map[string]string{
		api.SeccompPodAnnotationKey: "foo",
	}

	flexVolumePod := defaultPod()
	flexVolumePod.Spec.Volumes = []api.Volume{
		{
			Name: "flex-volume",
			VolumeSource: api.VolumeSource{
				FlexVolume: &api.FlexVolumeSource{
					Driver: "example/bar",
				},
			},
		},
	}

	successCases := map[string]struct {
		pod *api.Pod
		psp *policy.PodSecurityPolicy
	}{
		"pass hostNetwork validating PSP": {
			pod: hostNetworkPod,
			psp: hostNetworkPSP,
		},
		"pass hostPID validating PSP": {
			pod: hostPIDPod,
			psp: hostPIDPSP,
		},
		"pass hostIPC validating PSP": {
			pod: hostIPCPod,
			psp: hostIPCPSP,
		},
		"pass required supplemental group validating PSP": {
			pod: supGroupPod,
			psp: supGroupMustPSP,
		},
		"pass optional supplemental group validation PSP": {
			pod: supGroupPod,
			psp: supGroupMayPSP,
		},
		"pass optional supplemental group validation PSP - no pod group specified": {
			pod: defaultPod(),
			psp: supGroupMayPSP,
		},
		"pass required fs group validating PSP": {
			pod: fsGroupPod,
			psp: fsGroupMustPSP,
		},
		"pass optional fs group validating PSP": {
			pod: fsGroupPod,
			psp: fsGroupMayPSP,
		},
		"pass optional fs group validating PSP - no pod group specified": {
			pod: defaultPod(),
			psp: fsGroupMayPSP,
		},
		"pass selinux validating PSP": {
			pod: seLinuxPod,
			psp: seLinuxPSP,
		},
		"pass sysctl specific profile with safe kernel sysctl": {
			pod: safeSysctlKernelPod,
			psp: sysctlAllowAllPSP,
		},
		"pass sysctl specific profile with unsafe kernel sysctl": {
			pod: unsafeSysctlKernelPod,
			psp: sysctlAllowAllPSP,
		},
		"pass hostDir allowed directory validating PSP": {
			pod: hostPathDirPod,
			psp: hostPathDirPSP,
		},
		"pass hostDir all volumes allowed validating PSP": {
			pod: hostPathDirPod,
			psp: hostPathDirAsterisksPSP,
		},
		"pass seccomp validating PSP": {
			pod: seccompPod,
			psp: seccompPSP,
		},
		"flex volume driver in a whitelist (all volumes are allowed)": {
			pod: flexVolumePod,
			psp: allowFlexVolumesPSP(false, true),
		},
		"flex volume driver with empty whitelist (all volumes are allowed)": {
			pod: flexVolumePod,
			psp: allowFlexVolumesPSP(true, true),
		},
		"flex volume driver in a whitelist (only flex volumes are allowed)": {
			pod: flexVolumePod,
			psp: allowFlexVolumesPSP(false, false),
		},
		"flex volume driver with empty whitelist (only flex volumes volumes are allowed)": {
			pod: flexVolumePod,
			psp: allowFlexVolumesPSP(true, false),
		},
	}

	for k, v := range successCases {
		provider, err := NewSimpleProvider(v.psp, "namespace", NewSimpleStrategyFactory())
		if err != nil {
			t.Fatalf("unable to create provider %v", err)
		}
		errs := provider.ValidatePod(v.pod)
		if len(errs) != 0 {
			t.Errorf("%s expected validation pass but received errors %v", k, errs)
			continue
		}
	}
}

func TestValidateContainerSuccess(t *testing.T) {
	// success user strategy
	userPSP := defaultPSP()
	uid := int64(999)
	userPSP.Spec.RunAsUser = policy.RunAsUserStrategyOptions{
		Rule:   policy.RunAsUserStrategyMustRunAs,
		Ranges: []policy.IDRange{{Min: uid, Max: uid}},
	}
	userPod := defaultPod()
	userPod.Spec.Containers[0].SecurityContext.RunAsUser = &uid

	// success selinux strategy
	seLinuxPSP := defaultPSP()
	seLinuxPSP.Spec.SELinux = policy.SELinuxStrategyOptions{
		Rule: policy.SELinuxStrategyMustRunAs,
		SELinuxOptions: &v1.SELinuxOptions{
			Level: "foo",
		},
	}
	seLinuxPod := defaultPod()
	seLinuxPod.Spec.Containers[0].SecurityContext.SELinuxOptions = &api.SELinuxOptions{
		Level: "foo",
	}

	appArmorPSP := defaultPSP()
	appArmorPSP.Annotations = map[string]string{
		apparmor.AllowedProfilesAnnotationKey: apparmor.ProfileRuntimeDefault,
	}
	v1AppArmorPod := defaultV1Pod()
	apparmor.SetProfileName(v1AppArmorPod, defaultContainerName, apparmor.ProfileRuntimeDefault)
	appArmorPod := &api.Pod{}
	k8s_api_v1.Convert_v1_Pod_To_core_Pod(v1AppArmorPod, appArmorPod, nil)

	privPSP := defaultPSP()
	privPSP.Spec.Privileged = true
	privPod := defaultPod()
	var priv bool = true
	privPod.Spec.Containers[0].SecurityContext.Privileged = &priv

	capsPSP := defaultPSP()
	capsPSP.Spec.AllowedCapabilities = []v1.Capability{"foo"}
	capsPod := defaultPod()
	capsPod.Spec.Containers[0].SecurityContext.Capabilities = &api.Capabilities{
		Add: []api.Capability{"foo"},
	}

	// pod should be able to request caps that are in the required set even if not specified in the allowed set
	requiredCapsPSP := defaultPSP()
	requiredCapsPSP.Spec.DefaultAddCapabilities = []v1.Capability{"foo"}
	requiredCapsPod := defaultPod()
	requiredCapsPod.Spec.Containers[0].SecurityContext.Capabilities = &api.Capabilities{
		Add: []api.Capability{"foo"},
	}

	hostDirPSP := defaultPSP()
	hostDirPSP.Spec.Volumes = []policy.FSType{policy.HostPath}
	hostDirPod := defaultPod()
	hostDirPod.Spec.Volumes = []api.Volume{
		{
			Name: "bad volume",
			VolumeSource: api.VolumeSource{
				HostPath: &api.HostPathVolumeSource{},
			},
		},
	}

	hostPortPSP := defaultPSP()
	hostPortPSP.Spec.HostPorts = []policy.HostPortRange{{Min: 1, Max: 1}}
	hostPortPod := defaultPod()
	hostPortPod.Spec.Containers[0].Ports = []api.ContainerPort{{HostPort: 1}}

	readOnlyRootFSPodFalse := defaultPod()
	readOnlyRootFSFalse := false
	readOnlyRootFSPodFalse.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem = &readOnlyRootFSFalse

	readOnlyRootFSPodTrue := defaultPod()
	readOnlyRootFSTrue := true
	readOnlyRootFSPodTrue.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem = &readOnlyRootFSTrue

	seccompPSP := defaultPSP()
	seccompPSP.Annotations = map[string]string{
		seccomp.AllowedProfilesAnnotationKey: "foo",
	}

	seccompPod := defaultPod()
	seccompPod.Annotations = map[string]string{
		api.SeccompContainerAnnotationKeyPrefix + seccompPod.Spec.Containers[0].Name: "foo",
	}

	seccompPodInherit := defaultPod()
	seccompPodInherit.Annotations = map[string]string{
		api.SeccompPodAnnotationKey: "foo",
	}

	successCases := map[string]struct {
		pod *api.Pod
		psp *policy.PodSecurityPolicy
	}{
		"pass user must run as PSP": {
			pod: userPod,
			psp: userPSP,
		},
		"pass seLinux must run as PSP": {
			pod: seLinuxPod,
			psp: seLinuxPSP,
		},
		"pass AppArmor allowed profiles": {
			pod: appArmorPod,
			psp: appArmorPSP,
		},
		"pass priv validating PSP": {
			pod: privPod,
			psp: privPSP,
		},
		"pass allowed caps validating PSP": {
			pod: capsPod,
			psp: capsPSP,
		},
		"pass required caps validating PSP": {
			pod: requiredCapsPod,
			psp: requiredCapsPSP,
		},
		"pass hostDir validating PSP": {
			pod: hostDirPod,
			psp: hostDirPSP,
		},
		"pass hostPort validating PSP": {
			pod: hostPortPod,
			psp: hostPortPSP,
		},
		"pass read only root fs - nil": {
			pod: defaultPod(),
			psp: defaultPSP(),
		},
		"pass read only root fs - false": {
			pod: readOnlyRootFSPodFalse,
			psp: defaultPSP(),
		},
		"pass read only root fs - true": {
			pod: readOnlyRootFSPodTrue,
			psp: defaultPSP(),
		},
		"pass seccomp container annotation": {
			pod: seccompPod,
			psp: seccompPSP,
		},
		"pass seccomp inherit pod annotation": {
			pod: seccompPodInherit,
			psp: seccompPSP,
		},
	}

	for k, v := range successCases {
		provider, err := NewSimpleProvider(v.psp, "namespace", NewSimpleStrategyFactory())
		if err != nil {
			t.Fatalf("unable to create provider %v", err)
		}
		errs := provider.ValidateContainer(v.pod, &v.pod.Spec.Containers[0], field.NewPath(""))
		if len(errs) != 0 {
			t.Errorf("%s expected validation pass but received errors %v\n%s", k, errs, spew.Sdump(v.pod.ObjectMeta))
			continue
		}
	}
}

func TestGenerateContainerSecurityContextReadOnlyRootFS(t *testing.T) {
	truePSP := defaultPSP()
	truePSP.Spec.ReadOnlyRootFilesystem = true

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
		psp      *policy.PodSecurityPolicy
		expected *bool
	}{
		"false psp, nil sc": {
			psp:      defaultPSP(),
			pod:      defaultPod(),
			expected: nil,
		},
		"false psp, false sc": {
			psp:      defaultPSP(),
			pod:      falsePod,
			expected: expectFalse,
		},
		"false psp, true sc": {
			psp:      defaultPSP(),
			pod:      truePod,
			expected: expectTrue,
		},
		"true psp, nil sc": {
			psp:      truePSP,
			pod:      defaultPod(),
			expected: expectTrue,
		},
		"true psp, false sc": {
			psp: truePSP,
			pod: falsePod,
			// expect false even though it defaults to true to ensure it doesn't change set values
			// validation catches the mismatch, not generation
			expected: expectFalse,
		},
		"true psp, true sc": {
			psp:      truePSP,
			pod:      truePod,
			expected: expectTrue,
		},
	}

	for k, v := range tests {
		provider, err := NewSimpleProvider(v.psp, "namespace", NewSimpleStrategyFactory())
		if err != nil {
			t.Errorf("%s unable to create provider %v", k, err)
			continue
		}
		err = provider.DefaultContainerSecurityContext(v.pod, &v.pod.Spec.Containers[0])
		if err != nil {
			t.Errorf("%s unable to create container security context %v", k, err)
			continue
		}

		sc := v.pod.Spec.Containers[0].SecurityContext
		if v.expected == nil && sc.ReadOnlyRootFilesystem != nil {
			t.Errorf("%s expected a nil ReadOnlyRootFilesystem but got %t", k, *sc.ReadOnlyRootFilesystem)
		}
		if v.expected != nil && sc.ReadOnlyRootFilesystem == nil {
			t.Errorf("%s expected a non nil ReadOnlyRootFilesystem but received nil", k)
		}
		if v.expected != nil && sc.ReadOnlyRootFilesystem != nil && (*v.expected != *sc.ReadOnlyRootFilesystem) {
			t.Errorf("%s expected a non nil ReadOnlyRootFilesystem set to %t but got %t", k, *v.expected, *sc.ReadOnlyRootFilesystem)
		}

	}
}

func defaultPSP() *policy.PodSecurityPolicy {
	allowPrivilegeEscalation := true
	return &policy.PodSecurityPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "psp-sa",
			Annotations: map[string]string{},
		},
		Spec: policy.PodSecurityPolicySpec{
			RunAsUser: policy.RunAsUserStrategyOptions{
				Rule: policy.RunAsUserStrategyRunAsAny,
			},
			RunAsGroup: &policy.RunAsGroupStrategyOptions{
				Rule: policy.RunAsGroupStrategyRunAsAny,
			},
			SELinux: policy.SELinuxStrategyOptions{
				Rule: policy.SELinuxStrategyRunAsAny,
			},
			FSGroup: policy.FSGroupStrategyOptions{
				Rule: policy.FSGroupStrategyRunAsAny,
			},
			SupplementalGroups: policy.SupplementalGroupsStrategyOptions{
				Rule: policy.SupplementalGroupsStrategyRunAsAny,
			},
			AllowPrivilegeEscalation: &allowPrivilegeEscalation,
		},
	}
}

func defaultPod() *api.Pod {
	var notPriv bool = false
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{},
		},
		Spec: api.PodSpec{
			SecurityContext: &api.PodSecurityContext{
				// fill in for test cases
			},
			Containers: []api.Container{
				{
					Name: defaultContainerName,
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

func defaultV1Pod() *v1.Pod {
	var notPriv bool = false
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{},
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{
				// fill in for test cases
			},
			Containers: []v1.Container{
				{
					Name: defaultContainerName,
					SecurityContext: &v1.SecurityContext{
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
		fsType, err := psputil.GetVolumeFSType(volume)
		if err != nil {
			t.Errorf("error getting FSType for %s: %s", fieldVal.Name, err.Error())
			continue
		}

		// add the volume to the pod
		pod := defaultPod()
		pod.Spec.Volumes = []api.Volume{volume}

		// create a PSP that allows no volumes
		psp := defaultPSP()

		provider, err := NewSimpleProvider(psp, "namespace", NewSimpleStrategyFactory())
		if err != nil {
			t.Errorf("error creating provider for %s: %s", fieldVal.Name, err.Error())
			continue
		}

		// expect a denial for this PSP and test the error message to ensure it's related to the volumesource
		errs := provider.ValidatePod(pod)
		if len(errs) != 1 {
			t.Errorf("expected exactly 1 error for %s but got %v", fieldVal.Name, errs)
		} else {
			if !strings.Contains(errs.ToAggregate().Error(), fmt.Sprintf("%s volumes are not allowed to be used", fsType)) {
				t.Errorf("did not find the expected error, received: %v", errs)
			}
		}

		// now add the fstype directly to the psp and it should validate
		psp.Spec.Volumes = []policy.FSType{fsType}
		errs = provider.ValidatePod(pod)
		if len(errs) != 0 {
			t.Errorf("directly allowing volume expected no errors for %s but got %v", fieldVal.Name, errs)
		}

		// now change the psp to allow any volumes and the pod should still validate
		psp.Spec.Volumes = []policy.FSType{policy.All}
		errs = provider.ValidatePod(pod)
		if len(errs) != 0 {
			t.Errorf("wildcard volume expected no errors for %s but got %v", fieldVal.Name, errs)
		}
	}
}

func TestAllowPrivilegeEscalation(t *testing.T) {
	ptr := func(b bool) *bool { return &b }
	tests := []struct {
		pspAPE    bool  // PSP AllowPrivilegeEscalation
		pspDAPE   *bool // PSP DefaultAllowPrivilegeEscalation
		podAPE    *bool // Pod AllowPrivilegeEscalation
		expectErr bool
		expectAPE *bool // Expected value of pod APE (if no error)
	}{
		// Test all valid combinations of PSP AllowPrivilegeEscalation,
		// DefaultAllowPrivilegeEscalation, and Pod AllowPrivilegeEscalation.
		{true, nil, nil, false, nil},
		{true, nil, ptr(false), false, ptr(false)},
		{true, nil, ptr(true), false, ptr(true)},
		{true, ptr(false), nil, false, ptr(false)},
		{true, ptr(false), ptr(false), false, ptr(false)},
		{true, ptr(false), ptr(true), false, ptr(true)},
		{true, ptr(true), nil, false, ptr(true)},
		{true, ptr(true), ptr(false), false, ptr(false)},
		{true, ptr(true), ptr(true), false, ptr(true)},
		{false, nil, nil, false, ptr(false)},
		{false, nil, ptr(false), false, ptr(false)},
		{false, nil, ptr(true), true, nil},
		{false, ptr(false), nil, false, ptr(false)},
		{false, ptr(false), ptr(false), false, ptr(false)},
		{false, ptr(false), ptr(true), true, nil},
		// Invalid cases: pspAPE=false, pspDAPE=true
	}

	fmtPtr := func(b *bool) string {
		if b == nil {
			return "nil"
		}
		return strconv.FormatBool(*b)
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("pspAPE:%t_pspDAPE:%s_podAPE:%s", test.pspAPE, fmtPtr(test.pspDAPE), fmtPtr(test.podAPE)), func(t *testing.T) {
			pod := defaultPod()
			pod.Spec.Containers[0].SecurityContext.AllowPrivilegeEscalation = test.podAPE

			psp := defaultPSP()
			psp.Spec.AllowPrivilegeEscalation = &test.pspAPE
			psp.Spec.DefaultAllowPrivilegeEscalation = test.pspDAPE

			provider, err := NewSimpleProvider(psp, "namespace", NewSimpleStrategyFactory())
			require.NoError(t, err)

			err = provider.DefaultContainerSecurityContext(pod, &pod.Spec.Containers[0])
			require.NoError(t, err)

			errs := provider.ValidateContainer(pod, &pod.Spec.Containers[0], field.NewPath(""))
			if test.expectErr {
				assert.NotEmpty(t, errs, "expected validation error")
			} else {
				assert.Empty(t, errs, "expected no validation errors")
				ape := pod.Spec.Containers[0].SecurityContext.AllowPrivilegeEscalation
				assert.Equal(t, test.expectAPE, ape, "expected pod AllowPrivilegeEscalation")
			}
		})
	}
}
