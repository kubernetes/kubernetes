/*
Copyright The Kubernetes Authors.

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

package pod

import (
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

const validSecurityProfileOCIRef = "registry.example.com/seccomp/profile@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

type seccompProfileTestcase struct {
	id           string
	profile      *api.SeccompProfile
	expectedErrs field.ErrorList
	// podOnly cases exercise the custom validators, which coverage does not
	// track per location; the generated rules are checked at every location.
	podOnly bool
}

// seccompProfileTestcases returns the declarative seccomp profile cases for
// one seccompProfile location in a pod spec. The SecurityProfileOCI feature
// gate is disabled in these tests, so every case using the OCI type also
// expects the handwritten NotSupported error for the type.
func seccompProfileTestcases(fldPath *field.Path) []seccompProfileTestcase {
	typeNotSupported := field.NotSupported(fldPath.Child("type"), api.SeccompProfileTypeOCI, []api.SeccompProfileType(nil)).MarkFromImperative()
	ociPath := fldPath.Child("oci")
	basePath := ociPath.Child("baseProfile")
	ociProfile := func(base *api.SecurityProfileOCIBase) *api.SeccompProfile {
		return &api.SeccompProfile{
			Type: api.SeccompProfileTypeOCI,
			OCI:  &api.SecurityProfileOCI{Ref: validSecurityProfileOCIRef, BaseProfile: base},
		}
	}
	return []seccompProfileTestcase{{
		id:      "seccomp_type_required",
		profile: &api.SeccompProfile{},
		expectedErrs: field.ErrorList{
			field.Required(fldPath.Child("type"), "").MarkAlpha(),
		},
	}, {
		id:      "seccomp_oci_requires_type_oci",
		profile: &api.SeccompProfile{Type: api.SeccompProfileTypeRuntimeDefault, OCI: &api.SecurityProfileOCI{Ref: validSecurityProfileOCIRef}},
		expectedErrs: field.ErrorList{
			field.Invalid(ociPath, "", "").WithOrigin("union"),
		},
	}, {
		id:      "seccomp_type_oci_requires_oci",
		profile: &api.SeccompProfile{Type: api.SeccompProfileTypeOCI},
		expectedErrs: field.ErrorList{
			typeNotSupported,
			field.Invalid(ociPath, "", "").WithOrigin("union"),
		},
	}, {
		id:      "seccomp_oci_ref_required",
		profile: &api.SeccompProfile{Type: api.SeccompProfileTypeOCI, OCI: &api.SecurityProfileOCI{}},
		expectedErrs: field.ErrorList{
			typeNotSupported,
			field.Required(ociPath.Child("ref"), ""),
		},
	}, {
		id: "seccomp_oci_ref_too_long",
		profile: &api.SeccompProfile{Type: api.SeccompProfileTypeOCI, OCI: &api.SecurityProfileOCI{
			// The reference parser bounds the repository path, not the registry.
			Ref: strings.Repeat("a", 1000) + ".example.com/seccomp/profile@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		}},
		expectedErrs: field.ErrorList{
			typeNotSupported,
			field.TooLong(ociPath.Child("ref"), "", 1024).WithOrigin("maxLength"),
		},
	}, {
		id:      "seccomp_oci_ref_tag",
		podOnly: true,
		profile: &api.SeccompProfile{Type: api.SeccompProfileTypeOCI, OCI: &api.SecurityProfileOCI{Ref: "registry.example.com/seccomp/profile:v1"}},
		expectedErrs: field.ErrorList{
			typeNotSupported,
			field.Invalid(ociPath.Child("ref"), "", ""),
		},
	}, {
		id:      "seccomp_oci_base_type_required",
		profile: ociProfile(&api.SecurityProfileOCIBase{}),
		expectedErrs: field.ErrorList{
			typeNotSupported,
			field.Required(basePath.Child("type"), ""),
		},
	}, {
		id:      "seccomp_oci_base_type_not_supported",
		profile: ociProfile(&api.SecurityProfileOCIBase{Type: "Unconfined"}),
		expectedErrs: field.ErrorList{
			typeNotSupported,
			field.NotSupported(basePath.Child("type"), api.SecurityProfileOCIBaseType("Unconfined"), []api.SecurityProfileOCIBaseType(nil)),
		},
	}, {
		id:      "seccomp_oci_base_localhost_requires_localhost_profile",
		profile: ociProfile(&api.SecurityProfileOCIBase{Type: api.SecurityProfileOCIBaseTypeLocalhost}),
		expectedErrs: field.ErrorList{
			typeNotSupported,
			field.Invalid(basePath.Child("localhostProfile"), "", "").WithOrigin("union"),
		},
	}, {
		id:      "seccomp_oci_base_localhost_profile_requires_localhost",
		podOnly: true,
		profile: ociProfile(&api.SecurityProfileOCIBase{Type: api.SecurityProfileOCIBaseTypeRuntimeDefault, LocalhostProfile: new("profile.json")}),
		expectedErrs: field.ErrorList{
			typeNotSupported,
			field.Invalid(basePath.Child("localhostProfile"), "", "").WithOrigin("union"),
		},
	}, {
		id:      "seccomp_oci_base_localhost_profile_empty",
		podOnly: true,
		profile: ociProfile(&api.SecurityProfileOCIBase{Type: api.SecurityProfileOCIBaseTypeLocalhost, LocalhostProfile: new("")}),
		expectedErrs: field.ErrorList{
			typeNotSupported,
			field.Required(basePath.Child("localhostProfile"), ""),
		},
	}, {
		id:      "seccomp_oci_base_localhost_profile_absolute",
		podOnly: true,
		profile: ociProfile(&api.SecurityProfileOCIBase{Type: api.SecurityProfileOCIBaseTypeLocalhost, LocalhostProfile: new("/etc/profile.json")}),
		expectedErrs: field.ErrorList{
			typeNotSupported,
			field.Invalid(basePath.Child("localhostProfile"), "", ""),
		},
	}, {
		id:      "seccomp_oci_base_localhost_profile_too_long",
		profile: ociProfile(&api.SecurityProfileOCIBase{Type: api.SecurityProfileOCIBaseTypeLocalhost, LocalhostProfile: new(strings.Repeat("a", 4097))}),
		expectedErrs: field.ErrorList{
			typeNotSupported,
			field.TooLong(basePath.Child("localhostProfile"), "", 4096).WithOrigin("maxLength"),
		},
	}, {
		id:      "seccomp_oci_base_localhost_profile_backsteps",
		podOnly: true,
		profile: ociProfile(&api.SecurityProfileOCIBase{Type: api.SecurityProfileOCIBaseTypeLocalhost, LocalhostProfile: new("../profile.json")}),
		expectedErrs: field.ErrorList{
			typeNotSupported,
			field.Invalid(basePath.Child("localhostProfile"), "", ""),
		},
	}}
}

// securityProfileOCITestcases covers the declarative seccomp profile rules
// for the pod-level security context and for every container list.
func securityProfileOCITestcases(fldPath *field.Path) []PodSpecTestcase {
	containerWith := func(name string, p *api.SeccompProfile) api.Container {
		return podtest.MakeContainer(name, podtest.SetContainerSecurityContext(api.SecurityContext{SeccompProfile: p}))
	}
	locations := []struct {
		id    string
		path  *field.Path
		tweak func(p *api.SeccompProfile) podtest.Tweak
		// extraErrs are handwritten errors the location triggers on its own.
		extraErrs field.ErrorList
	}{{
		id:   "pod",
		path: fldPath.Child("securityContext", "seccompProfile"),
		tweak: func(p *api.SeccompProfile) podtest.Tweak {
			return podtest.SetSecurityContext(&api.PodSecurityContext{SeccompProfile: p})
		},
	}, {
		id:   "container",
		path: fldPath.Child("containers").Index(0).Child("securityContext", "seccompProfile"),
		tweak: func(p *api.SeccompProfile) podtest.Tweak {
			return podtest.SetContainers(containerWith("ctr", p))
		},
	}, {
		id:   "init_container",
		path: fldPath.Child("initContainers").Index(0).Child("securityContext", "seccompProfile"),
		tweak: func(p *api.SeccompProfile) podtest.Tweak {
			return podtest.SetInitContainers(containerWith("init", p))
		},
	}, {
		id:   "ephemeral_container",
		path: fldPath.Child("ephemeralContainers").Index(0).Child("securityContext", "seccompProfile"),
		tweak: func(p *api.SeccompProfile) podtest.Tweak {
			return podtest.SetEphemeralContainers(api.EphemeralContainer{EphemeralContainerCommon: api.EphemeralContainerCommon(containerWith("debug", p))})
		},
		// Ephemeral containers cannot be created with the pod or a template.
		extraErrs: field.ErrorList{field.Forbidden(fldPath.Child("ephemeralContainers"), "").MarkFromImperative()},
	}}

	var testcases []PodSpecTestcase
	for _, loc := range locations {
		for _, tc := range seccompProfileTestcases(loc.path) {
			if tc.podOnly && loc.id != "pod" {
				continue
			}
			testcases = append(testcases, PodSpecTestcase{
				ID:           loc.id + "_" + tc.id,
				Tweaks:       []podtest.Tweak{loc.tweak(tc.profile)},
				ExpectedErrs: append(append(field.ErrorList{}, loc.extraErrs...), tc.expectedErrs...),
			})
		}
	}
	return testcases
}
