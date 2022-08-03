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

package validation

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/utils/pointer"
)

func TestValidatePodDisruptionBudgetSpec(t *testing.T) {
	minAvailable := intstr.FromString("0%")
	maxUnavailable := intstr.FromString("10%")

	spec := policy.PodDisruptionBudgetSpec{
		MinAvailable:   &minAvailable,
		MaxUnavailable: &maxUnavailable,
	}
	errs := ValidatePodDisruptionBudgetSpec(spec, field.NewPath("foo"))
	if len(errs) == 0 {
		t.Errorf("unexpected success for %v", spec)
	}
}

func TestValidateMinAvailablePodDisruptionBudgetSpec(t *testing.T) {
	successCases := []intstr.IntOrString{
		intstr.FromString("0%"),
		intstr.FromString("1%"),
		intstr.FromString("100%"),
		intstr.FromInt(0),
		intstr.FromInt(1),
		intstr.FromInt(100),
	}
	for _, c := range successCases {
		spec := policy.PodDisruptionBudgetSpec{
			MinAvailable: &c,
		}
		errs := ValidatePodDisruptionBudgetSpec(spec, field.NewPath("foo"))
		if len(errs) != 0 {
			t.Errorf("unexpected failure %v for %v", errs, spec)
		}
	}

	failureCases := []intstr.IntOrString{
		intstr.FromString("1.1%"),
		intstr.FromString("nope"),
		intstr.FromString("-1%"),
		intstr.FromString("101%"),
		intstr.FromInt(-1),
	}
	for _, c := range failureCases {
		spec := policy.PodDisruptionBudgetSpec{
			MinAvailable: &c,
		}
		errs := ValidatePodDisruptionBudgetSpec(spec, field.NewPath("foo"))
		if len(errs) == 0 {
			t.Errorf("unexpected success for %v", spec)
		}
	}
}

func TestValidateMinAvailablePodAndMaxUnavailableDisruptionBudgetSpec(t *testing.T) {
	c1 := intstr.FromString("10%")
	c2 := intstr.FromInt(1)

	spec := policy.PodDisruptionBudgetSpec{
		MinAvailable:   &c1,
		MaxUnavailable: &c2,
	}
	errs := ValidatePodDisruptionBudgetSpec(spec, field.NewPath("foo"))
	if len(errs) == 0 {
		t.Errorf("unexpected success for %v", spec)
	}
}

func TestValidatePodDisruptionBudgetStatus(t *testing.T) {
	const expectNoErrors = false
	const expectErrors = true
	testCases := []struct {
		name                string
		pdbStatus           policy.PodDisruptionBudgetStatus
		expectErrForVersion map[schema.GroupVersion]bool
	}{
		{
			name: "DisruptionsAllowed: 10",
			pdbStatus: policy.PodDisruptionBudgetStatus{
				DisruptionsAllowed: 10,
			},
			expectErrForVersion: map[schema.GroupVersion]bool{
				policy.SchemeGroupVersion:        expectNoErrors,
				policyv1beta1.SchemeGroupVersion: expectNoErrors,
			},
		},
		{
			name: "CurrentHealthy: 5",
			pdbStatus: policy.PodDisruptionBudgetStatus{
				CurrentHealthy: 5,
			},
			expectErrForVersion: map[schema.GroupVersion]bool{
				policy.SchemeGroupVersion:        expectNoErrors,
				policyv1beta1.SchemeGroupVersion: expectNoErrors,
			},
		},
		{
			name: "DesiredHealthy: 3",
			pdbStatus: policy.PodDisruptionBudgetStatus{
				DesiredHealthy: 3,
			},
			expectErrForVersion: map[schema.GroupVersion]bool{
				policy.SchemeGroupVersion:        expectNoErrors,
				policyv1beta1.SchemeGroupVersion: expectNoErrors,
			},
		},
		{
			name: "ExpectedPods: 2",
			pdbStatus: policy.PodDisruptionBudgetStatus{
				ExpectedPods: 2,
			},
			expectErrForVersion: map[schema.GroupVersion]bool{
				policy.SchemeGroupVersion:        expectNoErrors,
				policyv1beta1.SchemeGroupVersion: expectNoErrors,
			},
		},
		{
			name: "DisruptionsAllowed: -10",
			pdbStatus: policy.PodDisruptionBudgetStatus{
				DisruptionsAllowed: -10,
			},
			expectErrForVersion: map[schema.GroupVersion]bool{
				policy.SchemeGroupVersion:        expectErrors,
				policyv1beta1.SchemeGroupVersion: expectNoErrors,
			},
		},
		{
			name: "CurrentHealthy: -5",
			pdbStatus: policy.PodDisruptionBudgetStatus{
				CurrentHealthy: -5,
			},
			expectErrForVersion: map[schema.GroupVersion]bool{
				policy.SchemeGroupVersion:        expectErrors,
				policyv1beta1.SchemeGroupVersion: expectNoErrors,
			},
		},
		{
			name: "DesiredHealthy: -3",
			pdbStatus: policy.PodDisruptionBudgetStatus{
				DesiredHealthy: -3,
			},
			expectErrForVersion: map[schema.GroupVersion]bool{
				policy.SchemeGroupVersion:        expectErrors,
				policyv1beta1.SchemeGroupVersion: expectNoErrors,
			},
		},
		{
			name: "ExpectedPods: -2",
			pdbStatus: policy.PodDisruptionBudgetStatus{
				ExpectedPods: -2,
			},
			expectErrForVersion: map[schema.GroupVersion]bool{
				policy.SchemeGroupVersion:        expectErrors,
				policyv1beta1.SchemeGroupVersion: expectNoErrors,
			},
		},
		{
			name: "Conditions valid",
			pdbStatus: policy.PodDisruptionBudgetStatus{
				Conditions: []metav1.Condition{
					{
						Type:   policyv1beta1.DisruptionAllowedCondition,
						Status: metav1.ConditionTrue,
						LastTransitionTime: metav1.Time{
							Time: time.Now().Add(-5 * time.Minute),
						},
						Reason:             policyv1beta1.SufficientPodsReason,
						Message:            "message",
						ObservedGeneration: 3,
					},
				},
			},
			expectErrForVersion: map[schema.GroupVersion]bool{
				policy.SchemeGroupVersion:        expectNoErrors,
				policyv1beta1.SchemeGroupVersion: expectNoErrors,
			},
		},
		{
			name: "Conditions not valid",
			pdbStatus: policy.PodDisruptionBudgetStatus{
				Conditions: []metav1.Condition{
					{
						Type:   policyv1beta1.DisruptionAllowedCondition,
						Status: metav1.ConditionTrue,
					},
					{
						Type:   policyv1beta1.DisruptionAllowedCondition,
						Status: metav1.ConditionFalse,
					},
				},
			},
			expectErrForVersion: map[schema.GroupVersion]bool{
				policy.SchemeGroupVersion:        expectErrors,
				policyv1beta1.SchemeGroupVersion: expectErrors,
			},
		},
	}

	for _, tc := range testCases {
		for apiVersion, expectErrors := range tc.expectErrForVersion {
			t.Run(fmt.Sprintf("apiVersion: %s, %s", apiVersion.String(), tc.name), func(t *testing.T) {
				errors := ValidatePodDisruptionBudgetStatusUpdate(tc.pdbStatus, policy.PodDisruptionBudgetStatus{},
					field.NewPath("status"), apiVersion)
				errCount := len(errors)

				if errCount > 0 && !expectErrors {
					t.Errorf("unexpected failure %v for %v", errors, tc.pdbStatus)
				}

				if errCount == 0 && expectErrors {
					t.Errorf("expected errors but didn't one for %v", tc.pdbStatus)
				}
			})
		}
	}
}

func TestValidatePodSecurityPolicy(t *testing.T) {
	validPSP := func() *policy.PodSecurityPolicy {
		return &policy.PodSecurityPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "foo",
				Annotations: map[string]string{},
			},
			Spec: policy.PodSecurityPolicySpec{
				SELinux: policy.SELinuxStrategyOptions{
					Rule: policy.SELinuxStrategyRunAsAny,
				},
				RunAsUser: policy.RunAsUserStrategyOptions{
					Rule: policy.RunAsUserStrategyRunAsAny,
				},
				RunAsGroup: &policy.RunAsGroupStrategyOptions{
					Rule: policy.RunAsGroupStrategyRunAsAny,
				},
				FSGroup: policy.FSGroupStrategyOptions{
					Rule: policy.FSGroupStrategyRunAsAny,
				},
				SupplementalGroups: policy.SupplementalGroupsStrategyOptions{
					Rule: policy.SupplementalGroupsStrategyRunAsAny,
				},
				AllowedHostPaths: []policy.AllowedHostPath{
					{PathPrefix: "/foo/bar"},
					{PathPrefix: "/baz/"},
				},
			},
		}
	}

	noUserOptions := validPSP()
	noUserOptions.Spec.RunAsUser.Rule = ""

	noGroupOptions := validPSP()
	noGroupOptions.Spec.RunAsGroup.Rule = ""

	noSELinuxOptions := validPSP()
	noSELinuxOptions.Spec.SELinux.Rule = ""

	invalidUserStratType := validPSP()
	invalidUserStratType.Spec.RunAsUser.Rule = "invalid"

	invalidGroupStratType := validPSP()
	invalidGroupStratType.Spec.RunAsGroup.Rule = "invalid"

	invalidSELinuxStratType := validPSP()
	invalidSELinuxStratType.Spec.SELinux.Rule = "invalid"

	invalidUIDPSP := validPSP()
	invalidUIDPSP.Spec.RunAsUser.Rule = policy.RunAsUserStrategyMustRunAs
	invalidUIDPSP.Spec.RunAsUser.Ranges = []policy.IDRange{{Min: -1, Max: 1}}

	invalidGIDPSP := validPSP()
	invalidGIDPSP.Spec.RunAsGroup.Rule = policy.RunAsGroupStrategyMustRunAs
	invalidGIDPSP.Spec.RunAsGroup.Ranges = []policy.IDRange{{Min: -1, Max: 1}}

	missingObjectMetaName := validPSP()
	missingObjectMetaName.ObjectMeta.Name = ""

	noFSGroupOptions := validPSP()
	noFSGroupOptions.Spec.FSGroup.Rule = ""

	invalidFSGroupStratType := validPSP()
	invalidFSGroupStratType.Spec.FSGroup.Rule = "invalid"

	noSupplementalGroupsOptions := validPSP()
	noSupplementalGroupsOptions.Spec.SupplementalGroups.Rule = ""

	invalidSupGroupStratType := validPSP()
	invalidSupGroupStratType.Spec.SupplementalGroups.Rule = "invalid"

	invalidRangeMinGreaterThanMax := validPSP()
	invalidRangeMinGreaterThanMax.Spec.FSGroup.Ranges = []policy.IDRange{
		{Min: 2, Max: 1},
	}

	invalidRangeNegativeMin := validPSP()
	invalidRangeNegativeMin.Spec.FSGroup.Ranges = []policy.IDRange{
		{Min: -1, Max: 10},
	}

	invalidRangeNegativeMax := validPSP()
	invalidRangeNegativeMax.Spec.FSGroup.Ranges = []policy.IDRange{
		{Min: 1, Max: -10},
	}

	wildcardAllowedCapAndRequiredDrop := validPSP()
	wildcardAllowedCapAndRequiredDrop.Spec.RequiredDropCapabilities = []api.Capability{"foo"}
	wildcardAllowedCapAndRequiredDrop.Spec.AllowedCapabilities = []api.Capability{policy.AllowAllCapabilities}

	requiredCapAddAndDrop := validPSP()
	requiredCapAddAndDrop.Spec.DefaultAddCapabilities = []api.Capability{"foo"}
	requiredCapAddAndDrop.Spec.RequiredDropCapabilities = []api.Capability{"foo"}

	allowedCapListedInRequiredDrop := validPSP()
	allowedCapListedInRequiredDrop.Spec.RequiredDropCapabilities = []api.Capability{"foo"}
	allowedCapListedInRequiredDrop.Spec.AllowedCapabilities = []api.Capability{"foo"}

	invalidAppArmorDefault := validPSP()
	invalidAppArmorDefault.Annotations = map[string]string{
		v1.AppArmorBetaDefaultProfileAnnotationKey: "not-good",
	}
	invalidAppArmorAllowed := validPSP()
	invalidAppArmorAllowed.Annotations = map[string]string{
		v1.AppArmorBetaAllowedProfilesAnnotationKey: v1.AppArmorBetaProfileRuntimeDefault + ",not-good",
	}

	invalidAllowedUnsafeSysctlPattern := validPSP()
	invalidAllowedUnsafeSysctlPattern.Spec.AllowedUnsafeSysctls = []string{"a.*.b"}

	invalidForbiddenSysctlPattern := validPSP()
	invalidForbiddenSysctlPattern.Spec.ForbiddenSysctls = []string{"a.*.b"}

	invalidOverlappingSysctls := validPSP()
	invalidOverlappingSysctls.Spec.ForbiddenSysctls = []string{"kernel.*", "net.ipv4.ip_local_port_range"}
	invalidOverlappingSysctls.Spec.AllowedUnsafeSysctls = []string{"kernel.shmmax", "net.ipv4.ip_local_port_range"}

	invalidDuplicatedSysctls := validPSP()
	invalidDuplicatedSysctls.Spec.ForbiddenSysctls = []string{"net.ipv4.ip_local_port_range"}
	invalidDuplicatedSysctls.Spec.AllowedUnsafeSysctls = []string{"net.ipv4.ip_local_port_range"}

	invalidSeccompDefault := validPSP()
	invalidSeccompDefault.Annotations = map[string]string{
		seccompDefaultProfileAnnotationKey: "not-good",
	}
	invalidSeccompAllowAnyDefault := validPSP()
	invalidSeccompAllowAnyDefault.Annotations = map[string]string{
		seccompDefaultProfileAnnotationKey: "*",
	}
	invalidSeccompAllowed := validPSP()
	invalidSeccompAllowed.Annotations = map[string]string{
		seccompAllowedProfilesAnnotationKey: api.SeccompProfileRuntimeDefault + ",not-good",
	}

	invalidAllowedHostPathMissingPath := validPSP()
	invalidAllowedHostPathMissingPath.Spec.AllowedHostPaths = []policy.AllowedHostPath{
		{PathPrefix: ""},
	}

	invalidAllowedHostPathBacksteps := validPSP()
	invalidAllowedHostPathBacksteps.Spec.AllowedHostPaths = []policy.AllowedHostPath{
		{PathPrefix: "/dont/allow/backsteps/.."},
	}

	invalidDefaultAllowPrivilegeEscalation := validPSP()
	pe := true
	invalidDefaultAllowPrivilegeEscalation.Spec.DefaultAllowPrivilegeEscalation = &pe

	emptyFlexDriver := validPSP()
	emptyFlexDriver.Spec.Volumes = []policy.FSType{policy.FlexVolume}
	emptyFlexDriver.Spec.AllowedFlexVolumes = []policy.AllowedFlexVolume{{}}

	nonEmptyFlexVolumes := validPSP()
	nonEmptyFlexVolumes.Spec.AllowedFlexVolumes = []policy.AllowedFlexVolume{{Driver: "example/driver"}}

	invalidProcMount := validPSP()
	invalidProcMount.Spec.AllowedProcMountTypes = []api.ProcMountType{api.ProcMountType("bogus")}

	allowedCSIDriverPSP := validPSP()
	allowedCSIDriverPSP.Spec.Volumes = []policy.FSType{policy.CSI}
	allowedCSIDriverPSP.Spec.AllowedCSIDrivers = []policy.AllowedCSIDriver{{}}

	type testCase struct {
		psp         *policy.PodSecurityPolicy
		errorType   field.ErrorType
		errorDetail string
	}
	errorCases := map[string]testCase{
		"no user options": {
			psp:         noUserOptions,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: `supported values: "MustRunAs", "MustRunAsNonRoot", "RunAsAny"`,
		},
		"no group options": {
			psp:         noGroupOptions,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: `supported values: "MustRunAs", "RunAsAny", "MayRunAs"`,
		},
		"no selinux options": {
			psp:         noSELinuxOptions,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: `supported values: "MustRunAs", "RunAsAny"`,
		},
		"no fsgroup options": {
			psp:         noFSGroupOptions,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: `supported values: "MayRunAs", "MustRunAs", "RunAsAny"`,
		},
		"no sup group options": {
			psp:         noSupplementalGroupsOptions,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: `supported values: "MayRunAs", "MustRunAs", "RunAsAny"`,
		},
		"invalid user strategy type": {
			psp:         invalidUserStratType,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: `supported values: "MustRunAs", "MustRunAsNonRoot", "RunAsAny"`,
		},
		"invalid group strategy type": {
			psp:         invalidGroupStratType,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: `supported values: "MustRunAs", "RunAsAny", "MayRunAs"`,
		},
		"invalid selinux strategy type": {
			psp:         invalidSELinuxStratType,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: `supported values: "MustRunAs", "RunAsAny"`,
		},
		"invalid sup group strategy type": {
			psp:         invalidSupGroupStratType,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: `supported values: "MayRunAs", "MustRunAs", "RunAsAny"`,
		},
		"invalid fs group strategy type": {
			psp:         invalidFSGroupStratType,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: `supported values: "MayRunAs", "MustRunAs", "RunAsAny"`,
		},
		"invalid uid": {
			psp:         invalidUIDPSP,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "min cannot be negative",
		},
		"invalid gid": {
			psp:         invalidGIDPSP,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "min cannot be negative",
		},
		"missing object meta name": {
			psp:         missingObjectMetaName,
			errorType:   field.ErrorTypeRequired,
			errorDetail: "name or generateName is required",
		},
		"invalid range min greater than max": {
			psp:         invalidRangeMinGreaterThanMax,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "min cannot be greater than max",
		},
		"invalid range negative min": {
			psp:         invalidRangeNegativeMin,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "min cannot be negative",
		},
		"invalid range negative max": {
			psp:         invalidRangeNegativeMax,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "max cannot be negative",
		},
		"non-empty required drops and all caps are allowed by a wildcard": {
			psp:         wildcardAllowedCapAndRequiredDrop,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "must be empty when all capabilities are allowed by a wildcard",
		},
		"invalid required caps": {
			psp:         requiredCapAddAndDrop,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "capability is listed in defaultAddCapabilities and requiredDropCapabilities",
		},
		"allowed cap listed in required drops": {
			psp:         allowedCapListedInRequiredDrop,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "capability is listed in allowedCapabilities and requiredDropCapabilities",
		},
		"invalid AppArmor default profile": {
			psp:         invalidAppArmorDefault,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "invalid AppArmor profile name: \"not-good\"",
		},
		"invalid AppArmor allowed profile": {
			psp:         invalidAppArmorAllowed,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "invalid AppArmor profile name: \"not-good\"",
		},
		"invalid allowed unsafe sysctl pattern": {
			psp:         invalidAllowedUnsafeSysctlPattern,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: fmt.Sprintf("must have at most 253 characters and match regex %s", SysctlContainSlashPatternFmt),
		},
		"invalid forbidden sysctl pattern": {
			psp:         invalidForbiddenSysctlPattern,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: fmt.Sprintf("must have at most 253 characters and match regex %s", SysctlContainSlashPatternFmt),
		},
		"invalid overlapping sysctl pattern": {
			psp:         invalidOverlappingSysctls,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: fmt.Sprintf("sysctl overlaps with %s", invalidOverlappingSysctls.Spec.ForbiddenSysctls[0]),
		},
		"invalid duplicated sysctls": {
			psp:         invalidDuplicatedSysctls,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: fmt.Sprintf("sysctl overlaps with %s", invalidDuplicatedSysctls.Spec.AllowedUnsafeSysctls[0]),
		},
		"invalid seccomp default profile": {
			psp:         invalidSeccompDefault,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "must be a valid seccomp profile",
		},
		"invalid seccomp allow any default profile": {
			psp:         invalidSeccompAllowAnyDefault,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "must be a valid seccomp profile",
		},
		"invalid seccomp allowed profile": {
			psp:         invalidSeccompAllowed,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "must be a valid seccomp profile",
		},
		"invalid defaultAllowPrivilegeEscalation": {
			psp:         invalidDefaultAllowPrivilegeEscalation,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "Cannot set DefaultAllowPrivilegeEscalation to true without also setting AllowPrivilegeEscalation to true",
		},
		"invalid allowed host path empty path": {
			psp:         invalidAllowedHostPathMissingPath,
			errorType:   field.ErrorTypeRequired,
			errorDetail: "is required",
		},
		"invalid allowed host path with backsteps": {
			psp:         invalidAllowedHostPathBacksteps,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "must not contain '..'",
		},
		"empty flex volume driver": {
			psp:         emptyFlexDriver,
			errorType:   field.ErrorTypeRequired,
			errorDetail: "must specify a driver",
		},
		"CSI policy with empty allowed driver list": {
			psp:       allowedCSIDriverPSP,
			errorType: field.ErrorTypeRequired,
		},
		"invalid allowedProcMountTypes": {
			psp:         invalidProcMount,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: `supported values: "Default", "Unmasked"`,
		},
	}

	for k, v := range errorCases {
		errs := ValidatePodSecurityPolicy(v.psp)
		if len(errs) == 0 {
			t.Errorf("%s expected errors but got none", k)
			continue
		}
		if errs[0].Type != v.errorType {
			t.Errorf("[%s] received an unexpected error type.  Expected: '%s' got: '%s'", k, v.errorType, errs[0].Type)
		}
		if errs[0].Detail != v.errorDetail {
			t.Errorf("[%s] received an unexpected error detail.  Expected '%s' got: '%s'", k, v.errorDetail, errs[0].Detail)
		}
	}

	// Update error is different for 'missing object meta name'.
	errorCases["missing object meta name"] = testCase{
		psp:         errorCases["missing object meta name"].psp,
		errorType:   field.ErrorTypeInvalid,
		errorDetail: "field is immutable",
	}

	// Should not be able to update to an invalid policy.
	for k, v := range errorCases {
		v.psp.ResourceVersion = "444" // Required for updates.
		errs := ValidatePodSecurityPolicyUpdate(validPSP(), v.psp)
		if len(errs) == 0 {
			t.Errorf("[%s] expected update errors but got none", k)
			continue
		}
		if errs[0].Type != v.errorType {
			t.Errorf("[%s] received an unexpected error type.  Expected: '%s' got: '%s'", k, v.errorType, errs[0].Type)
		}
		if errs[0].Detail != v.errorDetail {
			t.Errorf("[%s] received an unexpected error detail.  Expected '%s' got: '%s'", k, v.errorDetail, errs[0].Detail)
		}
	}

	mustRunAs := validPSP()
	mustRunAs.Spec.FSGroup.Rule = policy.FSGroupStrategyMustRunAs
	mustRunAs.Spec.SupplementalGroups.Rule = policy.SupplementalGroupsStrategyMustRunAs
	mustRunAs.Spec.RunAsUser.Rule = policy.RunAsUserStrategyMustRunAs
	mustRunAs.Spec.RunAsUser.Ranges = []policy.IDRange{
		{Min: 1, Max: 1},
	}
	mustRunAs.Spec.SELinux.Rule = policy.SELinuxStrategyMustRunAs

	runAsNonRoot := validPSP()
	runAsNonRoot.Spec.RunAsUser.Rule = policy.RunAsUserStrategyMustRunAsNonRoot

	caseInsensitiveAddDrop := validPSP()
	caseInsensitiveAddDrop.Spec.DefaultAddCapabilities = []api.Capability{"foo"}
	caseInsensitiveAddDrop.Spec.RequiredDropCapabilities = []api.Capability{"FOO"}

	caseInsensitiveAllowedDrop := validPSP()
	caseInsensitiveAllowedDrop.Spec.RequiredDropCapabilities = []api.Capability{"FOO"}
	caseInsensitiveAllowedDrop.Spec.AllowedCapabilities = []api.Capability{"foo"}

	validAppArmor := validPSP()
	validAppArmor.Annotations = map[string]string{
		v1.AppArmorBetaDefaultProfileAnnotationKey:  v1.AppArmorBetaProfileRuntimeDefault,
		v1.AppArmorBetaAllowedProfilesAnnotationKey: v1.AppArmorBetaProfileRuntimeDefault + "," + v1.AppArmorBetaProfileNamePrefix + "foo",
	}

	withForbiddenSysctl := validPSP()
	withForbiddenSysctl.Spec.ForbiddenSysctls = []string{"net.*"}

	withAllowedUnsafeSysctl := validPSP()
	withAllowedUnsafeSysctl.Spec.AllowedUnsafeSysctls = []string{"net.ipv4.tcp_max_syn_backlog"}

	validSeccomp := validPSP()
	validSeccomp.Annotations = map[string]string{
		seccompDefaultProfileAnnotationKey:  api.SeccompProfileRuntimeDefault,
		seccompAllowedProfilesAnnotationKey: api.SeccompProfileRuntimeDefault + ",unconfined,localhost/foo,*",
	}

	validDefaultAllowPrivilegeEscalation := validPSP()
	pe = true
	validDefaultAllowPrivilegeEscalation.Spec.DefaultAllowPrivilegeEscalation = &pe
	validDefaultAllowPrivilegeEscalation.Spec.AllowPrivilegeEscalation = true

	flexvolumeWhenFlexVolumesAllowed := validPSP()
	flexvolumeWhenFlexVolumesAllowed.Spec.Volumes = []policy.FSType{policy.FlexVolume}
	flexvolumeWhenFlexVolumesAllowed.Spec.AllowedFlexVolumes = []policy.AllowedFlexVolume{
		{Driver: "example/driver1"},
	}

	flexvolumeWhenAllVolumesAllowed := validPSP()
	flexvolumeWhenAllVolumesAllowed.Spec.Volumes = []policy.FSType{policy.All}
	flexvolumeWhenAllVolumesAllowed.Spec.AllowedFlexVolumes = []policy.AllowedFlexVolume{
		{Driver: "example/driver2"},
	}

	validProcMount := validPSP()
	validProcMount.Spec.AllowedProcMountTypes = []api.ProcMountType{api.DefaultProcMount, api.UnmaskedProcMount}

	allowedCSIDriversWithCSIFsType := validPSP()
	allowedCSIDriversWithCSIFsType.Spec.Volumes = []policy.FSType{policy.CSI}
	allowedCSIDriversWithCSIFsType.Spec.AllowedCSIDrivers = []policy.AllowedCSIDriver{{Name: "foo"}}

	allowedCSIDriversWithAllFsTypes := validPSP()
	allowedCSIDriversWithAllFsTypes.Spec.Volumes = []policy.FSType{policy.All}
	allowedCSIDriversWithAllFsTypes.Spec.AllowedCSIDrivers = []policy.AllowedCSIDriver{{Name: "bar"}}

	successCases := map[string]struct {
		psp *policy.PodSecurityPolicy
	}{
		"must run as": {
			psp: mustRunAs,
		},
		"run as any": {
			psp: validPSP(),
		},
		"run as non-root (user only)": {
			psp: runAsNonRoot,
		},
		"comparison for add -> drop is case sensitive": {
			psp: caseInsensitiveAddDrop,
		},
		"comparison for allowed -> drop is case sensitive": {
			psp: caseInsensitiveAllowedDrop,
		},
		"valid AppArmor annotations": {
			psp: validAppArmor,
		},
		"with network sysctls forbidden": {
			psp: withForbiddenSysctl,
		},
		"with unsafe net.ipv4.tcp_max_syn_backlog sysctl allowed": {
			psp: withAllowedUnsafeSysctl,
		},
		"valid seccomp annotations": {
			psp: validSeccomp,
		},
		"valid defaultAllowPrivilegeEscalation as true": {
			psp: validDefaultAllowPrivilegeEscalation,
		},
		"allow white-listed flexVolume when flex volumes are allowed": {
			psp: flexvolumeWhenFlexVolumesAllowed,
		},
		"allow white-listed flexVolume when all volumes are allowed": {
			psp: flexvolumeWhenAllVolumesAllowed,
		},
		"valid allowedProcMountTypes": {
			psp: validProcMount,
		},
		"allowed CSI drivers when FSType policy is set to CSI": {
			psp: allowedCSIDriversWithCSIFsType,
		},
		"allowed CSI drivers when FSType policy is set to All": {
			psp: allowedCSIDriversWithAllFsTypes,
		},
	}

	for k, v := range successCases {
		if errs := ValidatePodSecurityPolicy(v.psp); len(errs) != 0 {
			t.Errorf("Expected success for %s, got %v", k, errs)
		}

		// Should be able to update to a valid PSP.
		v.psp.ResourceVersion = "444" // Required for updates.
		if errs := ValidatePodSecurityPolicyUpdate(validPSP(), v.psp); len(errs) != 0 {
			t.Errorf("Expected success for %s update, got %v", k, errs)
		}
	}
}

func TestValidatePSPVolumes(t *testing.T) {
	validPSP := func() *policy.PodSecurityPolicy {
		return &policy.PodSecurityPolicy{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: policy.PodSecurityPolicySpec{
				SELinux: policy.SELinuxStrategyOptions{
					Rule: policy.SELinuxStrategyRunAsAny,
				},
				RunAsUser: policy.RunAsUserStrategyOptions{
					Rule: policy.RunAsUserStrategyRunAsAny,
				},
				RunAsGroup: &policy.RunAsGroupStrategyOptions{
					Rule: policy.RunAsGroupStrategyRunAsAny,
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

	volumes := getAllFSTypesAsSet()
	// add in the * value since that is a pseudo type that is not included by default
	volumes.Insert(string(policy.All))

	for _, strVolume := range volumes.List() {
		psp := validPSP()
		psp.Spec.Volumes = []policy.FSType{policy.FSType(strVolume)}
		errs := ValidatePodSecurityPolicy(psp)
		if len(errs) != 0 {
			t.Errorf("%s validation expected no errors but received %v", strVolume, errs)
		}
	}
}

func TestIsValidSysctlPattern(t *testing.T) {
	valid := []string{
		"a.b.c.d",
		"a",
		"a_b",
		"a-b",
		"abc",
		"abc.def",
		"*",
		"a.*",
		"*",
		"abc*",
		"a.abc*",
		"a.b.*",
		"a/b/c/d",
		"a/*",
		"a/b/*",
		"a.b/c*",
		"a.b/c.d",
		"a/b.c/d",
	}
	invalid := []string{
		"",
		"ä",
		"a_",
		"_",
		"_a",
		"_a._b",
		"__",
		"-",
		".",
		"a.",
		".a",
		"a.b.",
		"a*.b",
		"a*b",
		"*a",
		"Abc",
		"/",
		"a/",
		"/a",
		"a*/b",
		func(n int) string {
			x := make([]byte, n)
			for i := range x {
				x[i] = byte('a')
			}
			return string(x)
		}(256),
	}
	for _, s := range valid {
		if !IsValidSysctlPattern(s) {
			t.Errorf("%q expected to be a valid sysctl pattern", s)
		}
	}
	for _, s := range invalid {
		if IsValidSysctlPattern(s) {
			t.Errorf("%q expected to be an invalid sysctl pattern", s)
		}
	}
}

func TestValidatePSPRunAsUser(t *testing.T) {
	var testCases = []struct {
		name              string
		runAsUserStrategy policy.RunAsUserStrategyOptions
		fail              bool
	}{
		{"Invalid RunAsUserStrategy", policy.RunAsUserStrategyOptions{Rule: policy.RunAsUserStrategy("someInvalidStrategy")}, true},
		{"RunAsUserStrategyMustRunAs", policy.RunAsUserStrategyOptions{Rule: policy.RunAsUserStrategyMustRunAs}, false},
		{"RunAsUserStrategyMustRunAsNonRoot", policy.RunAsUserStrategyOptions{Rule: policy.RunAsUserStrategyMustRunAsNonRoot}, false},
		{"RunAsUserStrategyMustRunAsNonRoot With Valid Range", policy.RunAsUserStrategyOptions{Rule: policy.RunAsUserStrategyMustRunAs, Ranges: []policy.IDRange{{Min: 2, Max: 3}, {Min: 4, Max: 5}}}, false},
		{"RunAsUserStrategyMustRunAsNonRoot With Invalid Range", policy.RunAsUserStrategyOptions{Rule: policy.RunAsUserStrategyMustRunAs, Ranges: []policy.IDRange{{Min: 2, Max: 3}, {Min: 5, Max: 4}}}, true},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			errList := validatePSPRunAsUser(field.NewPath("status"), &testCase.runAsUserStrategy)
			actualErrors := len(errList)
			expectedErrors := 1
			if !testCase.fail {
				expectedErrors = 0
			}
			if actualErrors != expectedErrors {
				t.Errorf("In testCase %v, expected %v errors, got %v errors", testCase.name, expectedErrors, actualErrors)
			}
		})
	}
}

func TestValidatePSPFSGroup(t *testing.T) {
	var testCases = []struct {
		name            string
		fsGroupStrategy policy.FSGroupStrategyOptions
		fail            bool
	}{
		{"Invalid FSGroupStrategy", policy.FSGroupStrategyOptions{Rule: policy.FSGroupStrategyType("someInvalidStrategy")}, true},
		{"FSGroupStrategyMustRunAs", policy.FSGroupStrategyOptions{Rule: policy.FSGroupStrategyMustRunAs}, false},
		{"FSGroupStrategyMayRunAs", policy.FSGroupStrategyOptions{Rule: policy.FSGroupStrategyMayRunAs, Ranges: []policy.IDRange{{Min: 1, Max: 5}}}, false},
		{"FSGroupStrategyRunAsAny", policy.FSGroupStrategyOptions{Rule: policy.FSGroupStrategyRunAsAny}, false},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			errList := validatePSPFSGroup(field.NewPath("Status"), &testCase.fsGroupStrategy)
			actualErrors := len(errList)
			expectedErrors := 1
			if !testCase.fail {
				expectedErrors = 0
			}
			if actualErrors != expectedErrors {
				t.Errorf("In testCase %v, expected %v errors, got %v errors", testCase.name, expectedErrors, actualErrors)
			}
		})
	}
}

func TestValidatePSPSupplementalGroup(t *testing.T) {
	var testCases = []struct {
		name                      string
		supplementalGroupStrategy policy.SupplementalGroupsStrategyOptions
		fail                      bool
	}{
		{"Invalid SupplementalGroupStrategy", policy.SupplementalGroupsStrategyOptions{Rule: policy.SupplementalGroupsStrategyType("someInvalidStrategy")}, true},
		{"SupplementalGroupsStrategyMustRunAs", policy.SupplementalGroupsStrategyOptions{Rule: policy.SupplementalGroupsStrategyMustRunAs}, false},
		{"SupplementalGroupsStrategyMayRunAs", policy.SupplementalGroupsStrategyOptions{Rule: policy.SupplementalGroupsStrategyMayRunAs, Ranges: []policy.IDRange{{Min: 1, Max: 5}}}, false},
		{"SupplementalGroupsStrategyRunAsAny", policy.SupplementalGroupsStrategyOptions{Rule: policy.SupplementalGroupsStrategyRunAsAny}, false},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			errList := validatePSPSupplementalGroup(field.NewPath("Status"), &testCase.supplementalGroupStrategy)
			actualErrors := len(errList)
			expectedErrors := 1
			if !testCase.fail {
				expectedErrors = 0
			}
			if actualErrors != expectedErrors {
				t.Errorf("In testCase %v, expected %v errors, got %v errors", testCase.name, expectedErrors, actualErrors)
			}
		})
	}
}

func TestValidatePSPRunAsGroup(t *testing.T) {
	var testCases = []struct {
		name       string
		runAsGroup policy.RunAsGroupStrategyOptions
		fail       bool
	}{
		{"RunAsGroupStrategyMayRunAs", policy.RunAsGroupStrategyOptions{Rule: policy.RunAsGroupStrategyMayRunAs, Ranges: []policy.IDRange{{Min: 1, Max: 5}}}, false},
		{"RunAsGroupStrategyMustRunAs", policy.RunAsGroupStrategyOptions{Rule: policy.RunAsGroupStrategyMustRunAs, Ranges: []policy.IDRange{{Min: 1, Max: 5}}}, false},
		{"RunAsGroupStrategyRunAsAny", policy.RunAsGroupStrategyOptions{Rule: policy.RunAsGroupStrategyRunAsAny}, false},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			errList := validatePSPRunAsGroup(field.NewPath("Status"), &testCase.runAsGroup)
			actualErrors := len(errList)
			expectedErrors := 1
			if !testCase.fail {
				expectedErrors = 0
			}
			if actualErrors != expectedErrors {
				t.Errorf("In testCase %v, expected %v errors, got %v errors", testCase.name, expectedErrors, actualErrors)
			}
		})
	}
}

func TestValidatePSPSELinux(t *testing.T) {
	var testCases = []struct {
		name    string
		selinux policy.SELinuxStrategyOptions
		fail    bool
	}{
		{"SELinuxStrategyMustRunAs",
			policy.SELinuxStrategyOptions{
				Rule:           policy.SELinuxStrategyMustRunAs,
				SELinuxOptions: &api.SELinuxOptions{Level: "s9:z0,z1"}}, false},
		{"SELinuxStrategyMustRunAs",
			policy.SELinuxStrategyOptions{
				Rule:           policy.SELinuxStrategyMustRunAs,
				SELinuxOptions: &api.SELinuxOptions{Level: "s0"}}, false},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			errList := validatePSPSELinux(field.NewPath("Status"), &testCase.selinux)
			actualErrors := len(errList)
			expectedErrors := 1
			if !testCase.fail {
				expectedErrors = 0
			}
			if actualErrors != expectedErrors {
				t.Errorf("In testCase %v, expected %v errors, got %v errors", testCase.name, expectedErrors, actualErrors)
			}
		})
	}
}

func TestValidateRuntimeClassStrategy(t *testing.T) {
	var testCases = []struct {
		name         string
		strategy     *policy.RuntimeClassStrategyOptions
		expectErrors bool
	}{{
		name:     "nil strategy",
		strategy: nil,
	}, {
		name:     "empty strategy",
		strategy: &policy.RuntimeClassStrategyOptions{},
	}, {
		name: "allow all strategy",
		strategy: &policy.RuntimeClassStrategyOptions{
			AllowedRuntimeClassNames: []string{"*"},
		},
	}, {
		name: "valid defaulting & allow all",
		strategy: &policy.RuntimeClassStrategyOptions{
			DefaultRuntimeClassName:  pointer.StringPtr("native"),
			AllowedRuntimeClassNames: []string{"*"},
		},
	}, {
		name: "valid defaulting & allow explicit",
		strategy: &policy.RuntimeClassStrategyOptions{
			DefaultRuntimeClassName:  pointer.StringPtr("native"),
			AllowedRuntimeClassNames: []string{"foo", "native", "sandboxed"},
		},
	}, {
		name: "valid whitelisting",
		strategy: &policy.RuntimeClassStrategyOptions{
			AllowedRuntimeClassNames: []string{"foo", "native", "sandboxed"},
		},
	}, {
		name: "invalid default name",
		strategy: &policy.RuntimeClassStrategyOptions{
			DefaultRuntimeClassName: pointer.StringPtr("foo bar"),
		},
		expectErrors: true,
	}, {
		name: "disallowed default",
		strategy: &policy.RuntimeClassStrategyOptions{
			DefaultRuntimeClassName:  pointer.StringPtr("foo"),
			AllowedRuntimeClassNames: []string{"native", "sandboxed"},
		},
		expectErrors: true,
	}, {
		name: "nothing allowed default",
		strategy: &policy.RuntimeClassStrategyOptions{
			DefaultRuntimeClassName: pointer.StringPtr("foo"),
		},
		expectErrors: true,
	}, {
		name: "invalid whitelist name",
		strategy: &policy.RuntimeClassStrategyOptions{
			AllowedRuntimeClassNames: []string{"native", "sandboxed", "foo*"},
		},
		expectErrors: true,
	}, {
		name: "duplicate whitelist names",
		strategy: &policy.RuntimeClassStrategyOptions{
			AllowedRuntimeClassNames: []string{"native", "sandboxed", "native"},
		},
		expectErrors: true,
	}, {
		name: "allow all redundant whitelist",
		strategy: &policy.RuntimeClassStrategyOptions{
			AllowedRuntimeClassNames: []string{"*", "sandboxed", "native"},
		},
		expectErrors: true,
	}}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			errs := validateRuntimeClassStrategy(field.NewPath(""), test.strategy)
			if test.expectErrors {
				assert.NotEmpty(t, errs)
			} else {
				assert.Empty(t, errs)
			}
		})
	}
}

func TestAllowEphemeralVolumeType(t *testing.T) {
	pspWithoutGenericVolume := func() *policy.PodSecurityPolicy {
		return &policy.PodSecurityPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "psp",
				ResourceVersion: "1",
			},
			Spec: policy.PodSecurityPolicySpec{
				RunAsUser: policy.RunAsUserStrategyOptions{
					Rule: policy.RunAsUserStrategyMustRunAs,
				},
				SupplementalGroups: policy.SupplementalGroupsStrategyOptions{
					Rule: policy.SupplementalGroupsStrategyMustRunAs,
				},
				SELinux: policy.SELinuxStrategyOptions{
					Rule: policy.SELinuxStrategyMustRunAs,
				},
				FSGroup: policy.FSGroupStrategyOptions{
					Rule: policy.FSGroupStrategyMustRunAs,
				},
			},
		}
	}
	pspWithGenericVolume := func() *policy.PodSecurityPolicy {
		psp := pspWithoutGenericVolume()
		psp.Spec.Volumes = append(psp.Spec.Volumes, policy.Ephemeral)
		return psp
	}
	pspNil := func() *policy.PodSecurityPolicy {
		return nil
	}

	pspInfo := []struct {
		description      string
		hasGenericVolume bool
		psp              func() *policy.PodSecurityPolicy
	}{
		{
			description:      "PodSecurityPolicySpec Without GenericVolume",
			hasGenericVolume: false,
			psp:              pspWithoutGenericVolume,
		},
		{
			description:      "PodSecurityPolicySpec With GenericVolume",
			hasGenericVolume: true,
			psp:              pspWithGenericVolume,
		},
		{
			description:      "is nil",
			hasGenericVolume: false,
			psp:              pspNil,
		},
	}

	for _, oldPSPInfo := range pspInfo {
		for _, newPSPInfo := range pspInfo {
			oldPSP := oldPSPInfo.psp()
			newPSP := newPSPInfo.psp()
			if newPSP == nil {
				continue
			}

			t.Run(fmt.Sprintf("old PodSecurityPolicySpec %v, new PodSecurityPolicySpec %v", oldPSPInfo.description, newPSPInfo.description), func(t *testing.T) {
				var errs field.ErrorList
				if oldPSP == nil {
					errs = ValidatePodSecurityPolicy(newPSP)
				} else {
					errs = ValidatePodSecurityPolicyUpdate(oldPSP, newPSP)
				}
				if len(errs) > 0 {
					t.Errorf("expected no errors, got: %v", errs)
				}
			})
		}
	}
}
