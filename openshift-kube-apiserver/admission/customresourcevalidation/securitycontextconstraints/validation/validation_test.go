package validation

import (
	"fmt"
	"testing"

	kcorev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"

	securityv1 "github.com/openshift/api/security/v1"
)

func TestValidateSecurityContextConstraints(t *testing.T) {
	var invalidUID int64 = -1
	var invalidPriority int32 = -1
	var validPriority int32 = 1
	yes := true
	no := false

	validSCC := func() *securityv1.SecurityContextConstraints {
		return &securityv1.SecurityContextConstraints{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			SELinuxContext: securityv1.SELinuxContextStrategyOptions{
				Type: securityv1.SELinuxStrategyRunAsAny,
			},
			RunAsUser: securityv1.RunAsUserStrategyOptions{
				Type: securityv1.RunAsUserStrategyRunAsAny,
			},
			FSGroup: securityv1.FSGroupStrategyOptions{
				Type: securityv1.FSGroupStrategyRunAsAny,
			},
			SupplementalGroups: securityv1.SupplementalGroupsStrategyOptions{
				Type: securityv1.SupplementalGroupsStrategyRunAsAny,
			},
			Priority: &validPriority,
		}
	}

	noUserOptions := validSCC()
	noUserOptions.RunAsUser.Type = ""

	noSELinuxOptions := validSCC()
	noSELinuxOptions.SELinuxContext.Type = ""

	invalidUserStratType := validSCC()
	invalidUserStratType.RunAsUser.Type = "invalid"

	invalidSELinuxStratType := validSCC()
	invalidSELinuxStratType.SELinuxContext.Type = "invalid"

	invalidUIDSCC := validSCC()
	invalidUIDSCC.RunAsUser.Type = securityv1.RunAsUserStrategyMustRunAs
	invalidUIDSCC.RunAsUser.UID = &invalidUID

	missingObjectMetaName := validSCC()
	missingObjectMetaName.ObjectMeta.Name = ""

	noFSGroupOptions := validSCC()
	noFSGroupOptions.FSGroup.Type = ""

	invalidFSGroupStratType := validSCC()
	invalidFSGroupStratType.FSGroup.Type = "invalid"

	noSupplementalGroupsOptions := validSCC()
	noSupplementalGroupsOptions.SupplementalGroups.Type = ""

	invalidSupGroupStratType := validSCC()
	invalidSupGroupStratType.SupplementalGroups.Type = "invalid"

	invalidRangeMinGreaterThanMax := validSCC()
	invalidRangeMinGreaterThanMax.FSGroup.Ranges = []securityv1.IDRange{
		{Min: 2, Max: 1},
	}

	invalidRangeNegativeMin := validSCC()
	invalidRangeNegativeMin.FSGroup.Ranges = []securityv1.IDRange{
		{Min: -1, Max: 10},
	}

	invalidRangeNegativeMax := validSCC()
	invalidRangeNegativeMax.FSGroup.Ranges = []securityv1.IDRange{
		{Min: 1, Max: -10},
	}

	negativePriority := validSCC()
	negativePriority.Priority = &invalidPriority

	requiredCapAddAndDrop := validSCC()
	requiredCapAddAndDrop.DefaultAddCapabilities = []kcorev1.Capability{"foo"}
	requiredCapAddAndDrop.RequiredDropCapabilities = []kcorev1.Capability{"foo"}

	allowedCapListedInRequiredDrop := validSCC()
	allowedCapListedInRequiredDrop.RequiredDropCapabilities = []kcorev1.Capability{"foo"}
	allowedCapListedInRequiredDrop.AllowedCapabilities = []kcorev1.Capability{"foo"}

	wildcardAllowedCapAndRequiredDrop := validSCC()
	wildcardAllowedCapAndRequiredDrop.RequiredDropCapabilities = []kcorev1.Capability{"foo"}
	wildcardAllowedCapAndRequiredDrop.AllowedCapabilities = []kcorev1.Capability{securityv1.AllowAllCapabilities}

	emptyFlexDriver := validSCC()
	emptyFlexDriver.Volumes = []securityv1.FSType{securityv1.FSTypeFlexVolume}
	emptyFlexDriver.AllowedFlexVolumes = []securityv1.AllowedFlexVolume{{}}

	nonEmptyFlexVolumes := validSCC()
	nonEmptyFlexVolumes.AllowedFlexVolumes = []securityv1.AllowedFlexVolume{{Driver: "example/driver"}}

	invalidDefaultAllowPrivilegeEscalation := validSCC()
	invalidDefaultAllowPrivilegeEscalation.DefaultAllowPrivilegeEscalation = &yes
	invalidDefaultAllowPrivilegeEscalation.AllowPrivilegeEscalation = &no

	invalidAllowedUnsafeSysctlPattern := validSCC()
	invalidAllowedUnsafeSysctlPattern.AllowedUnsafeSysctls = []string{"a.*.b"}

	invalidForbiddenSysctlPattern := validSCC()
	invalidForbiddenSysctlPattern.ForbiddenSysctls = []string{"a.*.b"}

	invalidOverlappingSysctls := validSCC()
	invalidOverlappingSysctls.ForbiddenSysctls = []string{"kernel.*", "net.ipv4.ip_local_port_range"}
	invalidOverlappingSysctls.AllowedUnsafeSysctls = []string{"kernel.shmmax", "net.ipv4.ip_local_port_range"}

	invalidDuplicatedSysctls := validSCC()
	invalidDuplicatedSysctls.ForbiddenSysctls = []string{"net.ipv4.ip_local_port_range"}
	invalidDuplicatedSysctls.AllowedUnsafeSysctls = []string{"net.ipv4.ip_local_port_range"}

	errorCases := map[string]struct {
		scc         *securityv1.SecurityContextConstraints
		errorType   field.ErrorType
		errorDetail string
	}{
		"no user options": {
			scc:         noUserOptions,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "invalid strategy type.  Valid values are MustRunAs, MustRunAsNonRoot, MustRunAsRange, RunAsAny",
		},
		"no selinux options": {
			scc:         noSELinuxOptions,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "invalid strategy type.  Valid values are MustRunAs, RunAsAny",
		},
		"no fsgroup options": {
			scc:         noFSGroupOptions,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: "supported values: \"MustRunAs\", \"RunAsAny\"",
		},
		"no sup group options": {
			scc:         noSupplementalGroupsOptions,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: "supported values: \"MustRunAs\", \"RunAsAny\"",
		},
		"invalid user strategy type": {
			scc:         invalidUserStratType,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "invalid strategy type.  Valid values are MustRunAs, MustRunAsNonRoot, MustRunAsRange, RunAsAny",
		},
		"invalid selinux strategy type": {
			scc:         invalidSELinuxStratType,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "invalid strategy type.  Valid values are MustRunAs, RunAsAny",
		},
		"invalid sup group strategy type": {
			scc:         invalidSupGroupStratType,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: "supported values: \"MustRunAs\", \"RunAsAny\"",
		},
		"invalid fs group strategy type": {
			scc:         invalidFSGroupStratType,
			errorType:   field.ErrorTypeNotSupported,
			errorDetail: "supported values: \"MustRunAs\", \"RunAsAny\"",
		},
		"invalid uid": {
			scc:         invalidUIDSCC,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "uid cannot be negative",
		},
		"missing object meta name": {
			scc:         missingObjectMetaName,
			errorType:   field.ErrorTypeRequired,
			errorDetail: "name or generateName is required",
		},
		"invalid range min greater than max": {
			scc:         invalidRangeMinGreaterThanMax,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "min cannot be greater than max",
		},
		"invalid range negative min": {
			scc:         invalidRangeNegativeMin,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "min cannot be negative",
		},
		"invalid range negative max": {
			scc:         invalidRangeNegativeMax,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "max cannot be negative",
		},
		"negative priority": {
			scc:         negativePriority,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "priority cannot be negative",
		},
		"invalid required caps": {
			scc:         requiredCapAddAndDrop,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "capability is listed in defaultAddCapabilities and requiredDropCapabilities",
		},
		"allowed cap listed in required drops": {
			scc:         allowedCapListedInRequiredDrop,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "capability is listed in allowedCapabilities and requiredDropCapabilities",
		},
		"all caps allowed by a wildcard and required drops is not empty": {
			scc:         wildcardAllowedCapAndRequiredDrop,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "required capabilities must be empty when all capabilities are allowed by a wildcard",
		},
		"empty flex volume driver": {
			scc:         emptyFlexDriver,
			errorType:   field.ErrorTypeRequired,
			errorDetail: "must specify a driver",
		},
		"non-empty allowed flex volumes": {
			scc:         nonEmptyFlexVolumes,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "volumes does not include 'flexVolume' or '*', so no flex volumes are allowed",
		},
		"invalid defaultAllowPrivilegeEscalation": {
			scc:         invalidDefaultAllowPrivilegeEscalation,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: "Cannot set DefaultAllowPrivilegeEscalation to true without also setting AllowPrivilegeEscalation to true",
		},
		"invalid allowed unsafe sysctl pattern": {
			scc:         invalidAllowedUnsafeSysctlPattern,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: fmt.Sprintf("must have at most 253 characters and match regex %s", sysctlPatternFmt),
		},
		"invalid forbidden sysctl pattern": {
			scc:         invalidForbiddenSysctlPattern,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: fmt.Sprintf("must have at most 253 characters and match regex %s", sysctlPatternFmt),
		},
		"invalid overlapping sysctl pattern": {
			scc:         invalidOverlappingSysctls,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: fmt.Sprintf("sysctl overlaps with %s", invalidOverlappingSysctls.ForbiddenSysctls[0]),
		},
		"invalid duplicated sysctls": {
			scc:         invalidDuplicatedSysctls,
			errorType:   field.ErrorTypeInvalid,
			errorDetail: fmt.Sprintf("sysctl overlaps with %s", invalidDuplicatedSysctls.AllowedUnsafeSysctls[0]),
		},
	}

	for k, v := range errorCases {
		t.Run(k, func(t *testing.T) {
			if errs := ValidateSecurityContextConstraints(v.scc); len(errs) == 0 || errs[0].Type != v.errorType || errs[0].Detail != v.errorDetail {
				t.Errorf("Expected error type %q with detail %q, got %v", v.errorType, v.errorDetail, errs)
			}
		})
	}

	var validUID int64 = 1

	mustRunAs := validSCC()
	mustRunAs.FSGroup.Type = securityv1.FSGroupStrategyMustRunAs
	mustRunAs.SupplementalGroups.Type = securityv1.SupplementalGroupsStrategyMustRunAs
	mustRunAs.RunAsUser.Type = securityv1.RunAsUserStrategyMustRunAs
	mustRunAs.RunAsUser.UID = &validUID
	mustRunAs.SELinuxContext.Type = securityv1.SELinuxStrategyMustRunAs

	runAsNonRoot := validSCC()
	runAsNonRoot.RunAsUser.Type = securityv1.RunAsUserStrategyMustRunAsNonRoot

	caseInsensitiveAddDrop := validSCC()
	caseInsensitiveAddDrop.DefaultAddCapabilities = []kcorev1.Capability{"foo"}
	caseInsensitiveAddDrop.RequiredDropCapabilities = []kcorev1.Capability{"FOO"}

	caseInsensitiveAllowedDrop := validSCC()
	caseInsensitiveAllowedDrop.RequiredDropCapabilities = []kcorev1.Capability{"FOO"}
	caseInsensitiveAllowedDrop.AllowedCapabilities = []kcorev1.Capability{"foo"}

	flexvolumeWhenFlexVolumesAllowed := validSCC()
	flexvolumeWhenFlexVolumesAllowed.Volumes = []securityv1.FSType{securityv1.FSTypeFlexVolume}
	flexvolumeWhenFlexVolumesAllowed.AllowedFlexVolumes = []securityv1.AllowedFlexVolume{
		{Driver: "example/driver1"},
	}

	flexvolumeWhenAllVolumesAllowed := validSCC()
	flexvolumeWhenAllVolumesAllowed.Volumes = []securityv1.FSType{securityv1.FSTypeAll}
	flexvolumeWhenAllVolumesAllowed.AllowedFlexVolumes = []securityv1.AllowedFlexVolume{
		{Driver: "example/driver2"},
	}

	validDefaultAllowPrivilegeEscalation := validSCC()
	validDefaultAllowPrivilegeEscalation.DefaultAllowPrivilegeEscalation = &yes
	validDefaultAllowPrivilegeEscalation.AllowPrivilegeEscalation = &yes

	withForbiddenSysctl := validSCC()
	withForbiddenSysctl.ForbiddenSysctls = []string{"net.*"}

	withAllowedUnsafeSysctl := validSCC()
	withAllowedUnsafeSysctl.AllowedUnsafeSysctls = []string{"net.ipv4.tcp_max_syn_backlog"}

	successCases := map[string]struct {
		scc *securityv1.SecurityContextConstraints
	}{
		"must run as": {
			scc: mustRunAs,
		},
		"run as any": {
			scc: validSCC(),
		},
		"run as non-root (user only)": {
			scc: runAsNonRoot,
		},
		"comparison for add -> drop is case sensitive": {
			scc: caseInsensitiveAddDrop,
		},
		"comparison for allowed -> drop is case sensitive": {
			scc: caseInsensitiveAllowedDrop,
		},
		"allow white-listed flexVolume when flex volumes are allowed": {
			scc: flexvolumeWhenFlexVolumesAllowed,
		},
		"allow white-listed flexVolume when all volumes are allowed": {
			scc: flexvolumeWhenAllVolumesAllowed,
		},
		"valid defaultAllowPrivilegeEscalation as true": {
			scc: validDefaultAllowPrivilegeEscalation,
		},
		"with network sysctls forbidden": {
			scc: withForbiddenSysctl,
		},
		"with unsafe net.ipv4.tcp_max_syn_backlog sysctl allowed": {
			scc: withAllowedUnsafeSysctl,
		},
	}

	for k, v := range successCases {
		if errs := ValidateSecurityContextConstraints(v.scc); len(errs) != 0 {
			t.Errorf("Expected success for %q, got %v", k, errs)
		}
	}
}
