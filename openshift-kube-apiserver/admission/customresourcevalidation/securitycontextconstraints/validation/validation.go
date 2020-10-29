package validation

import (
	"fmt"
	"regexp"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/validation"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	kapivalidation "k8s.io/kubernetes/pkg/apis/core/validation"

	securityv1 "github.com/openshift/api/security/v1"
)

// ValidateSecurityContextConstraintsName can be used to check whether the given
// security context constraint name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateSecurityContextConstraintsName = apimachineryvalidation.NameIsDNSSubdomain

func ValidateSecurityContextConstraints(scc *securityv1.SecurityContextConstraints) field.ErrorList {
	allErrs := validation.ValidateObjectMeta(&scc.ObjectMeta, false, ValidateSecurityContextConstraintsName, field.NewPath("metadata"))

	if scc.Priority != nil {
		if *scc.Priority < 0 {
			allErrs = append(allErrs, field.Invalid(field.NewPath("priority"), *scc.Priority, "priority cannot be negative"))
		}
	}

	// ensure the user strategy has a valid type
	runAsUserPath := field.NewPath("runAsUser")
	switch scc.RunAsUser.Type {
	case securityv1.RunAsUserStrategyMustRunAs, securityv1.RunAsUserStrategyMustRunAsNonRoot, securityv1.RunAsUserStrategyRunAsAny, securityv1.RunAsUserStrategyMustRunAsRange:
		//good types
	default:
		msg := fmt.Sprintf("invalid strategy type.  Valid values are %s, %s, %s, %s", securityv1.RunAsUserStrategyMustRunAs, securityv1.RunAsUserStrategyMustRunAsNonRoot, securityv1.RunAsUserStrategyMustRunAsRange, securityv1.RunAsUserStrategyRunAsAny)
		allErrs = append(allErrs, field.Invalid(runAsUserPath.Child("type"), scc.RunAsUser.Type, msg))
	}

	// if specified, uid cannot be negative
	if scc.RunAsUser.UID != nil {
		if *scc.RunAsUser.UID < 0 {
			allErrs = append(allErrs, field.Invalid(runAsUserPath.Child("uid"), *scc.RunAsUser.UID, "uid cannot be negative"))
		}
	}

	// ensure the selinux strategy has a valid type
	seLinuxContextPath := field.NewPath("seLinuxContext")
	switch scc.SELinuxContext.Type {
	case securityv1.SELinuxStrategyMustRunAs, securityv1.SELinuxStrategyRunAsAny:
		//good types
	default:
		msg := fmt.Sprintf("invalid strategy type.  Valid values are %s, %s", securityv1.SELinuxStrategyMustRunAs, securityv1.SELinuxStrategyRunAsAny)
		allErrs = append(allErrs, field.Invalid(seLinuxContextPath.Child("type"), scc.SELinuxContext.Type, msg))
	}

	// ensure the fsgroup strategy has a valid type
	if scc.FSGroup.Type != securityv1.FSGroupStrategyMustRunAs && scc.FSGroup.Type != securityv1.FSGroupStrategyRunAsAny {
		allErrs = append(allErrs, field.NotSupported(field.NewPath("fsGroup", "type"), scc.FSGroup.Type,
			[]string{string(securityv1.FSGroupStrategyMustRunAs), string(securityv1.FSGroupStrategyRunAsAny)}))
	}
	allErrs = append(allErrs, validateIDRanges(scc.FSGroup.Ranges, field.NewPath("fsGroup"))...)

	if scc.SupplementalGroups.Type != securityv1.SupplementalGroupsStrategyMustRunAs &&
		scc.SupplementalGroups.Type != securityv1.SupplementalGroupsStrategyRunAsAny {
		allErrs = append(allErrs, field.NotSupported(field.NewPath("supplementalGroups", "type"), scc.SupplementalGroups.Type,
			[]string{string(securityv1.SupplementalGroupsStrategyMustRunAs), string(securityv1.SupplementalGroupsStrategyRunAsAny)}))
	}
	allErrs = append(allErrs, validateIDRanges(scc.SupplementalGroups.Ranges, field.NewPath("supplementalGroups"))...)

	// validate capabilities
	allErrs = append(allErrs, validateSCCCapsAgainstDrops(scc.RequiredDropCapabilities, scc.DefaultAddCapabilities, field.NewPath("defaultAddCapabilities"))...)
	allErrs = append(allErrs, validateSCCCapsAgainstDrops(scc.RequiredDropCapabilities, scc.AllowedCapabilities, field.NewPath("allowedCapabilities"))...)

	if hasCap(securityv1.AllowAllCapabilities, scc.AllowedCapabilities) && len(scc.RequiredDropCapabilities) > 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("requiredDropCapabilities"), scc.RequiredDropCapabilities,
			"required capabilities must be empty when all capabilities are allowed by a wildcard"))
	}

	allErrs = append(allErrs, validateSCCDefaultAllowPrivilegeEscalation(field.NewPath("defaultAllowPrivilegeEscalation"), scc.DefaultAllowPrivilegeEscalation, scc.AllowPrivilegeEscalation)...)

	allowsFlexVolumes := false
	hasNoneVolume := false

	if len(scc.Volumes) > 0 {
		for _, fsType := range scc.Volumes {
			if fsType == securityv1.FSTypeNone {
				hasNoneVolume = true

			} else if fsType == securityv1.FSTypeFlexVolume || fsType == securityv1.FSTypeAll {
				allowsFlexVolumes = true
			}
		}
	}

	if hasNoneVolume && len(scc.Volumes) > 1 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("volumes"), scc.Volumes,
			"if 'none' is specified, no other values are allowed"))
	}

	if len(scc.AllowedFlexVolumes) > 0 {
		if allowsFlexVolumes {
			for idx, allowedFlexVolume := range scc.AllowedFlexVolumes {
				if len(allowedFlexVolume.Driver) == 0 {
					allErrs = append(allErrs, field.Required(field.NewPath("allowedFlexVolumes").Index(idx).Child("driver"),
						"must specify a driver"))
				}
			}
		} else {
			allErrs = append(allErrs, field.Invalid(field.NewPath("allowedFlexVolumes"), scc.AllowedFlexVolumes,
				"volumes does not include 'flexVolume' or '*', so no flex volumes are allowed"))
		}
	}

	allowedUnsafeSysctlsPath := field.NewPath("allowedUnsafeSysctls")
	forbiddenSysctlsPath := field.NewPath("forbiddenSysctls")
	allErrs = append(allErrs, validateSCCSysctls(allowedUnsafeSysctlsPath, scc.AllowedUnsafeSysctls)...)
	allErrs = append(allErrs, validateSCCSysctls(forbiddenSysctlsPath, scc.ForbiddenSysctls)...)
	allErrs = append(allErrs, validatePodSecurityPolicySysctlListsDoNotOverlap(allowedUnsafeSysctlsPath, forbiddenSysctlsPath, scc.AllowedUnsafeSysctls, scc.ForbiddenSysctls)...)

	return allErrs
}

const sysctlPatternSegmentFmt string = "([a-z0-9][-_a-z0-9]*)?[a-z0-9*]"
const sysctlPatternFmt string = "(" + kapivalidation.SysctlSegmentFmt + "\\.)*" + sysctlPatternSegmentFmt

var sysctlPatternRegexp = regexp.MustCompile("^" + sysctlPatternFmt + "$")

func IsValidSysctlPattern(name string) bool {
	if len(name) > kapivalidation.SysctlMaxLength {
		return false
	}
	return sysctlPatternRegexp.MatchString(name)
}

// validatePodSecurityPolicySysctlListsDoNotOverlap validates the values in forbiddenSysctls and allowedSysctls fields do not overlap.
func validatePodSecurityPolicySysctlListsDoNotOverlap(allowedSysctlsFldPath, forbiddenSysctlsFldPath *field.Path, allowedUnsafeSysctls, forbiddenSysctls []string) field.ErrorList {
	allErrs := field.ErrorList{}
	for i, allowedSysctl := range allowedUnsafeSysctls {
		isAllowedSysctlPattern := false
		allowedSysctlPrefix := ""
		if strings.HasSuffix(allowedSysctl, "*") {
			isAllowedSysctlPattern = true
			allowedSysctlPrefix = strings.TrimSuffix(allowedSysctl, "*")
		}
		for j, forbiddenSysctl := range forbiddenSysctls {
			isForbiddenSysctlPattern := false
			forbiddenSysctlPrefix := ""
			if strings.HasSuffix(forbiddenSysctl, "*") {
				isForbiddenSysctlPattern = true
				forbiddenSysctlPrefix = strings.TrimSuffix(forbiddenSysctl, "*")
			}
			switch {
			case isAllowedSysctlPattern && isForbiddenSysctlPattern:
				if strings.HasPrefix(allowedSysctlPrefix, forbiddenSysctlPrefix) {
					allErrs = append(allErrs, field.Invalid(allowedSysctlsFldPath.Index(i), allowedUnsafeSysctls[i], fmt.Sprintf("sysctl overlaps with %v", forbiddenSysctl)))
				} else if strings.HasPrefix(forbiddenSysctlPrefix, allowedSysctlPrefix) {
					allErrs = append(allErrs, field.Invalid(forbiddenSysctlsFldPath.Index(j), forbiddenSysctls[j], fmt.Sprintf("sysctl overlaps with %v", allowedSysctl)))
				}
			case isAllowedSysctlPattern:
				if strings.HasPrefix(forbiddenSysctl, allowedSysctlPrefix) {
					allErrs = append(allErrs, field.Invalid(forbiddenSysctlsFldPath.Index(j), forbiddenSysctls[j], fmt.Sprintf("sysctl overlaps with %v", allowedSysctl)))
				}
			case isForbiddenSysctlPattern:
				if strings.HasPrefix(allowedSysctl, forbiddenSysctlPrefix) {
					allErrs = append(allErrs, field.Invalid(allowedSysctlsFldPath.Index(i), allowedUnsafeSysctls[i], fmt.Sprintf("sysctl overlaps with %v", forbiddenSysctl)))
				}
			default:
				if allowedSysctl == forbiddenSysctl {
					allErrs = append(allErrs, field.Invalid(allowedSysctlsFldPath.Index(i), allowedUnsafeSysctls[i], fmt.Sprintf("sysctl overlaps with %v", forbiddenSysctl)))
				}
			}
		}
	}
	return allErrs
}

// validatePodSecurityPolicySysctls validates the sysctls fields of PodSecurityPolicy.
func validateSCCSysctls(fldPath *field.Path, sysctls []string) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(sysctls) == 0 {
		return allErrs
	}

	coversAll := false
	for i, s := range sysctls {
		if len(s) == 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i), sysctls[i], fmt.Sprintf("empty sysctl not allowed")))
		} else if !IsValidSysctlPattern(string(s)) {
			allErrs = append(
				allErrs,
				field.Invalid(fldPath.Index(i), sysctls[i], fmt.Sprintf("must have at most %d characters and match regex %s",
					kapivalidation.SysctlMaxLength,
					sysctlPatternFmt,
				)),
			)
		} else if s[0] == '*' {
			coversAll = true
		}
	}

	if coversAll && len(sysctls) > 1 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("items"), fmt.Sprintf("if '*' is present, must not specify other sysctls")))
	}

	return allErrs
}

// validateSCCCapsAgainstDrops ensures an allowed cap is not listed in the required drops.
func validateSCCCapsAgainstDrops(requiredDrops []corev1.Capability, capsToCheck []corev1.Capability, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if requiredDrops == nil {
		return allErrs
	}
	for _, cap := range capsToCheck {
		if hasCap(cap, requiredDrops) {
			allErrs = append(allErrs, field.Invalid(fldPath, cap,
				fmt.Sprintf("capability is listed in %s and requiredDropCapabilities", fldPath.String())))
		}
	}
	return allErrs
}

// validateSCCDefaultAllowPrivilegeEscalation validates the DefaultAllowPrivilegeEscalation field against the AllowPrivilegeEscalation field of a SecurityContextConstraints.
func validateSCCDefaultAllowPrivilegeEscalation(fldPath *field.Path, defaultAllowPrivilegeEscalation, allowPrivilegeEscalation *bool) field.ErrorList {
	allErrs := field.ErrorList{}
	if defaultAllowPrivilegeEscalation != nil && allowPrivilegeEscalation != nil && *defaultAllowPrivilegeEscalation && !*allowPrivilegeEscalation {
		allErrs = append(allErrs, field.Invalid(fldPath, defaultAllowPrivilegeEscalation, "Cannot set DefaultAllowPrivilegeEscalation to true without also setting AllowPrivilegeEscalation to true"))
	}

	return allErrs
}

// hasCap checks for needle in haystack.
func hasCap(needle corev1.Capability, haystack []corev1.Capability) bool {
	for _, c := range haystack {
		if needle == c {
			return true
		}
	}
	return false
}

// validateIDRanges ensures the range is valid.
func validateIDRanges(rng []securityv1.IDRange, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	for i, r := range rng {
		// if 0 <= Min <= Max then we do not need to validate max.  It is always greater than or
		// equal to 0 and Min.
		minPath := fldPath.Child("ranges").Index(i).Child("min")
		maxPath := fldPath.Child("ranges").Index(i).Child("max")

		if r.Min < 0 {
			allErrs = append(allErrs, field.Invalid(minPath, r.Min, "min cannot be negative"))
		}
		if r.Max < 0 {
			allErrs = append(allErrs, field.Invalid(maxPath, r.Max, "max cannot be negative"))
		}
		if r.Min > r.Max {
			allErrs = append(allErrs, field.Invalid(minPath, r, "min cannot be greater than max"))
		}
	}

	return allErrs
}

func ValidateSecurityContextConstraintsUpdate(newScc, oldScc *securityv1.SecurityContextConstraints) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&newScc.ObjectMeta, &oldScc.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateSecurityContextConstraints(newScc)...)
	return allErrs
}
