package selinux

import (
	"fmt"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	coreapi "k8s.io/kubernetes/pkg/apis/core"

	securityv1 "github.com/openshift/api/security/v1"
	"github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/util"
)

type mustRunAs struct {
	opts *securityv1.SELinuxContextStrategyOptions
}

var _ SELinuxSecurityContextConstraintsStrategy = &mustRunAs{}

func NewMustRunAs(options *securityv1.SELinuxContextStrategyOptions) (SELinuxSecurityContextConstraintsStrategy, error) {
	if options == nil {
		return nil, fmt.Errorf("MustRunAs requires SELinuxContextStrategyOptions")
	}
	if options.SELinuxOptions == nil {
		return nil, fmt.Errorf("MustRunAs requires SELinuxOptions")
	}
	return &mustRunAs{
		opts: options,
	}, nil
}

// Generate creates the SELinuxOptions based on constraint rules.
func (s *mustRunAs) Generate(_ *coreapi.Pod, _ *coreapi.Container) (*coreapi.SELinuxOptions, error) {
	return ToInternalSELinuxOptions(s.opts.SELinuxOptions)
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *mustRunAs) Validate(fldPath *field.Path, _ *coreapi.Pod, _ *coreapi.Container, seLinux *coreapi.SELinuxOptions) field.ErrorList {
	allErrs := field.ErrorList{}

	if seLinux == nil {
		allErrs = append(allErrs, field.Required(fldPath, ""))
		return allErrs
	}
	if !equalLevels(s.opts.SELinuxOptions.Level, seLinux.Level) {
		detail := fmt.Sprintf("must be %s", s.opts.SELinuxOptions.Level)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("level"), seLinux.Level, detail))
	}
	if seLinux.Role != s.opts.SELinuxOptions.Role {
		detail := fmt.Sprintf("must be %s", s.opts.SELinuxOptions.Role)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("role"), seLinux.Role, detail))
	}
	if seLinux.Type != s.opts.SELinuxOptions.Type {
		detail := fmt.Sprintf("must be %s", s.opts.SELinuxOptions.Type)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("type"), seLinux.Type, detail))
	}
	if seLinux.User != s.opts.SELinuxOptions.User {
		detail := fmt.Sprintf("must be %s", s.opts.SELinuxOptions.User)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("user"), seLinux.User, detail))
	}

	return allErrs
}

// equalLevels compares SELinux levels for equality.
func equalLevels(expected, actual string) bool {
	if expected == actual {
		return true
	}
	// "s0:c6,c0" => [ "s0", "c6,c0" ]
	expectedParts := strings.SplitN(expected, ":", 2)
	actualParts := strings.SplitN(actual, ":", 2)

	// both SELinux levels must be in a format "sX:cY"
	if len(expectedParts) != 2 || len(actualParts) != 2 {
		return false
	}

	if !equalSensitivity(expectedParts[0], actualParts[0]) {
		return false
	}

	if !equalCategories(expectedParts[1], actualParts[1]) {
		return false
	}

	return true
}

// equalSensitivity compares sensitivities of the SELinux levels for equality.
func equalSensitivity(expected, actual string) bool {
	return expected == actual
}

// equalCategories compares categories of the SELinux levels for equality.
func equalCategories(expected, actual string) bool {
	expectedCategories := strings.Split(expected, ",")
	actualCategories := strings.Split(actual, ",")

	sort.Strings(expectedCategories)
	sort.Strings(actualCategories)

	return util.EqualStringSlices(expectedCategories, actualCategories)
}
