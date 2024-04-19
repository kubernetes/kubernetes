package user

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"

	securityv1 "github.com/openshift/api/security/v1"
)

// mustRunAs implements the RunAsUserSecurityContextConstraintsStrategy interface
type mustRunAs struct {
	opts *securityv1.RunAsUserStrategyOptions
}

var _ RunAsUserSecurityContextConstraintsStrategy = &mustRunAs{}

// NewMustRunAs provides a strategy that requires the container to run as a specific UID.
func NewMustRunAs(options *securityv1.RunAsUserStrategyOptions) (RunAsUserSecurityContextConstraintsStrategy, error) {
	if options == nil {
		return nil, fmt.Errorf("MustRunAs requires run as user options")
	}
	if options.UID == nil {
		return nil, fmt.Errorf("MustRunAs requires a UID")
	}
	return &mustRunAs{
		opts: options,
	}, nil
}

// Generate creates the uid based on policy rules.  MustRunAs returns the UID it is initialized with.
func (s *mustRunAs) Generate(pod *api.Pod, container *api.Container) (*int64, error) {
	return s.opts.UID, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *mustRunAs) Validate(fldPath *field.Path, _ *api.Pod, _ *api.Container, runAsNonRoot *bool, runAsUser *int64) field.ErrorList {
	allErrs := field.ErrorList{}

	if runAsUser == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("runAsUser"), ""))
		return allErrs
	}

	if *s.opts.UID != *runAsUser {
		detail := fmt.Sprintf("must be: %v", *s.opts.UID)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("runAsUser"), *runAsUser, detail))
		return allErrs
	}

	return allErrs
}
