package validation

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/runonceduration"
)

// ValidateRunOnceDurationConfig validates the RunOnceDuration plugin configuration
func ValidateRunOnceDurationConfig(config *runonceduration.RunOnceDurationConfig) field.ErrorList {
	allErrs := field.ErrorList{}
	if config == nil || config.ActiveDeadlineSecondsOverride == nil {
		return allErrs
	}
	if *config.ActiveDeadlineSecondsOverride <= 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("activeDeadlineSecondsOverride"), config.ActiveDeadlineSecondsOverride, "must be greater than 0"))
	}
	return allErrs
}
