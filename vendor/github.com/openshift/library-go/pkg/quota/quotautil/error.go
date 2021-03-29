package quotautil

import (
	"strings"

	apierrs "k8s.io/apimachinery/pkg/api/errors"
)

// errMessageString is a part of error message copied from quotaAdmission.Admit() method in
// k8s.io/kubernetes/plugin/pkg/admission/resourcequota/admission.go module
const errQuotaMessageString = `exceeded quota:`
const errQuotaUnknownMessageString = `status unknown for quota:`
const errLimitsMessageString = `exceeds the maximum limit`

// IsErrorQuotaExceeded returns true if the given error stands for a denied request caused by detected quota
// abuse.
func IsErrorQuotaExceeded(err error) bool {
	if isForbidden := apierrs.IsForbidden(err); isForbidden || apierrs.IsInvalid(err) {
		lowered := strings.ToLower(err.Error())
		// the limit error message can be accompanied only by Invalid reason
		if strings.Contains(lowered, errLimitsMessageString) {
			return true
		}
		// the quota error message can be accompanied only by Forbidden reason
		if isForbidden && (strings.Contains(lowered, errQuotaMessageString) || strings.Contains(lowered, errQuotaUnknownMessageString)) {
			return true
		}
	}
	return false
}

// IsErrorLimitExceeded returns true if the given error is a limit error.
func IsErrorLimitExceeded(err error) bool {
	if isForbidden := apierrs.IsForbidden(err); isForbidden || apierrs.IsInvalid(err) {
		lowered := strings.ToLower(err.Error())
		// the limit error message can be accompanied only by Invalid reason
		if strings.Contains(lowered, errLimitsMessageString) {
			return true
		}
	}
	return false
}
