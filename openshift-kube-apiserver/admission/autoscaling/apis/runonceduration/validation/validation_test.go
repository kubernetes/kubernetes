package validation

import (
	"testing"

	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/runonceduration"
)

func TestRunOnceDurationConfigValidation(t *testing.T) {
	// Check invalid duration returns an error
	var invalidSecs int64 = -1
	invalidConfig := &runonceduration.RunOnceDurationConfig{
		ActiveDeadlineSecondsOverride: &invalidSecs,
	}
	errs := ValidateRunOnceDurationConfig(invalidConfig)
	if len(errs) == 0 {
		t.Errorf("Did not get expected error on invalid config")
	}

	// Check that valid duration returns no error
	var validSecs int64 = 5
	validConfig := &runonceduration.RunOnceDurationConfig{
		ActiveDeadlineSecondsOverride: &validSecs,
	}
	errs = ValidateRunOnceDurationConfig(validConfig)
	if len(errs) > 0 {
		t.Errorf("Unexpected error on valid config")
	}
}
