package dns

import (
	"testing"

	operatorv1 "github.com/openshift/api/operator/v1"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// TestFailValidateDNSSpec verifies that validateDNSSpec rejects invalid specs.
func TestFailValidateDNSSpec(t *testing.T) {
	errorCases := map[string]struct {
		spec       operatorv1.DNSSpec
		errorType  field.ErrorType
		errorField string
	}{
		"invalid toleration": {
			spec: operatorv1.DNSSpec{
				NodePlacement: operatorv1.DNSNodePlacement{
					Tolerations: []corev1.Toleration{{
						Key:      "x",
						Operator: corev1.TolerationOpExists,
						Effect:   "NoExcute",
					}},
				},
			},
			errorType:  field.ErrorTypeNotSupported,
			errorField: "spec.nodePlacement.tolerations[0].effect",
		},
		"invalid node selector": {
			spec: operatorv1.DNSSpec{
				NodePlacement: operatorv1.DNSNodePlacement{
					NodeSelector: map[string]string{
						"-": "foo",
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.nodePlacement.nodeSelector",
		},
	}

	for tcName, tc := range errorCases {
		errs := validateDNSSpec(tc.spec)
		if len(errs) == 0 {
			t.Errorf("%q: should have failed but did not", tcName)
		}

		for _, e := range errs {
			if e.Type != tc.errorType {
				t.Errorf("%q: expected errors of type '%s', got %v:", tcName, tc.errorType, e)
			}

			if e.Field != tc.errorField {
				t.Errorf("%q: expected errors in field '%s', got %v:", tcName, tc.errorField, e)
			}
		}
	}
}

// TestSucceedValidateDNSSpec verifies that validateDNSSpec accepts valid specs.
func TestSucceedValidateDNSSpec(t *testing.T) {
	successCases := map[string]operatorv1.DNSSpec{
		"empty": {},
		"toleration + node selector": {
			NodePlacement: operatorv1.DNSNodePlacement{
				NodeSelector: map[string]string{
					"node-role.kubernetes.io/master": "",
				},
				Tolerations: []corev1.Toleration{{
					Key:      "node-role.kubernetes.io/master",
					Operator: corev1.TolerationOpExists,
					Effect:   corev1.TaintEffectNoExecute,
				}},
			},
		},
	}

	for tcName, s := range successCases {
		errs := validateDNSSpec(s)
		if len(errs) != 0 {
			t.Errorf("%q: expected success, but failed: %v", tcName, errs.ToAggregate().Error())
		}
	}
}
