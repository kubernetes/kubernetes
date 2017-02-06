package validation

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateServiceSubnet(t *testing.T) {
	var tests = []struct {
		s        string
		f        *field.Path
		expected bool
	}{
		{"", nil, false},
		{"this is not a cidr", nil, false}, // not a CIDR
		{"10.0.0.1", nil, false},           // not a CIDR
		{"192.0.2.0/1", nil, false},        // CIDR too smal
		{"192.0.2.0/24", nil, true},
	}
	for _, rt := range tests {
		actual := ValidateServiceSubnet(rt.s, rt.f)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateServiceSubnet:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}
