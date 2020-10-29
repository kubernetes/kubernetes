package features

import (
	"strings"
	"testing"

	configv1 "github.com/openshift/api/config/v1"
)

func TestValidateCreateSpec(t *testing.T) {
	tests := []struct {
		name        string
		featureSet  string
		expectedErr string
	}{
		{
			name:        "empty",
			featureSet:  "",
			expectedErr: "",
		},
		{
			name:        "techpreview",
			featureSet:  string(configv1.TechPreviewNoUpgrade),
			expectedErr: "",
		},
		{
			name:        "not real",
			featureSet:  "fake-value",
			expectedErr: "Unsupported value",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := validateFeatureGateSpecCreate(configv1.FeatureGateSpec{FeatureGateSelection: configv1.FeatureGateSelection{FeatureSet: configv1.FeatureSet(tc.featureSet)}})
			switch {
			case len(actual) == 0 && len(tc.expectedErr) == 0:
			case len(actual) == 0 && len(tc.expectedErr) != 0:
				t.Fatal(tc.expectedErr)
			case len(actual) != 0 && len(tc.expectedErr) == 0:
				t.Fatal(actual)
			case len(actual) != 0 && len(tc.expectedErr) != 0:
				found := false
				for _, actualErr := range actual {
					found = found || strings.Contains(actualErr.Error(), tc.expectedErr)
				}
				if !found {
					t.Fatal(actual)
				}
			default:
			}

		})
	}
}

func TestValidateUpdateSpec(t *testing.T) {
	tests := []struct {
		name          string
		featureSet    string
		oldFeatureSet string
		expectedErr   string
	}{
		{
			name:          "empty",
			featureSet:    "",
			oldFeatureSet: "",
			expectedErr:   "",
		},
		{
			name:          "change to techpreview",
			featureSet:    string(configv1.TechPreviewNoUpgrade),
			oldFeatureSet: "",
			expectedErr:   "",
		},
		{
			name:          "change from techpreview",
			featureSet:    "",
			oldFeatureSet: string(configv1.TechPreviewNoUpgrade),
			expectedErr:   "once enabled, tech preview features may not be disabled",
		},
		{
			name:          "change from custom",
			featureSet:    string(configv1.TechPreviewNoUpgrade),
			oldFeatureSet: string(configv1.CustomNoUpgrade),
			expectedErr:   "once enabled, custom feature gates may not be disabled",
		},
		{
			name:          "unknown, but no change",
			featureSet:    "fake-value",
			oldFeatureSet: "fake-value",
			expectedErr:   "",
		},
		{
			name:          "unknown, with change",
			featureSet:    "fake-value",
			oldFeatureSet: "fake-value-2",
			expectedErr:   "Unsupported value",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := validateFeatureGateSpecUpdate(
				configv1.FeatureGateSpec{FeatureGateSelection: configv1.FeatureGateSelection{FeatureSet: configv1.FeatureSet(tc.featureSet)}},
				configv1.FeatureGateSpec{FeatureGateSelection: configv1.FeatureGateSelection{FeatureSet: configv1.FeatureSet(tc.oldFeatureSet)}},
			)
			switch {
			case len(actual) == 0 && len(tc.expectedErr) == 0:
			case len(actual) == 0 && len(tc.expectedErr) != 0:
				t.Fatal(tc.expectedErr)
			case len(actual) != 0 && len(tc.expectedErr) == 0:
				t.Fatal(actual)
			case len(actual) != 0 && len(tc.expectedErr) != 0:
				found := false
				for _, actualErr := range actual {
					found = found || strings.Contains(actualErr.Error(), tc.expectedErr)
				}
				if !found {
					t.Fatal(actual)
				}
			default:
			}

		})
	}
}
