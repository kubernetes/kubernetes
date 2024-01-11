//go:build compatibility_testing

package features_test

import (
	"testing"

	features "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestCompatibilityFeatures(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompatibilityTestingAlphaFeature, true)()

}
