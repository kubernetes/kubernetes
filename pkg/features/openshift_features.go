package features

import (
	"k8s.io/component-base/featuregate"
)

var RouteExternalCertificate featuregate.Feature = "RouteExternalCertificate"

// registerOpenshiftFeatures injects openshift-specific feature gates
func registerOpenshiftFeatures() {
	defaultKubernetesFeatureGates[RouteExternalCertificate] = featuregate.FeatureSpec{
		Default:    false,
		PreRelease: featuregate.Alpha,
	}
}
