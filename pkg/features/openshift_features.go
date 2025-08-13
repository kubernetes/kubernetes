package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
)

var (
	RouteExternalCertificate        featuregate.Feature = "RouteExternalCertificate"
	MinimumKubeletVersion           featuregate.Feature = "MinimumKubeletVersion"
	StoragePerformantSecurityPolicy featuregate.Feature = "StoragePerformantSecurityPolicy"
)

// registerOpenshiftFeatures injects openshift-specific feature gates
func registerOpenshiftFeatures() {
	// Introduced in 4.16
	defaultVersionedKubernetesFeatureGates[RouteExternalCertificate] = featuregate.VersionedSpecs{
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
	}
	// Introduced in 4.19
	defaultVersionedKubernetesFeatureGates[MinimumKubeletVersion] = featuregate.VersionedSpecs{
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
	}
	// Introduced in 4.20
	defaultVersionedKubernetesFeatureGates[StoragePerformantSecurityPolicy] = featuregate.VersionedSpecs{
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
	}

	defaultKubernetesFeatureGateDependencies[RouteExternalCertificate] = []featuregate.Feature{}
	defaultKubernetesFeatureGateDependencies[MinimumKubeletVersion] = []featuregate.Feature{}
	defaultKubernetesFeatureGateDependencies[StoragePerformantSecurityPolicy] = []featuregate.Feature{}
}
