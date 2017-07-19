package options

import (
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

var featureGates = map[utilfeature.Feature]utilfeature.FeatureSpec{
	features.ExternalTrafficLocalOnly: {Default: true, PreRelease: utilfeature.GA},
	features.AppArmor:                 {Default: true, PreRelease: utilfeature.Beta},
	features.StreamingProxyRedirects:  {Default: true, PreRelease: utilfeature.Beta},
	features.PodPriority:              {Default: false, PreRelease: utilfeature.Alpha},

	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	genericfeatures.AdvancedAuditing: {Default: false, PreRelease: utilfeature.Alpha},
}
