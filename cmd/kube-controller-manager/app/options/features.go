package options

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

var featureGates = map[utilfeature.Feature]utilfeature.FeatureSpec{
	features.DynamicVolumeProvisioning: {Default: true, PreRelease: utilfeature.Alpha},
	features.TaintBasedEvictions:       {Default: false, PreRelease: utilfeature.Alpha},
}
