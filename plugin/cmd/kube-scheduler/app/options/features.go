package options

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

var featureGates = map[utilfeature.Feature]utilfeature.FeatureSpec{
	features.PersistentLocalVolumes: {Default: false, PreRelease: utilfeature.Alpha},
}
