package features

import (
	clientfeatures "k8s.io/client-go/features"
	"k8s.io/component-base/featuregate"
)

type clientFeatureGateAdapter struct {
	mfg featuregate.MutableFeatureGate
}

func (a *clientFeatureGateAdapter) Enabled(name clientfeatures.Feature) bool {
	return a.mfg.Enabled(featuregate.Feature(name))
}

func (a *clientFeatureGateAdapter) Add(in map[clientfeatures.Feature]clientfeatures.FeatureSpec) error {
	out := map[featuregate.Feature]featuregate.FeatureSpec{}
	for name, spec := range in {
		converted := featuregate.FeatureSpec{
			Default:       spec.Default,
			LockToDefault: spec.LockToDefault,
		}
		switch spec.PreRelease {
		case clientfeatures.Alpha:
			converted.PreRelease = featuregate.Alpha
		case clientfeatures.Beta:
			converted.PreRelease = featuregate.Beta
		case clientfeatures.GA:
			converted.PreRelease = featuregate.GA
		case clientfeatures.Deprecated:
			converted.PreRelease = featuregate.Deprecated
		default:
			panic("todo")
		}
		out[featuregate.Feature(name)] = converted
	}
	return a.mfg.Add(out)
}
