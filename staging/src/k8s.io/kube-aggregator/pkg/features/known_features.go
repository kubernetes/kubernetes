package features

import (
	clientfeatures "k8s.io/client-go/features"
	"k8s.io/utils/featuregates"
)

var (
	libraryFeatureSet = featuregates.NewSimpleFeatureSet()
)

func init() {
	libraryFeatureSet.AddFeatureSetsOrDie(
		clientfeatures.LibraryFeatureSet(),
	)
}

func LibraryFeatureSet() featuregates.FeatureSet {
	return libraryFeatureSet
}
