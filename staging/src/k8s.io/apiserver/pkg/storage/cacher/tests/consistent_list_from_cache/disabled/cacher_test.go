package disabled

import (
	"testing"

	"k8s.io/apiserver/pkg/features"
	cachertests "k8s.io/apiserver/pkg/storage/cacher/tests"
	"k8s.io/apiserver/pkg/storage/cacher/tests/consistent_list_from_cache"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func init() {
	cachertests.InitTestSchema()
}

func TestList(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentListFromCache, false)()
	consistent_list_from_cache.RunTestList(t)
}
