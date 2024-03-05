package enabled

import (
	"testing"

	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage/cacher/tests/consistent_list_from_cache"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestWaitUntilFreshAndListFromCache(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentListFromCache, true)()
	consistent_list_from_cache.RunWaitUntilFreshAndListFromCacheTest(t)
}

func TestWaitUntilFreshAndListTimeout(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentListFromCache, true)()
	consistent_list_from_cache.RunWaitUntilFreshAndListTimeoutTest(t)
}
