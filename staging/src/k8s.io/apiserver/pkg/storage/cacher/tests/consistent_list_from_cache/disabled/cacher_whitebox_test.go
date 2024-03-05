package disabled

import (
	"testing"

	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/tests/consistent_list_from_cache"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestGetListCacheBypass(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentListFromCache, false)()
	testCases := append(consistent_list_from_cache.CommonBypassTestCases,
		consistent_list_from_cache.BypassTestCase{Opts: storage.ListOptions{ResourceVersion: ""}, ExpectBypass: true},
	)
	for _, tc := range testCases {
		consistent_list_from_cache.RunGetListCacheBypassTest(t, tc.Opts, tc.ExpectBypass)
	}
}
