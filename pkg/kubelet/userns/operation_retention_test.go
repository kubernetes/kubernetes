//go:build !windows

/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package userns

import (
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
)

func TestOrphanUserNamespaceWaitsForPodOperation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.UserNamespacesSupport, true)
	for _, onDisk := range []bool{true, false} {
		t.Run(map[bool]string{true: "on disk", false: "allocation only"}[onDisk], func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			uid := types.UID("orphan-uid")
			host := &testUserNsPodsManager{podDir: t.TempDir()}
			if onDisk {
				host.podList = []types.UID{uid}
			}
			manager, err := MakeUserNsManager(logger, host, nil)
			require.NoError(t, err)
			require.NoError(t, manager.record(logger, uid, minimumMappingUID, testUserNsLength))
			retained := true
			predicate := func(actual types.UID) bool { require.Equal(t, uid, actual); return retained }
			require.NoError(t, manager.CleanupOrphanedPodUsernsAllocations(ctx, nil, nil, predicate))
			require.True(t, manager.podAllocated(uid), "runtime operation owns the allocation even without API/runtime inventory")
			retained = false
			require.NoError(t, manager.CleanupOrphanedPodUsernsAllocations(ctx, nil, nil, predicate))
			require.False(t, manager.podAllocated(uid))
		})
	}
}
