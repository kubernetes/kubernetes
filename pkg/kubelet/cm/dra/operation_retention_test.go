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

package dra

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/ktesting"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1"
	"k8s.io/kubernetes/pkg/kubelet/config"
)

func TestReconcileRetainsPodOperationClaims(t *testing.T) {
	for _, providerKind := range []string{"pending", "completed", "unaware"} {
		t.Run(providerKind, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			manager, err := NewManager(logger, fake.NewClientset(), t.TempDir())
			require.NoError(t, err)
			defer manager.Stop()
			manager.initDRAPluginManager(ctx, getFakeNode, time.Second)
			server, err := setupFakeDRADriverGRPCServer(ctx, false, nil, nil, nil, nil)
			require.NoError(t, err)
			defer server.teardownFn()
			require.NoError(t, manager.GetWatcherHandler().RegisterPlugin(ctx, driverName, server.socketName, []string{drapb.DRAPluginService}, nil))
			for _, uid := range []string{"pending", "unrelated"} {
				claim := genTestClaimInfo(types.UID(uid+"-claim-uid"), []string{uid}, true, nil)
				claim.ClaimName = uid + "-claim"
				manager.cache.add(claim)
			}
			manager.activePods = func() []*v1.Pod { return nil }
			manager.sourcesReady = config.NewSourcesReady(func(sets.Set[string]) bool { return true })
			retained := providerKind == "pending"
			if providerKind != "unaware" {
				manager.podOperationMayUsePodResources = func(uid types.UID) bool { return uid == "pending" && retained }
			}
			hasClaim := func(name string) bool {
				manager.cache.RLock()
				defer manager.cache.RUnlock()
				_, exists := manager.cache.get(name, namespace)
				return exists
			}
			manager.reconcileLoop(ctx)
			require.Equal(t, retained, hasClaim("pending-claim"), "pending runtime work still owns its prepared ResourceClaim")
			require.False(t, hasClaim("unrelated-claim"))
			if retained {
				require.Equal(t, uint32(1), server.server.unprepareResourceCalls.Load())
			} else {
				require.Equal(t, uint32(2), server.server.unprepareResourceCalls.Load())
			}
			retained = false
			manager.reconcileLoop(ctx)
			require.False(t, hasClaim("pending-claim"))
			require.Equal(t, uint32(2), server.server.unprepareResourceCalls.Load())
		})
	}
}
