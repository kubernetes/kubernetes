/*
Copyright 2025 The Kubernetes Authors.

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

package discovery

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiserverfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	testutil "k8s.io/kubernetes/test/utils"
)

func TestMergedDiscoveryBasic(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// Enable feature gates for merged discovery and peer proxy.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, apiserverfeatures.APIServerIdentity, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.UnknownVersionInteroperabilityProxy, true)

	// Create shared etcd.
	etcd := framework.SharedEtcd()

	// Create certificates for aggregation and client-cert auth.
	proxyCA, err := createProxyCertContent()
	require.NoError(t, err)

	// Start two API servers with different API configurations.
	server.SetHostnameFuncForTests("test-server-a")
	serverA := kubeapiservertesting.StartTestServerOrDie(t, &kubeapiservertesting.TestServerInstanceOptions{
		EnableCertAuth: true,
		ProxyCA:        &proxyCA,
	}, []string{
		"--runtime-config=apps/v1=false",
	}, etcd)
	t.Cleanup(serverA.TearDownFn)
	clientA, err := kubernetes.NewForConfig(serverA.ClientConfig)
	require.NoError(t, err)

	server.SetHostnameFuncForTests("test-server-b")
	serverB := kubeapiservertesting.StartTestServerOrDie(t, &kubeapiservertesting.TestServerInstanceOptions{
		EnableCertAuth: true,
		ProxyCA:        &proxyCA,
	}, []string{
		"--runtime-config=batch/v1=false",
	}, etcd)
	t.Cleanup(serverB.TearDownFn)

	clientB, err := kubernetes.NewForConfig(serverB.ClientConfig)
	require.NoError(t, err)

	t.Run("CheckIdentityLeases", func(t *testing.T) {
		// Check for identity leases in the kube-system namespace
		leases, err := clientA.CoordinationV1().Leases("kube-system").List(ctx, metav1.ListOptions{
			LabelSelector: "apiserver.kubernetes.io/identity=kube-apiserver",
		})
		if err != nil {
			t.Logf("Failed to list identity leases: %v", err)
		} else {
			t.Logf("Found %d identity leases:", len(leases.Items))
		}
	})

	t.Run("LocalDiscoveryWorks", func(t *testing.T) {
		testLocalDiscovery(t, ctx, clientA, clientB)
	})

	t.Run("MergedDiscoveryEndpoint", func(t *testing.T) {
		testMergedDiscoveryEndpoint(t, ctx, clientA, clientB)
	})

	t.Run("ServerIDsInMergedDiscovery", func(t *testing.T) {
		testServerIDsInMergedDiscovery(t, ctx, clientA, clientB)
	})
}

func testLocalDiscovery(t *testing.T, ctx context.Context, clientA, clientB kubernetes.Interface) {
	// Verify serverA does NOT have apps/v1 in local discovery (disabled via runtime-config)
	err := wait.PollUntilContextTimeout(ctx, 1*time.Second, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		groups, err := clientA.Discovery().ServerGroups()
		if err != nil {
			return false, nil
		}

		for _, group := range groups.Groups {
			if group.Name == "apps" {
				for _, version := range group.Versions {
					if version.Version == "v1" {
						// If we find apps/v1, this is unexpected - return false to continue polling.
						return false, nil
					}
				}
			}
		}
		return true, nil
	})
	require.NoError(t, err, "ServerA should NOT have apps/v1 in local discovery (disabled via runtime-config)")

	// Verify serverB does NOT have batch/v1 in local discovery (disabled via runtime-config)
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		groups, err := clientB.Discovery().ServerGroups()
		if err != nil {
			return false, nil
		}

		for _, group := range groups.Groups {
			if group.Name == "batch" {
				for _, version := range group.Versions {
					if version.Version == "v1" {
						// If we find batch/v1, this is unexpected - return false to continue polling.
						return false, nil
					}
				}
			}
		}
		return true, nil
	})
	require.NoError(t, err, "ServerB should NOT have batch/v1 in local discovery (disabled via runtime-config)")

	// Verify serverA has batch/v1 (should be enabled by default)
	groups, err := clientA.Discovery().ServerGroups()
	require.NoError(t, err)

	hasBatchV1 := false
	for _, group := range groups.Groups {
		if group.Name == "batch" {
			for _, version := range group.Versions {
				if version.Version == "v1" {
					hasBatchV1 = true
					break
				}
			}
		}
	}
	require.True(t, hasBatchV1, "ServerA should have batch/v1 (enabled by default)")

	// Verify serverB has apps/v1 (should be enabled by default)
	groups, err = clientB.Discovery().ServerGroups()
	require.NoError(t, err)

	hasAppsV1 := false
	for _, group := range groups.Groups {
		if group.Name == "apps" {
			for _, version := range group.Versions {
				if version.Version == "v1" {
					hasAppsV1 = true
					break
				}
			}
		}
	}
	require.True(t, hasAppsV1, "ServerB should have apps/v1 (enabled by default)")

	t.Log("ServerA: has batch/v1, missing apps/v1")
	t.Log("ServerB: has apps/v1, missing batch/v1")
}

func testMergedDiscoveryEndpoint(t *testing.T, ctx context.Context, clientA, clientB kubernetes.Interface) {
	testCases := []struct {
		name   string
		client kubernetes.Interface
	}{
		{"serverA", clientA},
		{"serverB", clientB},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s_merged_discovery", tc.name), func(t *testing.T) {
			testClientSet := testClientSet{kubeClientSet: tc.client}
			testCtx, cancel := context.WithTimeout(ctx, 1*time.Minute)
			defer cancel()

			// Wait for merged discovery to contain both apps and batch groups.
			err := WaitForMergedDiscoveryWithCondition(testCtx, testClientSet, func(result apidiscoveryv2.APIGroupDiscoveryList) bool {
				hasApps, hasBatch := false, false

				for _, group := range result.Items {
					if group.Name == "apps" {
						hasApps = true
					}
					if group.Name == "batch" {
						hasBatch = true
					}
				}

				return hasApps && hasBatch
			})

			if err != nil {
				t.Logf("Failed to get expected groups from %s after: %v", tc.name, err)
			}

			require.NoError(t, err, "Failed to get expected groups from %s within timeout", tc.name)
			t.Logf("Successfully validated %s merged discovery contains both apps and batch groups", tc.name)
		})
	}
}

func testServerIDsInMergedDiscovery(t *testing.T, ctx context.Context, clientA, clientB kubernetes.Interface) {
	// Test that server IDs are properly included in merged discovery when requested.
	testCases := []struct {
		name   string
		client kubernetes.Interface
	}{
		{"serverA", clientA},
		{"serverB", clientB},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testClientSet := testClientSet{kubeClientSet: tc.client}
			testCtx, cancel := context.WithTimeout(ctx, 1*time.Minute)
			defer cancel()

			// Fetch merged discovery with server IDs included.
			result, err := FetchMergedDiscoveryWithServerIDs(testCtx, testClientSet)
			require.NoError(t, err, "Failed to fetch merged discovery with server IDs from %s", tc.name)

			// Find the apps and batch groups.
			var appsGroup, batchGroup *apidiscoveryv2.APIGroupDiscovery
			for i := range result.Items {
				if result.Items[i].Name == "apps" {
					appsGroup = &result.Items[i]
				}
				if result.Items[i].Name == "batch" {
					batchGroup = &result.Items[i]
				}
			}

			require.NotNil(t, appsGroup, "Should find apps group in merged discovery")
			require.NotNil(t, batchGroup, "Should find batch group in merged discovery")

			// Check that apps group resources have server IDs.
			appsHasServerIDs := false
			for _, version := range appsGroup.Versions {
				for _, resource := range version.Resources {
					if len(resource.ServerIDs) > 0 {
						appsHasServerIDs = true
						t.Logf("Apps resource %s/%s has server IDs: %v",
							version.Version, resource.Resource, resource.ServerIDs)
					}
				}
			}

			// Check that batch group resources have server IDs.
			batchHasServerIDs := false
			for _, version := range batchGroup.Versions {
				for _, resource := range version.Resources {
					if len(resource.ServerIDs) > 0 {
						batchHasServerIDs = true
						t.Logf("Batch resource %s/%s has server IDs: %v",
							version.Version, resource.Resource, resource.ServerIDs)
					}
				}
			}

			require.True(t, appsHasServerIDs, "Apps group should have server IDs when includeServerIds=true")
			require.True(t, batchHasServerIDs, "Batch group should have server IDs when includeServerIds=true")

			t.Logf("Successfully validated that both apps and batch groups have server IDs in merged discovery from %s", tc.name)
		})
	}
}

func createProxyCertContent() (kubeapiservertesting.ProxyCA, error) {
	result := kubeapiservertesting.ProxyCA{}
	proxySigningKey, err := testutil.NewPrivateKey()
	if err != nil {
		return result, err
	}
	proxySigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "front-proxy-ca"}, proxySigningKey)
	if err != nil {
		return result, err
	}

	result = kubeapiservertesting.ProxyCA{
		ProxySigningCert: proxySigningCert,
		ProxySigningKey:  proxySigningKey,
	}
	return result, nil
}
