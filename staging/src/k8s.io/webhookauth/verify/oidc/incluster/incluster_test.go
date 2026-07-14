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

package incluster_test

import (
	"context"
	"testing"

	"k8s.io/webhookauth/verify/oidc/incluster"
)

// TestInCluster_NotInClusterErrors documents the fail-closed contract: outside a
// cluster the in-cluster REST config cannot be loaded, so InCluster returns an
// error rather than a nil verifier. The heavy lifting (local discovery, local
// JWKS, deferred audience) is exercised against oidc.NewLocalKeySetVerifier in
// the oidc package; this only pins the thin client-go wiring.
func TestInCluster_NotInClusterErrors(t *testing.T) {
	// Force "not in cluster": rest.InClusterConfig requires these env vars.
	t.Setenv("KUBERNETES_SERVICE_HOST", "")
	t.Setenv("KUBERNETES_SERVICE_PORT", "")

	if _, err := incluster.InCluster(context.Background()); err == nil {
		t.Fatal("expected InCluster to error when not running in a cluster")
	}
}
