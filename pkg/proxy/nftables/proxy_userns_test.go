//go:build usernstest

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

package nftables

import (
	"testing"

	v1 "k8s.io/api/core/v1"
)

// TestProxyRulesInUserNS verifies that the nftables proxier can install real
// kernel nftables rules inside an unprivileged user+network namespace.
//
// Run via: make test-userns
//
// See https://github.com/kubernetes/kubernetes/issues/130926
func TestProxyRulesInUserNS(t *testing.T) {
	nft, fp := NewFakeProxier(v1.IPv4Protocol)
	if err := fp.syncProxyRules(); err != nil {
		t.Fatalf("syncProxyRules() failed inside unprivileged user+network namespace: %v", err)
	}
	assertNFTablesTransactionEqual(t, getLine(), baseRules, nft.Dump())
}
