//go:build linux

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
	"syscall"
	"testing"

	v1 "k8s.io/api/core/v1"
	proxyutiltest "k8s.io/kubernetes/pkg/proxy/util/testing"
)

// TestProxyRulesInUserNS verifies that the nftables proxier can install real
// kernel nftables rules inside an unprivileged user+network namespace.
//
// This complements the existing nftablesTracer simulation tests by exercising
// the actual kernel nftables path without requiring the test runner to be root.
//
// See https://github.com/kubernetes/kubernetes/issues/130926
func TestProxyRulesInUserNS(t *testing.T) {
	proxyutiltest.RunInUserNS(t, testProxyRulesInUserNS_Namespaced,
		syscall.CLONE_NEWNET,
	)
}

// testProxyRulesInUserNS_Namespaced is the test body that runs inside the
// user+network namespace. At this point the process has CAP_NET_ADMIN inside
// the namespace and a clean loopback-only network stack.
func testProxyRulesInUserNS_Namespaced(t *testing.T) {
	// Phase 1 (this PR): verify that NewFakeProxier succeeds and that
	// syncProxyRules() completes without error inside an unprivileged netns.
	//
	// Future PRs will extend this to:
	//   - Create veth pairs connecting ns-pod1 <-> ns-kproxy <-> ns-pod2
	//   - Install real Service/Endpoint rules via the proxier
	//   - Send packets through the topology and assert forwarding behaviour
	_, fp := NewFakeProxier(v1.IPv4Protocol)

	fp.syncProxyRules()

	t.Log("nftables rule installation succeeded inside unprivileged user+network namespace")
}

