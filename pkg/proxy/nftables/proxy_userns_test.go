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
	"context"
	"fmt"
	"net"
	"testing"
	"time"

	"golang.org/x/time/rate"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/conntrack"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/runner"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	proxyutiltest "k8s.io/kubernetes/pkg/proxy/util/testing"
	netutils "k8s.io/utils/net"
)

// TestProxyRulesInUserNS verifies that the nftables proxier can install real
// kernel nftables rules inside an unprivileged user+network namespace.
//
// Run via: make test-userns
// (which uses: unshare --user --map-root-user --net $(MAKE) test)
//
// See https://github.com/kubernetes/kubernetes/issues/130926
func TestProxyRulesInUserNS(t *testing.T) {
	nft, err := getNFTablesInterface(v1.IPv4Protocol)
	if err != nil {
		t.Skipf("nftables not available: %v", err)
	}

	podCIDR := "10.0.0.0/8"
	serviceCIDRs := "172.30.0.0/16"
	nodePortAddresses := []string{fmt.Sprintf("%s/32", testNodeIP), fmt.Sprintf("%s/128", testNodeIPv6)}
	detectLocal := proxyutil.NewDetectLocalByCIDR(podCIDR)

	networkInterfacer := proxyutiltest.NewFakeNetwork()
	itf := net.Interface{Index: 0, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0}
	addrs := []net.Addr{
		&net.IPNet{IP: netutils.ParseIPSloppy("127.0.0.1"), Mask: net.CIDRMask(8, 32)},
	}
	networkInterfacer.AddInterfaceAddr(&itf, addrs)

	fp := &Proxier{
		ipFamily:            v1.IPv4Protocol,
		svcPortMap:          make(proxy.ServicePortMap),
		serviceChanges:      proxy.NewServiceChangeTracker(v1.IPv4Protocol, newServiceInfo, nil),
		endpointsMap:        make(proxy.EndpointsMap),
		endpointsChanges:    proxy.NewEndpointsChangeTracker(v1.IPv4Protocol, testNodeName, newEndpointInfo, nil),
		needFullSync:        true,
		nftables:            nft,
		masqueradeMark:      "0x4000",
		masqueradeRule:      "mark set mark or 0x4000",
		conntrack:           conntrack.NewFake(),
		localDetector:       detectLocal,
		nodeName:            testNodeName,
		serviceHealthServer: healthcheck.NewFakeServiceHealthServer(),
		nodeIP:              netutils.ParseIPSloppy(testNodeIP),
		nodePortAddresses:   proxyutil.NewNodePortAddresses(v1.IPv4Protocol, nodePortAddresses),
		networkInterfacer:   networkInterfacer,
		staleChains:         make(map[string]time.Time),
		serviceCIDRs:        serviceCIDRs,
		logRateLimiter:      rate.NewLimiter(rate.Every(24*time.Hour), 1),
		clusterIPs:          newNFTElementStorage("set", clusterIPsSet),
		serviceIPs:          newNFTElementStorage("map", serviceIPsMap),
		firewallIPs:         newNFTElementStorage("map", firewallIPsMap),
		noEndpointServices:  newNFTElementStorage("map", noEndpointServicesMap),
		noEndpointNodePorts: newNFTElementStorage("map", noEndpointNodePortsMap),
		serviceNodePorts:    newNFTElementStorage("map", serviceNodePortsMap),
		hairpinConnections:  newNFTElementStorage("set", hairpinConnectionsSet),
	}
	fp.setInitialized(true)
	fp.syncRunner = runner.NewBoundedFrequencyRunner("test-sync-runner", fp.syncProxyRules, 0, 30*time.Second, time.Minute)

	if err := fp.syncProxyRules(); err != nil {
		t.Fatalf("syncProxyRules() failed inside unprivileged user+network namespace: %v", err)
	}

	// Verify that real kernel nftables rules were created by querying the
	// actual nftables table. This confirms we are not relying on mocks.
	ctx := context.Background()
	objects, err := nft.ListAll(ctx)
	if err != nil {
		t.Fatalf("ListAll() failed after syncProxyRules(): %v", err)
	}

	// Assert the core chains that syncProxyRules() must create are present.
	expectedChains := []string{
		filterForwardChain,
		filterInputChain,
		natPreroutingChain,
		natPostroutingChain,
	}
	chains := objects["chain"]
	chainSet := make(map[string]bool, len(chains))
	for _, c := range chains {
		chainSet[c] = true
	}
	for _, want := range expectedChains {
		if !chainSet[want] {
			t.Errorf("expected chain %q not found in kernel nftables table; got chains: %v", want, chains)
		}
	}
}
