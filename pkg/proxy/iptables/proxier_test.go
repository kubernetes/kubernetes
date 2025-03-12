//go:build linux
// +build linux

/*
Copyright 2015 The Kubernetes Authors.

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

package iptables

import (
	"bytes"
	"fmt"
	"net"
	"reflect"
	"regexp"
	stdruntime "runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/lithammer/dedent"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	klogtesting "k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/conntrack"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	proxyutiltest "k8s.io/kubernetes/pkg/proxy/util/testing"
	"k8s.io/kubernetes/pkg/util/async"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	iptablestest "k8s.io/kubernetes/pkg/util/iptables/testing"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

// Conventions for tests using NewFakeProxier:
//
// Pod IPs:             10.0.0.0/8
// Service ClusterIPs:  172.30.0.0/16
// Node IPs:            192.168.0.0/24
// Local Node IP:       192.168.0.2
// Service ExternalIPs: 192.168.99.0/24
// LoadBalancer IPs:    1.2.3.4, 5.6.7.8, 9.10.11.12
// Non-cluster IPs:     203.0.113.0/24
// LB Source Range:     203.0.113.0/25

const testNodeName = "test-node"
const testNodeIP = "192.168.0.2"
const testNodeIPAlt = "192.168.1.2"
const testExternalIP = "192.168.99.11"
const testNodeIPv6 = "2001:db8::1"
const testNodeIPv6Alt = "2001:db8:1::2"
const testExternalClient = "203.0.113.2"
const testExternalClientBlocked = "203.0.113.130"

var testNodeIPs = []string{testNodeIP, testNodeIPAlt, testExternalIP, testNodeIPv6, testNodeIPv6Alt}

func NewFakeProxier(ipt utiliptables.Interface) *Proxier {
	// TODO: Call NewProxier after refactoring out the goroutine
	// invocation into a Run() method.
	ipfamily := v1.IPv4Protocol
	podCIDR := "10.0.0.0/8"
	if ipt.IsIPv6() {
		ipfamily = v1.IPv6Protocol
		podCIDR = "fd00:10::/64"
	}
	detectLocal := proxyutil.NewDetectLocalByCIDR(podCIDR)

	networkInterfacer := proxyutiltest.NewFakeNetwork()
	itf := net.Interface{Index: 0, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0}
	addrs := []net.Addr{
		&net.IPNet{IP: netutils.ParseIPSloppy("127.0.0.1"), Mask: net.CIDRMask(8, 32)},
		&net.IPNet{IP: netutils.ParseIPSloppy("::1/128"), Mask: net.CIDRMask(128, 128)},
	}
	networkInterfacer.AddInterfaceAddr(&itf, addrs)
	itf1 := net.Interface{Index: 1, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0}
	addrs1 := []net.Addr{
		&net.IPNet{IP: netutils.ParseIPSloppy(testNodeIP), Mask: net.CIDRMask(24, 32)},
		&net.IPNet{IP: netutils.ParseIPSloppy(testNodeIPAlt), Mask: net.CIDRMask(24, 32)},
		&net.IPNet{IP: netutils.ParseIPSloppy(testExternalIP), Mask: net.CIDRMask(24, 32)},
		&net.IPNet{IP: netutils.ParseIPSloppy(testNodeIPv6), Mask: net.CIDRMask(64, 128)},
		&net.IPNet{IP: netutils.ParseIPSloppy(testNodeIPv6Alt), Mask: net.CIDRMask(64, 128)},
	}
	networkInterfacer.AddInterfaceAddr(&itf1, addrs1)

	p := &Proxier{
		ipFamily:                 ipfamily,
		svcPortMap:               make(proxy.ServicePortMap),
		serviceChanges:           proxy.NewServiceChangeTracker(ipfamily, newServiceInfo, nil),
		endpointsMap:             make(proxy.EndpointsMap),
		endpointsChanges:         proxy.NewEndpointsChangeTracker(ipfamily, testNodeName, newEndpointInfo, nil),
		needFullSync:             true,
		iptables:                 ipt,
		masqueradeMark:           "0x4000",
		conntrack:                conntrack.NewFake(),
		localDetector:            detectLocal,
		nodeName:                 testNodeName,
		serviceHealthServer:      healthcheck.NewFakeServiceHealthServer(),
		precomputedProbabilities: make([]string, 0, 1001),
		iptablesData:             bytes.NewBuffer(nil),
		existingFilterChainsData: bytes.NewBuffer(nil),
		filterChains:             proxyutil.NewLineBuffer(),
		filterRules:              proxyutil.NewLineBuffer(),
		natChains:                proxyutil.NewLineBuffer(),
		natRules:                 proxyutil.NewLineBuffer(),
		nodeIP:                   netutils.ParseIPSloppy(testNodeIP),
		localhostNodePorts:       true,
		nodePortAddresses:        proxyutil.NewNodePortAddresses(ipfamily, nil),
		networkInterfacer:        networkInterfacer,
		nfAcctCounters: map[string]bool{
			metrics.IPTablesCTStateInvalidDroppedNFAcctCounter: true,
			metrics.LocalhostNodePortAcceptedNFAcctCounter:     true,
		},
	}
	p.setInitialized(true)
	p.syncRunner = async.NewBoundedFrequencyRunner("test-sync-runner", p.syncProxyRules, 0, time.Minute, 1)
	return p
}

// parseIPTablesData takes iptables-save output and returns a map of table name to array of lines.
func parseIPTablesData(ruleData string) (map[string][]string, error) {
	// Split ruleData at the "COMMIT" lines; given valid input, this will result in
	// one element for each table plus an extra empty element (since the ruleData
	// should end with a "COMMIT" line).
	rawTables := strings.Split(strings.TrimPrefix(ruleData, "\n"), "COMMIT\n")
	nTables := len(rawTables) - 1
	if nTables < 2 || rawTables[nTables] != "" {
		return nil, fmt.Errorf("bad ruleData (%d tables)\n%s", nTables, ruleData)
	}

	tables := make(map[string][]string, nTables)
	for i, table := range rawTables[:nTables] {
		lines := strings.Split(strings.Trim(table, "\n"), "\n")
		// The first line should be, eg, "*nat" or "*filter"
		if lines[0][0] != '*' {
			return nil, fmt.Errorf("bad ruleData (table %d starts with %q)", i+1, lines[0])
		}
		// add back the "COMMIT" line that got eaten by the strings.Split above
		lines = append(lines, "COMMIT")
		tables[lines[0][1:]] = lines
	}

	if tables["nat"] == nil {
		return nil, fmt.Errorf("bad ruleData (no %q table)", "nat")
	}
	if tables["filter"] == nil {
		return nil, fmt.Errorf("bad ruleData (no %q table)", "filter")
	}
	return tables, nil
}

func TestParseIPTablesData(t *testing.T) {
	for _, tc := range []struct {
		name   string
		input  string
		output map[string][]string
		error  string
	}{
		{
			name: "basic test",
			input: dedent.Dedent(`
				*filter
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				:KUBE-NODEPORTS - [0:0]
				:KUBE-POSTROUTING - [0:0]
				:KUBE-MARK-MASQ - [0:0]
				:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
				:KUBE-SEP-SXIVWICOYRO3J4NJ - [0:0]
				-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
				-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
				-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
				-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 10.20.30.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
				-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 10.20.30.41 --dport 80 ! -s 10.0.0.0/24 -j KUBE-MARK-MASQ
				-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment ns1/svc1:p80 -j KUBE-SEP-SXIVWICOYRO3J4NJ
				-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -s 10.180.0.1 -j KUBE-MARK-MASQ
				-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.1:80
				-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
				COMMIT
				`),
			output: map[string][]string{
				"filter": {
					`*filter`,
					`:KUBE-SERVICES - [0:0]`,
					`:KUBE-EXTERNAL-SERVICES - [0:0]`,
					`:KUBE-FORWARD - [0:0]`,
					`:KUBE-NODEPORTS - [0:0]`,
					`-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT`,
					`-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP`,
					`-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT`,
					`-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT`,
					`COMMIT`,
				},
				"nat": {
					`*nat`,
					`:KUBE-SERVICES - [0:0]`,
					`:KUBE-NODEPORTS - [0:0]`,
					`:KUBE-POSTROUTING - [0:0]`,
					`:KUBE-MARK-MASQ - [0:0]`,
					`:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]`,
					`:KUBE-SEP-SXIVWICOYRO3J4NJ - [0:0]`,
					`-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN`,
					`-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000`,
					`-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE`,
					`-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000`,
					`-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 10.20.30.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O`,
					`-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 10.20.30.41 --dport 80 ! -s 10.0.0.0/24 -j KUBE-MARK-MASQ`,
					`-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment ns1/svc1:p80 -j KUBE-SEP-SXIVWICOYRO3J4NJ`,
					`-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -s 10.180.0.1 -j KUBE-MARK-MASQ`,
					`-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.1:80`,
					`-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS`,
					`COMMIT`,
				},
			},
		},
		{
			name: "not enough tables",
			input: dedent.Dedent(`
				*filter
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				`),
			error: "bad ruleData (1 tables)",
		},
		{
			name: "trailing junk",
			input: dedent.Dedent(`
				*filter
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				junk
				`),
			error: "bad ruleData (2 tables)",
		},
		{
			name: "bad start line",
			input: dedent.Dedent(`
				*filter
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				`),
			error: `bad ruleData (table 2 starts with ":KUBE-SERVICES - [0:0]")`,
		},
		{
			name: "no nat",
			input: dedent.Dedent(`
				*filter
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				*mangle
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				`),
			error: `bad ruleData (no "nat" table)`,
		},
		{
			name: "no filter",
			input: dedent.Dedent(`
				*mangle
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				`),
			error: `bad ruleData (no "filter" table)`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := parseIPTablesData(tc.input)
			if err == nil {
				if tc.error != "" {
					t.Errorf("unexpectedly did not get error")
				} else {
					assert.Equal(t, tc.output, out)
				}
			} else {
				if tc.error == "" {
					t.Errorf("got unexpected error: %v", err)
				} else if !strings.HasPrefix(err.Error(), tc.error) {
					t.Errorf("got wrong error: %v (expected %q)", err, tc.error)
				}
			}
		})
	}
}

func countRules(logger klog.Logger, tableName utiliptables.Table, ruleData string) int {
	dump, err := iptablestest.ParseIPTablesDump(ruleData)
	if err != nil {
		logger.Error(err, "error parsing iptables rules")
		return -1
	}

	rules := 0
	table, err := dump.GetTable(tableName)
	if err != nil {
		logger.Error(err, "can't find table", "table", tableName)
		return -1
	}

	for _, c := range table.Chains {
		rules += len(c.Rules)
	}
	return rules
}

func countRulesFromMetric(logger klog.Logger, tableName utiliptables.Table, ipFamily string) int {
	numRulesFloat, err := testutil.GetGaugeMetricValue(metrics.IPTablesRulesTotal.WithLabelValues(string(tableName), ipFamily))
	if err != nil {
		logger.Error(err, "metrics are not registered?")
		return -1
	}
	return int(numRulesFloat)
}

func countRulesFromLastSyncMetric(logger klog.Logger, tableName utiliptables.Table, ipFamily string) int {
	numRulesFloat, err := testutil.GetGaugeMetricValue(metrics.IPTablesRulesLastSync.WithLabelValues(string(tableName), ipFamily))
	if err != nil {
		logger.Error(err, "metrics are not registered?")
		return -1
	}
	return int(numRulesFloat)
}

// findAllMatches takes an array of lines and a pattern with one parenthesized group, and
// returns a sorted array of all of the unique matches of the parenthesized group.
func findAllMatches(lines []string, pattern string) []string {
	regex := regexp.MustCompile(pattern)
	allMatches := sets.New[string]()
	for _, line := range lines {
		match := regex.FindStringSubmatch(line)
		if len(match) == 2 {
			allMatches.Insert(match[1])
		}
	}
	return sets.List(allMatches)
}

// checkIPTablesRuleJumps checks that every `-j` in the given rules jumps to a chain
// that we created and added rules to
func checkIPTablesRuleJumps(ruleData string) error {
	tables, err := parseIPTablesData(ruleData)
	if err != nil {
		return err
	}

	for tableName, lines := range tables {
		// Find all of the lines like ":KUBE-SERVICES", indicating chains that
		// iptables-restore would create when loading the data.
		createdChains := sets.New[string](findAllMatches(lines, `^:([^ ]*)`)...)
		// Find all of the lines like "-X KUBE-SERVICES ..." indicating chains
		// that we are deleting because they are no longer used, and remove
		// those chains from createdChains.
		createdChains = createdChains.Delete(findAllMatches(lines, `-X ([^ ]*)`)...)

		// Find all of the lines like "-A KUBE-SERVICES ..." indicating chains
		// that we are adding at least one rule to.
		filledChains := sets.New[string](findAllMatches(lines, `-A ([^ ]*)`)...)

		// Find all of the chains that are jumped to by some rule so we can make
		// sure we only jump to valid chains.
		jumpedChains := sets.New[string](findAllMatches(lines, `-j ([^ ]*)`)...)
		// Ignore jumps to chains that we expect to exist even if kube-proxy
		// didn't create them itself.
		jumpedChains.Delete("ACCEPT", "REJECT", "DROP", "MARK", "RETURN", "DNAT", "SNAT", "MASQUERADE")

		// Find cases where we have "-A FOO ... -j BAR" but no ":BAR", meaning
		// that we are jumping to a chain that was not created.
		missingChains := jumpedChains.Difference(createdChains)
		missingChains = missingChains.Union(filledChains.Difference(createdChains))
		if len(missingChains) > 0 {
			return fmt.Errorf("some chains in %s are used but were not created: %v", tableName, missingChains.UnsortedList())
		}

		// Find cases where we have "-A FOO ... -j BAR", but no "-A BAR ...",
		// meaning that we are jumping to a chain that we didn't write out any
		// rules for, which is normally a bug. (Except that KUBE-SERVICES always
		// jumps to KUBE-NODEPORTS, even when there are no NodePort rules.)
		emptyChains := jumpedChains.Difference(filledChains)
		emptyChains.Delete(string(kubeNodePortsChain))
		if len(emptyChains) > 0 {
			return fmt.Errorf("some chains in %s are jumped to but have no rules: %v", tableName, emptyChains.UnsortedList())
		}

		// Find cases where we have ":BAR" but no "-A FOO ... -j BAR", meaning
		// that we are creating an empty chain but not using it for anything.
		extraChains := createdChains.Difference(jumpedChains)
		extraChains.Delete(string(kubeServicesChain), string(kubeExternalServicesChain), string(kubeNodePortsChain), string(kubePostroutingChain), string(kubeForwardChain), string(kubeMarkMasqChain), string(kubeProxyFirewallChain), string(kubeletFirewallChain))
		if len(extraChains) > 0 {
			return fmt.Errorf("some chains in %s are created but not used: %v", tableName, extraChains.UnsortedList())
		}
	}

	return nil
}

func TestCheckIPTablesRuleJumps(t *testing.T) {
	for _, tc := range []struct {
		name  string
		input string
		error string
	}{
		{
			name: "valid",
			input: dedent.Dedent(`
				*filter
				COMMIT
				*nat
				:KUBE-MARK-MASQ - [0:0]
				:KUBE-SERVICES - [0:0]
				:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
				-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 10.20.30.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
				-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 10.20.30.41 --dport 80 ! -s 10.0.0.0/24 -j KUBE-MARK-MASQ
				COMMIT
				`),
			error: "",
		},
		{
			name: "can't jump to chain that wasn't created",
			input: dedent.Dedent(`
				*filter
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
				COMMIT
				`),
			error: "some chains in nat are used but were not created: [KUBE-SVC-XPGD46QRK7WJZT7O]",
		},
		{
			name: "can't jump to chain that has no rules",
			input: dedent.Dedent(`
				*filter
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
				COMMIT
				`),
			error: "some chains in nat are jumped to but have no rules: [KUBE-SVC-XPGD46QRK7WJZT7O]",
		},
		{
			name: "can't add rules to a chain that wasn't created",
			input: dedent.Dedent(`
				*filter
				COMMIT
				*nat
				:KUBE-MARK-MASQ - [0:0]
				:KUBE-SERVICES - [0:0]
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" ...
				-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
				COMMIT
				`),
			error: "some chains in nat are used but were not created: [KUBE-SVC-XPGD46QRK7WJZT7O]",
		},
		{
			name: "can't jump to chain that wasn't created",
			input: dedent.Dedent(`
				*filter
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
				COMMIT
				`),
			error: "some chains in nat are used but were not created: [KUBE-SVC-XPGD46QRK7WJZT7O]",
		},
		{
			name: "can't jump to chain that has no rules",
			input: dedent.Dedent(`
				*filter
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
				COMMIT
				`),
			error: "some chains in nat are jumped to but have no rules: [KUBE-SVC-XPGD46QRK7WJZT7O]",
		},
		{
			name: "can't add rules to a chain that wasn't created",
			input: dedent.Dedent(`
				*filter
				COMMIT
				*nat
				:KUBE-MARK-MASQ - [0:0]
				:KUBE-SERVICES - [0:0]
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" ...
				-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
				COMMIT
				`),
			error: "some chains in nat are used but were not created: [KUBE-SVC-XPGD46QRK7WJZT7O]",
		},
		{
			name: "can't create chain and then not use it",
			input: dedent.Dedent(`
				*filter
				COMMIT
				*nat
				:KUBE-MARK-MASQ - [0:0]
				:KUBE-SERVICES - [0:0]
				:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" ...
				COMMIT
				`),
			error: "some chains in nat are created but not used: [KUBE-SVC-XPGD46QRK7WJZT7O]",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := checkIPTablesRuleJumps(tc.input)
			if err == nil {
				if tc.error != "" {
					t.Errorf("unexpectedly did not get error")
				}
			} else {
				if tc.error == "" {
					t.Errorf("got unexpected error: %v", err)
				} else if !strings.HasPrefix(err.Error(), tc.error) {
					t.Errorf("got wrong error: %v (expected %q)", err, tc.error)
				}
			}
		})
	}
}

// orderByCommentServiceName is a helper function that orders two IPTables rules
// based on the service name in their comment. (If either rule has no comment then the
// return value is undefined.)
func orderByCommentServiceName(rule1, rule2 *iptablestest.Rule) bool {
	if rule1.Comment == nil || rule2.Comment == nil {
		return false
	}
	name1, name2 := rule1.Comment.Value, rule2.Comment.Value

	// The service name is the comment up to the first space or colon
	i := strings.IndexAny(name1, " :")
	if i != -1 {
		name1 = name1[:i]
	}
	i = strings.IndexAny(name2, " :")
	if i != -1 {
		name2 = name2[:i]
	}

	return name1 < name2
}

// sortIPTablesRules sorts `iptables-restore` output so as to not depend on the order that
// Services get processed in, while preserving the relative ordering of related rules.
func sortIPTablesRules(ruleData string) (string, error) {
	dump, err := iptablestest.ParseIPTablesDump(ruleData)
	if err != nil {
		return "", err
	}

	// Sort tables
	sort.Slice(dump.Tables, func(i, j int) bool {
		return dump.Tables[i].Name < dump.Tables[j].Name
	})

	// Sort chains
	for t := range dump.Tables {
		table := &dump.Tables[t]
		sort.Slice(table.Chains, func(i, j int) bool {
			switch {
			case table.Chains[i].Name == kubeNodePortsChain:
				// KUBE-NODEPORTS comes before anything
				return true
			case table.Chains[j].Name == kubeNodePortsChain:
				// anything goes after KUBE-NODEPORTS
				return false
			case table.Chains[i].Name == kubeServicesChain:
				// KUBE-SERVICES comes before anything (except KUBE-NODEPORTS)
				return true
			case table.Chains[j].Name == kubeServicesChain:
				// anything (except KUBE-NODEPORTS) goes after KUBE-SERVICES
				return false
			case strings.HasPrefix(string(table.Chains[i].Name), "KUBE-") && !strings.HasPrefix(string(table.Chains[j].Name), "KUBE-"):
				// KUBE-* comes before non-KUBE-*
				return true
			case !strings.HasPrefix(string(table.Chains[i].Name), "KUBE-") && strings.HasPrefix(string(table.Chains[j].Name), "KUBE-"):
				// non-KUBE-* goes after KUBE-*
				return false
			default:
				// We have two KUBE-* chains or two non-KUBE-* chains; either
				// way they sort alphabetically
				return table.Chains[i].Name < table.Chains[j].Name
			}
		})
	}

	// Sort KUBE-NODEPORTS chains by service name
	chain, _ := dump.GetChain(utiliptables.TableFilter, kubeNodePortsChain)
	if chain != nil {
		sort.SliceStable(chain.Rules, func(i, j int) bool {
			return orderByCommentServiceName(chain.Rules[i], chain.Rules[j])
		})
	}
	chain, _ = dump.GetChain(utiliptables.TableNAT, kubeNodePortsChain)
	if chain != nil {
		sort.SliceStable(chain.Rules, func(i, j int) bool {
			return orderByCommentServiceName(chain.Rules[i], chain.Rules[j])
		})
	}

	// Sort KUBE-SERVICES chains by service name (but keeping the "must be the last
	// rule" rule in the "nat" table's KUBE-SERVICES chain last).
	chain, _ = dump.GetChain(utiliptables.TableFilter, kubeServicesChain)
	if chain != nil {
		sort.SliceStable(chain.Rules, func(i, j int) bool {
			return orderByCommentServiceName(chain.Rules[i], chain.Rules[j])
		})
	}
	chain, _ = dump.GetChain(utiliptables.TableNAT, kubeServicesChain)
	if chain != nil {
		sort.SliceStable(chain.Rules, func(i, j int) bool {
			if chain.Rules[i].Comment != nil && strings.Contains(chain.Rules[i].Comment.Value, "must be the last rule") {
				return false
			} else if chain.Rules[j].Comment != nil && strings.Contains(chain.Rules[j].Comment.Value, "must be the last rule") {
				return true
			}
			return orderByCommentServiceName(chain.Rules[i], chain.Rules[j])
		})
	}

	return dump.String(), nil
}

func TestSortIPTablesRules(t *testing.T) {
	for _, tc := range []struct {
		name   string
		input  string
		output string
		error  string
	}{
		{
			name: "basic test using each match type",
			input: dedent.Dedent(`
				*filter
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FIREWALL - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				:KUBE-PROXY-FIREWALL - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-EXTERNAL-SERVICES -m comment --comment "ns2/svc2:p80 has no local endpoints" -m tcp -p tcp -d 192.168.99.22 --dport 80 -j DROP
				-A KUBE-EXTERNAL-SERVICES -m comment --comment "ns2/svc2:p80 has no local endpoints" -m tcp -p tcp -d 1.2.3.4 --dport 80 -j DROP
				-A KUBE-EXTERNAL-SERVICES -m comment --comment "ns2/svc2:p80 has no local endpoints" -m addrtype --dst-type LOCAL -m tcp -p tcp --dport 3001 -j DROP
				-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				-A KUBE-PROXY-FIREWALL -m comment --comment "ns5/svc5:p80 traffic not accepted by KUBE-FW-NUKIZ6OKUXPJNT4C" -m tcp -p tcp -d 5.6.7.8 --dport 80 -j DROP
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				:KUBE-NODEPORTS - [0:0]
				:KUBE-POSTROUTING - [0:0]
				:KUBE-MARK-MASQ - [0:0]
				:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
				:KUBE-SEP-SXIVWICOYRO3J4NJ - [0:0]
				:KUBE-SVC-GNZBNJ2PO5MGZ6GT - [0:0]
				:KUBE-EXT-GNZBNJ2PO5MGZ6GT - [0:0]
				:KUBE-SVL-GNZBNJ2PO5MGZ6GT - [0:0]
				:KUBE-FW-GNZBNJ2PO5MGZ6GT - [0:0]
				:KUBE-SEP-RS4RBKLTHTF2IUXJ - [0:0]
				:KUBE-SVC-X27LE4BHSL4DOUIK - [0:0]
				:KUBE-SEP-OYPFS5VJICHGATKP - [0:0]
				:KUBE-SVC-4SW47YFZTEDKD3PK - [0:0]
				:KUBE-SEP-UKSFD7AGPMPPLUHC - [0:0]
				:KUBE-SEP-C6EBXVWJJZMIWKLZ - [0:0]
				-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
				-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
				-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
				-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
				-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
				-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment ns1/svc1:p80 -j KUBE-SEP-SXIVWICOYRO3J4NJ
				-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -s 10.180.0.1 -j KUBE-MARK-MASQ
				-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.1:80
				-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 cluster IP" -m tcp -p tcp -d 172.30.0.42 --dport 80 -j KUBE-SVC-GNZBNJ2PO5MGZ6GT
				-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 external IP" -m tcp -p tcp -d 192.168.99.11 --dport 80 -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
				-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 loadbalancer IP" -m tcp -p tcp -d 1.2.3.4 --dport 80 -j KUBE-FW-GNZBNJ2PO5MGZ6GT
				-A KUBE-SVC-GNZBNJ2PO5MGZ6GT -m comment --comment "ns2/svc2:p80 cluster IP" -m tcp -p tcp -d 172.30.0.42 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
				-A KUBE-SVC-GNZBNJ2PO5MGZ6GT -m comment --comment ns2/svc2:p80 -j KUBE-SEP-RS4RBKLTHTF2IUXJ
				-A KUBE-SEP-RS4RBKLTHTF2IUXJ -m comment --comment ns2/svc2:p80 -s 10.180.0.2 -j KUBE-MARK-MASQ
				-A KUBE-SEP-RS4RBKLTHTF2IUXJ -m comment --comment ns2/svc2:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.2:80
				-A KUBE-FW-GNZBNJ2PO5MGZ6GT -m comment --comment "ns2/svc2:p80 loadbalancer IP" -s 203.0.113.0/25 -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
				-A KUBE-FW-GNZBNJ2PO5MGZ6GT -m comment --comment "other traffic to s2/svc2:p80 will be dropped by KUBE-PROXY-FIREWALL"
				-A KUBE-NODEPORTS -m comment --comment ns2/svc2:p80 -m tcp -p tcp --dport 3001 -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
				-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -m comment --comment "Redirect pods trying to reach external loadbalancer VIP to clusterIP" -s 10.0.0.0/8 -j KUBE-SVC-GNZBNJ2PO5MGZ6GT
				-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -m comment --comment "masquerade LOCAL traffic for ns2/svc2:p80 LB IP" -m addrtype --src-type LOCAL -j KUBE-MARK-MASQ
				-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -m comment --comment "route LOCAL traffic for ns2/svc2:p80 LB IP to service chain" -m addrtype --src-type LOCAL -j KUBE-SVC-GNZBNJ2PO5MGZ6GT
				-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -j KUBE-SVL-GNZBNJ2PO5MGZ6GT
				-A KUBE-SVL-GNZBNJ2PO5MGZ6GT -m comment --comment "ns2/svc2:p80 has no local endpoints" -j KUBE-MARK-DROP
				-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 -j KUBE-SVC-X27LE4BHSL4DOUIK
				-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
				-A KUBE-NODEPORTS -m comment --comment ns3/svc3:p80 -m tcp -p tcp --dport 3002 -j KUBE-SVC-X27LE4BHSL4DOUIK
				-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment ns3/svc3:p80 -m tcp -p tcp --dport 3002 -j KUBE-MARK-MASQ
				-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment ns3/svc3:p80 -j KUBE-SEP-OYPFS5VJICHGATKP
				-A KUBE-SEP-OYPFS5VJICHGATKP -m comment --comment ns3/svc3:p80 -s 10.180.0.3 -j KUBE-MARK-MASQ
				-A KUBE-SEP-OYPFS5VJICHGATKP -m comment --comment ns3/svc3:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.3:80
				-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 -j KUBE-SVC-4SW47YFZTEDKD3PK
				-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 external IP" -m tcp -p tcp -d 192.168.99.22 --dport 80 -j KUBE-SVC-4SW47YFZTEDKD3PK
				-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
				-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment "ns4/svc4:p80 external IP" -m tcp -p tcp -d 192.168.99.22 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
				-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment ns4/svc4:p80 -m statistic --mode random --probability 0.5000000000 -j KUBE-SEP-UKSFD7AGPMPPLUHC
				-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment ns4/svc4:p80 -j KUBE-SEP-C6EBXVWJJZMIWKLZ
				-A KUBE-SEP-UKSFD7AGPMPPLUHC -m comment --comment ns4/svc4:p80 -s 10.180.0.4 -j KUBE-MARK-MASQ
				-A KUBE-SEP-UKSFD7AGPMPPLUHC -m comment --comment ns4/svc4:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.4:80
				-A KUBE-SEP-C6EBXVWJJZMIWKLZ -m comment --comment ns4/svc4:p80 -s 10.180.0.5 -j KUBE-MARK-MASQ
				-A KUBE-SEP-C6EBXVWJJZMIWKLZ -m comment --comment ns4/svc4:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.5:80
				-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
				COMMIT
				`),
			output: dedent.Dedent(`
				*filter
				:KUBE-NODEPORTS - [0:0]
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FIREWALL - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-PROXY-FIREWALL - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-EXTERNAL-SERVICES -m comment --comment "ns2/svc2:p80 has no local endpoints" -m tcp -p tcp -d 192.168.99.22 --dport 80 -j DROP
				-A KUBE-EXTERNAL-SERVICES -m comment --comment "ns2/svc2:p80 has no local endpoints" -m tcp -p tcp -d 1.2.3.4 --dport 80 -j DROP
				-A KUBE-EXTERNAL-SERVICES -m comment --comment "ns2/svc2:p80 has no local endpoints" -m addrtype --dst-type LOCAL -m tcp -p tcp --dport 3001 -j DROP
				-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				-A KUBE-PROXY-FIREWALL -m comment --comment "ns5/svc5:p80 traffic not accepted by KUBE-FW-NUKIZ6OKUXPJNT4C" -m tcp -p tcp -d 5.6.7.8 --dport 80 -j DROP
				COMMIT
				*nat
				:KUBE-NODEPORTS - [0:0]
				:KUBE-SERVICES - [0:0]
				:KUBE-EXT-GNZBNJ2PO5MGZ6GT - [0:0]
				:KUBE-FW-GNZBNJ2PO5MGZ6GT - [0:0]
				:KUBE-MARK-MASQ - [0:0]
				:KUBE-POSTROUTING - [0:0]
				:KUBE-SEP-C6EBXVWJJZMIWKLZ - [0:0]
				:KUBE-SEP-OYPFS5VJICHGATKP - [0:0]
				:KUBE-SEP-RS4RBKLTHTF2IUXJ - [0:0]
				:KUBE-SEP-SXIVWICOYRO3J4NJ - [0:0]
				:KUBE-SEP-UKSFD7AGPMPPLUHC - [0:0]
				:KUBE-SVC-4SW47YFZTEDKD3PK - [0:0]
				:KUBE-SVC-GNZBNJ2PO5MGZ6GT - [0:0]
				:KUBE-SVC-X27LE4BHSL4DOUIK - [0:0]
				:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
				:KUBE-SVL-GNZBNJ2PO5MGZ6GT - [0:0]
				-A KUBE-NODEPORTS -m comment --comment ns2/svc2:p80 -m tcp -p tcp --dport 3001 -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
				-A KUBE-NODEPORTS -m comment --comment ns3/svc3:p80 -m tcp -p tcp --dport 3002 -j KUBE-SVC-X27LE4BHSL4DOUIK
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
				-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 cluster IP" -m tcp -p tcp -d 172.30.0.42 --dport 80 -j KUBE-SVC-GNZBNJ2PO5MGZ6GT
				-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 external IP" -m tcp -p tcp -d 192.168.99.11 --dport 80 -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
				-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 loadbalancer IP" -m tcp -p tcp -d 1.2.3.4 --dport 80 -j KUBE-FW-GNZBNJ2PO5MGZ6GT
				-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 -j KUBE-SVC-X27LE4BHSL4DOUIK
				-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 -j KUBE-SVC-4SW47YFZTEDKD3PK
				-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 external IP" -m tcp -p tcp -d 192.168.99.22 --dport 80 -j KUBE-SVC-4SW47YFZTEDKD3PK
				-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
				-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -m comment --comment "Redirect pods trying to reach external loadbalancer VIP to clusterIP" -s 10.0.0.0/8 -j KUBE-SVC-GNZBNJ2PO5MGZ6GT
				-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -m comment --comment "masquerade LOCAL traffic for ns2/svc2:p80 LB IP" -m addrtype --src-type LOCAL -j KUBE-MARK-MASQ
				-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -m comment --comment "route LOCAL traffic for ns2/svc2:p80 LB IP to service chain" -m addrtype --src-type LOCAL -j KUBE-SVC-GNZBNJ2PO5MGZ6GT
				-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -j KUBE-SVL-GNZBNJ2PO5MGZ6GT
				-A KUBE-FW-GNZBNJ2PO5MGZ6GT -m comment --comment "ns2/svc2:p80 loadbalancer IP" -s 203.0.113.0/25 -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
				-A KUBE-FW-GNZBNJ2PO5MGZ6GT -m comment --comment "other traffic to s2/svc2:p80 will be dropped by KUBE-PROXY-FIREWALL"
				-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
				-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
				-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
				-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
				-A KUBE-SEP-C6EBXVWJJZMIWKLZ -m comment --comment ns4/svc4:p80 -s 10.180.0.5 -j KUBE-MARK-MASQ
				-A KUBE-SEP-C6EBXVWJJZMIWKLZ -m comment --comment ns4/svc4:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.5:80
				-A KUBE-SEP-OYPFS5VJICHGATKP -m comment --comment ns3/svc3:p80 -s 10.180.0.3 -j KUBE-MARK-MASQ
				-A KUBE-SEP-OYPFS5VJICHGATKP -m comment --comment ns3/svc3:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.3:80
				-A KUBE-SEP-RS4RBKLTHTF2IUXJ -m comment --comment ns2/svc2:p80 -s 10.180.0.2 -j KUBE-MARK-MASQ
				-A KUBE-SEP-RS4RBKLTHTF2IUXJ -m comment --comment ns2/svc2:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.2:80
				-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -s 10.180.0.1 -j KUBE-MARK-MASQ
				-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.1:80
				-A KUBE-SEP-UKSFD7AGPMPPLUHC -m comment --comment ns4/svc4:p80 -s 10.180.0.4 -j KUBE-MARK-MASQ
				-A KUBE-SEP-UKSFD7AGPMPPLUHC -m comment --comment ns4/svc4:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.4:80
				-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
				-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment "ns4/svc4:p80 external IP" -m tcp -p tcp -d 192.168.99.22 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
				-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment ns4/svc4:p80 -m statistic --mode random --probability 0.5000000000 -j KUBE-SEP-UKSFD7AGPMPPLUHC
				-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment ns4/svc4:p80 -j KUBE-SEP-C6EBXVWJJZMIWKLZ
				-A KUBE-SVC-GNZBNJ2PO5MGZ6GT -m comment --comment "ns2/svc2:p80 cluster IP" -m tcp -p tcp -d 172.30.0.42 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
				-A KUBE-SVC-GNZBNJ2PO5MGZ6GT -m comment --comment ns2/svc2:p80 -j KUBE-SEP-RS4RBKLTHTF2IUXJ
				-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
				-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment ns3/svc3:p80 -m tcp -p tcp --dport 3002 -j KUBE-MARK-MASQ
				-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment ns3/svc3:p80 -j KUBE-SEP-OYPFS5VJICHGATKP
				-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
				-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment ns1/svc1:p80 -j KUBE-SEP-SXIVWICOYRO3J4NJ
				-A KUBE-SVL-GNZBNJ2PO5MGZ6GT -m comment --comment "ns2/svc2:p80 has no local endpoints" -j KUBE-MARK-DROP
				COMMIT
				`),
		},
		{
			name: "extra tables",
			input: dedent.Dedent(`
				*filter
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				*mangle
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				`),
			output: dedent.Dedent(`
				*filter
				:KUBE-NODEPORTS - [0:0]
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				*mangle
				:KUBE-NODEPORTS - [0:0]
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				*nat
				:KUBE-NODEPORTS - [0:0]
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				`),
		},
		{
			name: "correctly match same service name in different styles of comments",
			input: dedent.Dedent(`
				*filter
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 cluster IP" svc2 line 1
				-A KUBE-SERVICES -m comment --comment ns2/svc2 svc2 line 2
				-A KUBE-SERVICES -m comment --comment "ns2/svc2 blah" svc2 line 3
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" svc1 line 1
				-A KUBE-SERVICES -m comment --comment ns1/svc1 svc1 line 2
				-A KUBE-SERVICES -m comment --comment "ns1/svc1 blah" svc1 line 3
				-A KUBE-SERVICES -m comment --comment ns4/svc4 svc4 line 1
				-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 cluster IP" svc4 line 2
				-A KUBE-SERVICES -m comment --comment "ns4/svc4 blah" svc4 line 3
				-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" svc3 line 1
				-A KUBE-SERVICES -m comment --comment "ns3/svc3 blah" svc3 line 2
				-A KUBE-SERVICES -m comment --comment ns3/svc3 svc3 line 3
				COMMIT
				`),
			output: dedent.Dedent(`
				*filter
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" svc1 line 1
				-A KUBE-SERVICES -m comment --comment ns1/svc1 svc1 line 2
				-A KUBE-SERVICES -m comment --comment "ns1/svc1 blah" svc1 line 3
				-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 cluster IP" svc2 line 1
				-A KUBE-SERVICES -m comment --comment ns2/svc2 svc2 line 2
				-A KUBE-SERVICES -m comment --comment "ns2/svc2 blah" svc2 line 3
				-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" svc3 line 1
				-A KUBE-SERVICES -m comment --comment "ns3/svc3 blah" svc3 line 2
				-A KUBE-SERVICES -m comment --comment ns3/svc3 svc3 line 3
				-A KUBE-SERVICES -m comment --comment ns4/svc4 svc4 line 1
				-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 cluster IP" svc4 line 2
				-A KUBE-SERVICES -m comment --comment "ns4/svc4 blah" svc4 line 3
				COMMIT
				`),
		},
		{
			name: "unexpected junk lines are preserved",
			input: dedent.Dedent(`
				*filter
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				:KUBE-SEP-RS4RBKLTHTF2IUXJ - [0:0]
				:KUBE-AAAAA - [0:0]
				:KUBE-ZZZZZ - [0:0]
				:WHY-IS-THIS-CHAIN-HERE - [0:0]
				-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 cluster IP" svc2 line 1
				-A KUBE-SEP-RS4RBKLTHTF2IUXJ -m comment --comment ns2/svc2:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.2:80
				-A KUBE-ZZZZZ -m comment --comment "mystery chain number 1"
				-A KUBE-SERVICES -m comment --comment ns2/svc2 svc2 line 2
				-A WHY-IS-THIS-CHAIN-HERE -j ACCEPT
				-A KUBE-SERVICES -m comment --comment "ns2/svc2 blah" svc2 line 3
				-A KUBE-AAAAA -m comment --comment "mystery chain number 2"
				COMMIT
				`),
			output: dedent.Dedent(`
				*filter
				COMMIT
				*nat
				:KUBE-SERVICES - [0:0]
				:KUBE-AAAAA - [0:0]
				:KUBE-SEP-RS4RBKLTHTF2IUXJ - [0:0]
				:KUBE-ZZZZZ - [0:0]
				:WHY-IS-THIS-CHAIN-HERE - [0:0]
				-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 cluster IP" svc2 line 1
				-A KUBE-SERVICES -m comment --comment ns2/svc2 svc2 line 2
				-A KUBE-SERVICES -m comment --comment "ns2/svc2 blah" svc2 line 3
				-A KUBE-AAAAA -m comment --comment "mystery chain number 2"
				-A KUBE-SEP-RS4RBKLTHTF2IUXJ -m comment --comment ns2/svc2:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.2:80
				-A KUBE-ZZZZZ -m comment --comment "mystery chain number 1"
				-A WHY-IS-THIS-CHAIN-HERE -j ACCEPT
				COMMIT
				`),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := sortIPTablesRules(tc.input)
			if err == nil {
				if tc.error != "" {
					t.Errorf("unexpectedly did not get error")
				} else {
					assert.Equal(t, strings.TrimPrefix(tc.output, "\n"), out)
				}
			} else {
				if tc.error == "" {
					t.Errorf("got unexpected error: %v", err)
				} else if !strings.HasPrefix(err.Error(), tc.error) {
					t.Errorf("got wrong error: %v (expected %q)", err, tc.error)
				}
			}
		})
	}
}

// getLine returns the line number of the caller, if possible.  This is useful in
// tests with a large number of cases - when something goes wrong you can find
// which case more easily.
func getLine() int {
	_, _, line, ok := stdruntime.Caller(1)
	if ok {
		return line
	}
	return 0
}

// assertIPTablesRulesEqual asserts that the generated rules in result match the rules in
// expected, ignoring irrelevant ordering differences. By default this also checks the
// rules for consistency (eg, no jumps to chains that aren't defined), but that can be
// disabled by passing false for checkConsistency if you are passing a partial set of rules.
func assertIPTablesRulesEqual(t *testing.T, line int, checkConsistency bool, expected, result string) {
	expected = strings.TrimLeft(expected, " \t\n")

	result, err := sortIPTablesRules(strings.TrimLeft(result, " \t\n"))
	if err != nil {
		t.Fatalf("%s", err)
	}

	lineStr := ""
	if line != 0 {
		lineStr = fmt.Sprintf(" (from line %d)", line)
	}
	if diff := cmp.Diff(expected, result); diff != "" {
		t.Errorf("rules do not match%s:\ndiff:\n%s\nfull result:\n```\n%s```", lineStr, diff, result)
	}

	if checkConsistency {
		err = checkIPTablesRuleJumps(expected)
		if err != nil {
			t.Fatalf("%s%s", err, lineStr)
		}
	}
}

// assertIPTablesChainEqual asserts that the indicated chain in the indicated table in
// result contains exactly the rules in expected (in that order).
func assertIPTablesChainEqual(t *testing.T, line int, table utiliptables.Table, chain utiliptables.Chain, expected, result string) {
	expected = strings.TrimLeft(expected, " \t\n")

	dump, err := iptablestest.ParseIPTablesDump(strings.TrimLeft(result, " \t\n"))
	if err != nil {
		t.Fatalf("%s", err)
	}

	result = ""
	if ch, _ := dump.GetChain(table, chain); ch != nil {
		for _, rule := range ch.Rules {
			result += rule.Raw + "\n"
		}
	}

	lineStr := ""
	if line != 0 {
		lineStr = fmt.Sprintf(" (from line %d)", line)
	}
	if diff := cmp.Diff(expected, result); diff != "" {
		t.Errorf("rules do not match%s:\ndiff:\n%s\nfull result:\n```\n%s```", lineStr, diff, result)
	}
}

// addressMatches helps test whether an iptables rule such as "! -s 192.168.0.0/16" matches
// ipStr. address.Value is either an IP address ("1.2.3.4") or a CIDR string
// ("1.2.3.0/24").
func addressMatches(t *testing.T, address *iptablestest.IPTablesValue, ipStr string) bool {
	ip := netutils.ParseIPSloppy(ipStr)
	if ip == nil {
		t.Fatalf("Bad IP in test case: %s", ipStr)
	}

	var matches bool
	if strings.Contains(address.Value, "/") {
		_, cidr, err := netutils.ParseCIDRSloppy(address.Value)
		if err != nil {
			t.Errorf("Bad CIDR in kube-proxy output: %v", err)
		}
		matches = cidr.Contains(ip)
	} else {
		ip2 := netutils.ParseIPSloppy(address.Value)
		if ip2 == nil {
			t.Errorf("Bad IP/CIDR in kube-proxy output: %s", address.Value)
		}
		matches = ip.Equal(ip2)
	}
	return (!address.Negated && matches) || (address.Negated && !matches)
}

// iptablesTracer holds data used while virtually tracing a packet through a set of
// iptables rules
type iptablesTracer struct {
	ipt      *iptablestest.FakeIPTables
	localIPs sets.Set[string]
	t        *testing.T

	// matches accumulates the list of rules that were matched, for debugging purposes.
	matches []string

	// outputs accumulates the list of matched terminal rule targets (endpoint
	// IP:ports, or a special target like "REJECT") and is eventually used to generate
	// the return value of tracePacket.
	outputs []string

	// markMasq tracks whether the packet has been marked for masquerading
	markMasq bool
}

// newIPTablesTracer creates an iptablesTracer. nodeIPs are the IPs to treat as local
// node IPs (for determining whether rules with "--src-type LOCAL" or "--dst-type LOCAL"
// match).
func newIPTablesTracer(t *testing.T, ipt *iptablestest.FakeIPTables, nodeIPs []string) *iptablesTracer {
	localIPs := sets.New("127.0.0.1", "::1")
	localIPs.Insert(nodeIPs...)

	return &iptablesTracer{
		ipt:      ipt,
		localIPs: localIPs,
		t:        t,
	}
}

// ruleMatches checks if the given iptables rule matches (at least probabilistically) a
// packet with the given sourceIP, destIP, and destPort.
func (tracer *iptablesTracer) ruleMatches(rule *iptablestest.Rule, sourceIP, protocol, destIP, destPort string) bool {
	// The sub-rules within an iptables rule are ANDed together, so the rule only
	// matches if all of them match. So go through the subrules, and if any of them
	// DON'T match, then fail.

	if rule.SourceAddress != nil && !addressMatches(tracer.t, rule.SourceAddress, sourceIP) {
		return false
	}
	if rule.SourceType != nil {
		addrtype := "not-matched"
		if tracer.localIPs.Has(sourceIP) {
			addrtype = "LOCAL"
		}
		if !rule.SourceType.Matches(addrtype) {
			return false
		}
	}

	if rule.Protocol != nil && !rule.Protocol.Matches(protocol) {
		return false
	}

	if rule.DestinationAddress != nil && !addressMatches(tracer.t, rule.DestinationAddress, destIP) {
		return false
	}
	if rule.DestinationType != nil {
		addrtype := "not-matched"
		if tracer.localIPs.Has(destIP) {
			addrtype = "LOCAL"
		}
		if !rule.DestinationType.Matches(addrtype) {
			return false
		}
	}
	if rule.DestinationPort != nil && !rule.DestinationPort.Matches(destPort) {
		return false
	}

	// Any rule that checks for past state/history does not match
	if rule.AffinityCheck != nil || rule.MarkCheck != nil || rule.CTStateCheck != nil {
		return false
	}

	// Anything else is assumed to match
	return true
}

// runChain runs the given packet through the rules in the given table and chain, updating
// tracer's internal state accordingly. It returns true if it hits a terminal action.
func (tracer *iptablesTracer) runChain(table utiliptables.Table, chain utiliptables.Chain, sourceIP, protocol, destIP, destPort string) bool {
	c, _ := tracer.ipt.Dump.GetChain(table, chain)
	if c == nil {
		return false
	}

	for _, rule := range c.Rules {
		if rule.Jump == nil {
			continue
		}

		if !tracer.ruleMatches(rule, sourceIP, protocol, destIP, destPort) {
			continue
		}
		// record the matched rule for debugging purposes
		tracer.matches = append(tracer.matches, rule.Raw)

		switch rule.Jump.Value {
		case "KUBE-MARK-MASQ":
			tracer.markMasq = true
			continue

		case "ACCEPT", "REJECT", "DROP":
			// (only valid in filter)
			tracer.outputs = append(tracer.outputs, rule.Jump.Value)
			return true

		case "DNAT":
			// (only valid in nat)
			tracer.outputs = append(tracer.outputs, rule.DNATDestination.Value)
			return true

		default:
			// We got a "-j KUBE-SOMETHING", so process that chain
			terminated := tracer.runChain(table, utiliptables.Chain(rule.Jump.Value), sourceIP, protocol, destIP, destPort)

			// If the subchain hit a terminal rule AND the rule that sent us
			// to that chain was non-probabilistic, then this chain terminates
			// as well. But if we went there because of a --probability rule,
			// then we want to keep accumulating further matches against this
			// chain.
			if terminated && rule.Probability == nil {
				return true
			}
		}
	}

	return false
}

// tracePacket determines what would happen to a packet with the given sourceIP, protocol,
// destIP, and destPort, given the indicated iptables ruleData. nodeIP is the local node
// IP (for rules matching "LOCAL"). (The protocol value should be lowercase as in iptables
// rules, not uppercase as in corev1.)
//
// The return values are: an array of matched rules (for debugging), the final packet
// destinations (a comma-separated list of IPs, or one of the special targets "ACCEPT",
// "DROP", or "REJECT"), and whether the packet would be masqueraded.
func tracePacket(t *testing.T, ipt *iptablestest.FakeIPTables, sourceIP, protocol, destIP, destPort string, nodeIPs []string) ([]string, string, bool) {
	tracer := newIPTablesTracer(t, ipt, nodeIPs)

	// nat:PREROUTING goes first
	tracer.runChain(utiliptables.TableNAT, utiliptables.ChainPrerouting, sourceIP, protocol, destIP, destPort)

	// After the PREROUTING rules run, pending DNATs are processed (which would affect
	// the destination IP that later rules match against).
	if len(tracer.outputs) != 0 {
		destIP = strings.Split(tracer.outputs[0], ":")[0]
	}

	// Now the filter rules get run; exactly which ones depend on whether this is an
	// inbound, outbound, or intra-host packet, which we don't know. So we just run
	// the interesting tables manually. (Theoretically this could cause conflicts in
	// the future in which case we'd have to do something more complicated.)
	tracer.runChain(utiliptables.TableFilter, kubeServicesChain, sourceIP, protocol, destIP, destPort)
	tracer.runChain(utiliptables.TableFilter, kubeExternalServicesChain, sourceIP, protocol, destIP, destPort)
	tracer.runChain(utiliptables.TableFilter, kubeNodePortsChain, sourceIP, protocol, destIP, destPort)
	tracer.runChain(utiliptables.TableFilter, kubeProxyFirewallChain, sourceIP, protocol, destIP, destPort)

	// Finally, the nat:POSTROUTING rules run, but the only interesting thing that
	// happens there is that the masquerade mark gets turned into actual masquerading.

	return tracer.matches, strings.Join(tracer.outputs, ", "), tracer.markMasq
}

type packetFlowTest struct {
	name     string
	sourceIP string
	protocol v1.Protocol
	destIP   string
	destPort int
	output   string
	masq     bool
}

func runPacketFlowTests(t *testing.T, line int, ipt *iptablestest.FakeIPTables, nodeIPs []string, testCases []packetFlowTest) {
	lineStr := ""
	if line != 0 {
		lineStr = fmt.Sprintf(" (from line %d)", line)
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			protocol := strings.ToLower(string(tc.protocol))
			if protocol == "" {
				protocol = "tcp"
			}
			matches, output, masq := tracePacket(t, ipt, tc.sourceIP, protocol, tc.destIP, fmt.Sprintf("%d", tc.destPort), nodeIPs)
			var errors []string
			if output != tc.output {
				errors = append(errors, fmt.Sprintf("wrong output: expected %q got %q", tc.output, output))
			}
			if masq != tc.masq {
				errors = append(errors, fmt.Sprintf("wrong masq: expected %v got %v", tc.masq, masq))
			}
			if errors != nil {
				t.Errorf("Test %q of a %s packet from %s to %s:%d%s got result:\n%s\n\nBy matching:\n%s\n\n",
					tc.name, protocol, tc.sourceIP, tc.destIP, tc.destPort, lineStr, strings.Join(errors, "\n"), strings.Join(matches, "\n"))
			}
		})
	}
}

// This tests tracePacket against static data, just to make sure we match things in the
// way we expect to.
func TestTracePacket(t *testing.T) {
	rules := dedent.Dedent(`
		*filter
		:INPUT - [0:0]
		:FORWARD - [0:0]
		:OUTPUT - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A INPUT -m comment --comment kubernetes health check service ports -j KUBE-NODEPORTS
		-A INPUT -m conntrack --ctstate NEW -m comment --comment kubernetes externally-visible service portals -j KUBE-EXTERNAL-SERVICES
		-A FORWARD -m comment --comment kubernetes forwarding rules -j KUBE-FORWARD
		-A FORWARD -m conntrack --ctstate NEW -m comment --comment kubernetes service portals -j KUBE-SERVICES
		-A FORWARD -m conntrack --ctstate NEW -m comment --comment kubernetes externally-visible service portals -j KUBE-EXTERNAL-SERVICES
		-A OUTPUT -m conntrack --ctstate NEW -m comment --comment kubernetes service portals -j KUBE-SERVICES
		-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
		-A KUBE-SERVICES -m comment --comment "ns6/svc6:p80 has no endpoints" -m tcp -p tcp -d 172.30.0.46 --dport 80 -j REJECT
		-A KUBE-EXTERNAL-SERVICES -m comment --comment "ns2/svc2:p80 has no local endpoints" -m tcp -p tcp -d 192.168.99.22 --dport 80 -j DROP
		-A KUBE-EXTERNAL-SERVICES -m comment --comment "ns2/svc2:p80 has no local endpoints" -m tcp -p tcp -d 1.2.3.4 --dport 80 -j DROP
		-A KUBE-EXTERNAL-SERVICES -m comment --comment "ns2/svc2:p80 has no local endpoints" -m addrtype --dst-type LOCAL -m tcp -p tcp --dport 3001 -j DROP
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
		-A KUBE-PROXY-FIREWALL -m comment --comment "ns5/svc5:p80 traffic not accepted by KUBE-FW-NUKIZ6OKUXPJNT4C" -m tcp -p tcp -d 5.6.7.8 --dport 80 -j DROP
		COMMIT
		*nat
		:PREROUTING - [0:0]
		:INPUT - [0:0]
		:OUTPUT - [0:0]
		:POSTROUTING - [0:0]
		:KUBE-EXT-4SW47YFZTEDKD3PK - [0:0]
		:KUBE-EXT-GNZBNJ2PO5MGZ6GT - [0:0]
		:KUBE-EXT-NUKIZ6OKUXPJNT4C - [0:0]
		:KUBE-EXT-X27LE4BHSL4DOUIK - [0:0]
		:KUBE-FW-NUKIZ6OKUXPJNT4C - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-NODEPORTS - [0:0]
		:KUBE-POSTROUTING - [0:0]
		:KUBE-SEP-C6EBXVWJJZMIWKLZ - [0:0]
		:KUBE-SEP-I77PXRDZVX7PMWMN - [0:0]
		:KUBE-SEP-OYPFS5VJICHGATKP - [0:0]
		:KUBE-SEP-RS4RBKLTHTF2IUXJ - [0:0]
		:KUBE-SEP-SXIVWICOYRO3J4NJ - [0:0]
		:KUBE-SEP-UKSFD7AGPMPPLUHC - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-SVC-4SW47YFZTEDKD3PK - [0:0]
		:KUBE-SVC-GNZBNJ2PO5MGZ6GT - [0:0]
		:KUBE-SVC-NUKIZ6OKUXPJNT4C - [0:0]
		:KUBE-SVC-X27LE4BHSL4DOUIK - [0:0]
		:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
		-A PREROUTING -m comment --comment kubernetes service portals -j KUBE-SERVICES
		-A OUTPUT -m comment --comment kubernetes service portals -j KUBE-SERVICES
		-A POSTROUTING -m comment --comment kubernetes postrouting rules -j KUBE-POSTROUTING
		-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
		-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
		-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
		-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
		-A KUBE-NODEPORTS -m comment --comment ns2/svc2:p80 -m tcp -p tcp --dport 3001 -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
		-A KUBE-NODEPORTS -m comment --comment ns3/svc3:p80 -m tcp -p tcp --dport 3003 -j KUBE-EXT-X27LE4BHSL4DOUIK
		-A KUBE-NODEPORTS -m comment --comment ns5/svc5:p80 -m tcp -p tcp --dport 3002 -j KUBE-EXT-NUKIZ6OKUXPJNT4C
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 cluster IP" -m tcp -p tcp -d 172.30.0.42 --dport 80 -j KUBE-SVC-GNZBNJ2PO5MGZ6GT
		-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 external IP" -m tcp -p tcp -d 192.168.99.22 --dport 80 -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
		-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 loadbalancer IP" -m tcp -p tcp -d 1.2.3.4 --dport 80 -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
		-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 -j KUBE-SVC-X27LE4BHSL4DOUIK
		-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 -j KUBE-SVC-4SW47YFZTEDKD3PK
		-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 external IP" -m tcp -p tcp -d 192.168.99.33 --dport 80 -j KUBE-EXT-4SW47YFZTEDKD3PK
		-A KUBE-SERVICES -m comment --comment "ns5/svc5:p80 cluster IP" -m tcp -p tcp -d 172.30.0.45 --dport 80 -j KUBE-SVC-NUKIZ6OKUXPJNT4C
		-A KUBE-SERVICES -m comment --comment "ns5/svc5:p80 loadbalancer IP" -m tcp -p tcp -d 5.6.7.8 --dport 80 -j KUBE-FW-NUKIZ6OKUXPJNT4C
		-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
		-A KUBE-EXT-4SW47YFZTEDKD3PK -m comment --comment "masquerade traffic for ns4/svc4:p80 external destinations" -j KUBE-MARK-MASQ
		-A KUBE-EXT-4SW47YFZTEDKD3PK -j KUBE-SVC-4SW47YFZTEDKD3PK
		-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -m comment --comment "pod traffic for ns2/svc2:p80 external destinations" -s 10.0.0.0/8 -j KUBE-SVC-GNZBNJ2PO5MGZ6GT
		-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -m comment --comment "masquerade LOCAL traffic for ns2/svc2:p80 external destinations" -m addrtype --src-type LOCAL -j KUBE-MARK-MASQ
		-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -m comment --comment "route LOCAL traffic for ns2/svc2:p80 external destinations" -m addrtype --src-type LOCAL -j KUBE-SVC-GNZBNJ2PO5MGZ6GT
		-A KUBE-EXT-NUKIZ6OKUXPJNT4C -m comment --comment "masquerade traffic for ns5/svc5:p80 external destinations" -j KUBE-MARK-MASQ
		-A KUBE-EXT-NUKIZ6OKUXPJNT4C -j KUBE-SVC-NUKIZ6OKUXPJNT4C
		-A KUBE-EXT-X27LE4BHSL4DOUIK -m comment --comment "masquerade traffic for ns3/svc3:p80 external destinations" -j KUBE-MARK-MASQ
		-A KUBE-EXT-X27LE4BHSL4DOUIK -j KUBE-SVC-X27LE4BHSL4DOUIK
		-A KUBE-FW-NUKIZ6OKUXPJNT4C -m comment --comment "ns5/svc5:p80 loadbalancer IP" -s 203.0.113.0/25 -j KUBE-EXT-NUKIZ6OKUXPJNT4C
		-A KUBE-FW-NUKIZ6OKUXPJNT4C -m comment --comment "other traffic to ns5/svc5:p80 will be dropped by KUBE-PROXY-FIREWALL"
		-A KUBE-SEP-C6EBXVWJJZMIWKLZ -m comment --comment ns4/svc4:p80 -s 10.180.0.5 -j KUBE-MARK-MASQ
		-A KUBE-SEP-C6EBXVWJJZMIWKLZ -m comment --comment ns4/svc4:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.5:80
		-A KUBE-SEP-I77PXRDZVX7PMWMN -m comment --comment ns5/svc5:p80 -s 10.180.0.3 -j KUBE-MARK-MASQ
		-A KUBE-SEP-I77PXRDZVX7PMWMN -m comment --comment ns5/svc5:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.3:80
		-A KUBE-SEP-OYPFS5VJICHGATKP -m comment --comment ns3/svc3:p80 -s 10.180.0.3 -j KUBE-MARK-MASQ
		-A KUBE-SEP-OYPFS5VJICHGATKP -m comment --comment ns3/svc3:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.3:80
		-A KUBE-SEP-RS4RBKLTHTF2IUXJ -m comment --comment ns2/svc2:p80 -s 10.180.0.2 -j KUBE-MARK-MASQ
		-A KUBE-SEP-RS4RBKLTHTF2IUXJ -m comment --comment ns2/svc2:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.2:80
		-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -s 10.180.0.1 -j KUBE-MARK-MASQ
		-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.1:80
		-A KUBE-SEP-UKSFD7AGPMPPLUHC -m comment --comment ns4/svc4:p80 -s 10.180.0.4 -j KUBE-MARK-MASQ
		-A KUBE-SEP-UKSFD7AGPMPPLUHC -m comment --comment ns4/svc4:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.4:80
		-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment "ns4/svc4:p80 -> 10.180.0.4:80" -m statistic --mode random --probability 0.5000000000 -j KUBE-SEP-UKSFD7AGPMPPLUHC
		-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment "ns4/svc4:p80 -> 10.180.0.5:80" -j KUBE-SEP-C6EBXVWJJZMIWKLZ
		-A KUBE-SVC-GNZBNJ2PO5MGZ6GT -m comment --comment "ns2/svc2:p80 cluster IP" -m tcp -p tcp -d 172.30.0.42 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-GNZBNJ2PO5MGZ6GT -m comment --comment "ns2/svc2:p80 -> 10.180.0.2:80" -j KUBE-SEP-RS4RBKLTHTF2IUXJ
		-A KUBE-SVC-NUKIZ6OKUXPJNT4C -m comment --comment "ns5/svc5:p80 cluster IP" -m tcp -p tcp -d 172.30.0.45 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-NUKIZ6OKUXPJNT4C -m comment --comment "ns5/svc5:p80 -> 10.180.0.3:80" -j KUBE-SEP-I77PXRDZVX7PMWMN
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 -> 10.180.0.3:80" -j KUBE-SEP-OYPFS5VJICHGATKP
		-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 -> 10.180.0.1:80" -j KUBE-SEP-SXIVWICOYRO3J4NJ
		COMMIT
		`)

	ipt := iptablestest.NewFake()
	err := ipt.RestoreAll([]byte(rules), utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	if err != nil {
		t.Fatalf("Restore of test data failed: %v", err)
	}

	runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
		{
			name:     "no match",
			sourceIP: "10.0.0.2",
			destIP:   "10.0.0.3",
			destPort: 80,
			output:   "",
		},
		{
			name:     "single endpoint",
			sourceIP: "10.0.0.2",
			destIP:   "172.30.0.41",
			destPort: 80,
			output:   "10.180.0.1:80",
		},
		{
			name:     "multiple endpoints",
			sourceIP: "10.0.0.2",
			destIP:   "172.30.0.44",
			destPort: 80,
			output:   "10.180.0.4:80, 10.180.0.5:80",
		},
		{
			name:     "LOCAL, KUBE-MARK-MASQ",
			sourceIP: testNodeIP,
			destIP:   "192.168.99.22",
			destPort: 80,
			output:   "10.180.0.2:80",
			masq:     true,
		},
		{
			name:     "DROP",
			sourceIP: testExternalClient,
			destIP:   "192.168.99.22",
			destPort: 80,
			output:   "DROP",
		},
		{
			name:     "ACCEPT (NodePortHealthCheck)",
			sourceIP: testNodeIP,
			destIP:   testNodeIP,
			destPort: 30000,
			output:   "ACCEPT",
		},
		{
			name:     "REJECT",
			sourceIP: "10.0.0.2",
			destIP:   "172.30.0.46",
			destPort: 80,
			output:   "REJECT",
		},
	})
}

// TestOverallIPTablesRules creates a variety of services and verifies that the generated
// rules are exactly as expected.
func TestOverallIPTablesRules(t *testing.T) {
	logger, _ := klogtesting.NewTestContext(t)
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)
	metrics.RegisterMetrics(kubeproxyconfig.ProxyModeIPTables)

	makeServiceMap(fp,
		// create ClusterIP service
		makeTestService("ns1", "svc1", func(svc *v1.Service) {
			svc.Spec.ClusterIP = "172.30.0.41"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p80",
				Port:     80,
				Protocol: v1.ProtocolTCP,
			}}
		}),
		// create LoadBalancer service with Local traffic policy
		makeTestService("ns2", "svc2", func(svc *v1.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
			svc.Spec.ClusterIP = "172.30.0.42"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p80",
				Port:     80,
				Protocol: v1.ProtocolTCP,
				NodePort: 3001,
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: "1.2.3.4",
			}}
			svc.Spec.ExternalIPs = []string{"192.168.99.22"}
			svc.Spec.HealthCheckNodePort = 30000
		}),
		// create NodePort service
		makeTestService("ns3", "svc3", func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = "172.30.0.43"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p80",
				Port:     80,
				Protocol: v1.ProtocolTCP,
				NodePort: 3003,
			}}
		}),
		// create ExternalIP service
		makeTestService("ns4", "svc4", func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = "172.30.0.44"
			svc.Spec.ExternalIPs = []string{"192.168.99.33"}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       "p80",
				Port:       80,
				Protocol:   v1.ProtocolTCP,
				TargetPort: intstr.FromInt32(80),
			}}
		}),
		// create LoadBalancer service with Cluster traffic policy, source ranges,
		// and session affinity
		makeTestService("ns5", "svc5", func(svc *v1.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyCluster
			svc.Spec.ClusterIP = "172.30.0.45"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p80",
				Port:     80,
				Protocol: v1.ProtocolTCP,
				NodePort: 3002,
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: "5.6.7.8",
			}}
			svc.Spec.HealthCheckNodePort = 30000
			// Extra whitespace to ensure that invalid value will not result
			// in a crash, for backward compatibility.
			svc.Spec.LoadBalancerSourceRanges = []string{" 203.0.113.0/25"}

			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			svc.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
				ClientIP: &v1.ClientIPConfig{
					TimeoutSeconds: ptr.To[int32](10800),
				},
			}
		}),
		// create ClusterIP service with no endpoints
		makeTestService("ns6", "svc6", func(svc *v1.Service) {
			svc.Spec.Type = "ClusterIP"
			svc.Spec.ClusterIP = "172.30.0.46"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       "p80",
				Port:       80,
				Protocol:   v1.ProtocolTCP,
				TargetPort: intstr.FromInt32(80),
			}}
		}),
	)
	populateEndpointSlices(fp,
		// create ClusterIP service endpoints
		makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{"10.180.0.1"},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p80"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
		// create Local LoadBalancer endpoints. Note that since we aren't setting
		// its NodeName, this endpoint will be considered non-local and ignored.
		makeTestEndpointSlice("ns2", "svc2", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{"10.180.0.2"},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p80"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
		// create NodePort service endpoints
		makeTestEndpointSlice("ns3", "svc3", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{"10.180.0.3"},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p80"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
		// create ExternalIP service endpoints
		makeTestEndpointSlice("ns4", "svc4", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{"10.180.0.4"},
			}, {
				Addresses: []string{"10.180.0.5"},
				NodeName:  ptr.To(testNodeName),
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p80"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
		// create Cluster LoadBalancer endpoints
		makeTestEndpointSlice("ns5", "svc5", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{"10.180.0.3"},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p80"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	fp.syncProxyRules()

	expected := dedent.Dedent(`
		*filter
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
		-A KUBE-SERVICES -m comment --comment "ns6/svc6:p80 has no endpoints" -m tcp -p tcp -d 172.30.0.46 --dport 80 -j REJECT
		-A KUBE-EXTERNAL-SERVICES -m comment --comment "ns2/svc2:p80 has no local endpoints" -m tcp -p tcp -d 192.168.99.22 --dport 80 -j DROP
		-A KUBE-EXTERNAL-SERVICES -m comment --comment "ns2/svc2:p80 has no local endpoints" -m tcp -p tcp -d 1.2.3.4 --dport 80 -j DROP
		-A KUBE-EXTERNAL-SERVICES -m comment --comment "ns2/svc2:p80 has no local endpoints" -m addrtype --dst-type LOCAL -m tcp -p tcp --dport 3001 -j DROP
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A KUBE-FORWARD -m conntrack --ctstate INVALID -m nfacct --nfacct-name ct_state_invalid_dropped_pkts -j DROP
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
		-A KUBE-PROXY-FIREWALL -m comment --comment "ns5/svc5:p80 traffic not accepted by KUBE-FW-NUKIZ6OKUXPJNT4C" -m tcp -p tcp -d 5.6.7.8 --dport 80 -j DROP
		COMMIT
		*nat
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXT-4SW47YFZTEDKD3PK - [0:0]
		:KUBE-EXT-GNZBNJ2PO5MGZ6GT - [0:0]
		:KUBE-EXT-NUKIZ6OKUXPJNT4C - [0:0]
		:KUBE-EXT-X27LE4BHSL4DOUIK - [0:0]
		:KUBE-FW-NUKIZ6OKUXPJNT4C - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-POSTROUTING - [0:0]
		:KUBE-SEP-C6EBXVWJJZMIWKLZ - [0:0]
		:KUBE-SEP-I77PXRDZVX7PMWMN - [0:0]
		:KUBE-SEP-OYPFS5VJICHGATKP - [0:0]
		:KUBE-SEP-RS4RBKLTHTF2IUXJ - [0:0]
		:KUBE-SEP-SXIVWICOYRO3J4NJ - [0:0]
		:KUBE-SEP-UKSFD7AGPMPPLUHC - [0:0]
		:KUBE-SVC-4SW47YFZTEDKD3PK - [0:0]
		:KUBE-SVC-GNZBNJ2PO5MGZ6GT - [0:0]
		:KUBE-SVC-NUKIZ6OKUXPJNT4C - [0:0]
		:KUBE-SVC-X27LE4BHSL4DOUIK - [0:0]
		:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
		-A KUBE-NODEPORTS -m comment --comment ns2/svc2:p80 -m tcp -p tcp -d 127.0.0.0/8 --dport 3001 -m nfacct --nfacct-name localhost_nps_accepted_pkts -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
		-A KUBE-NODEPORTS -m comment --comment ns2/svc2:p80 -m tcp -p tcp --dport 3001 -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
		-A KUBE-NODEPORTS -m comment --comment ns3/svc3:p80 -m tcp -p tcp -d 127.0.0.0/8 --dport 3003 -m nfacct --nfacct-name localhost_nps_accepted_pkts -j KUBE-EXT-X27LE4BHSL4DOUIK
		-A KUBE-NODEPORTS -m comment --comment ns3/svc3:p80 -m tcp -p tcp --dport 3003 -j KUBE-EXT-X27LE4BHSL4DOUIK
		-A KUBE-NODEPORTS -m comment --comment ns5/svc5:p80 -m tcp -p tcp -d 127.0.0.0/8 --dport 3002 -m nfacct --nfacct-name localhost_nps_accepted_pkts -j KUBE-EXT-NUKIZ6OKUXPJNT4C
		-A KUBE-NODEPORTS -m comment --comment ns5/svc5:p80 -m tcp -p tcp --dport 3002 -j KUBE-EXT-NUKIZ6OKUXPJNT4C
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 cluster IP" -m tcp -p tcp -d 172.30.0.42 --dport 80 -j KUBE-SVC-GNZBNJ2PO5MGZ6GT
		-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 external IP" -m tcp -p tcp -d 192.168.99.22 --dport 80 -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
		-A KUBE-SERVICES -m comment --comment "ns2/svc2:p80 loadbalancer IP" -m tcp -p tcp -d 1.2.3.4 --dport 80 -j KUBE-EXT-GNZBNJ2PO5MGZ6GT
		-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 -j KUBE-SVC-X27LE4BHSL4DOUIK
		-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 -j KUBE-SVC-4SW47YFZTEDKD3PK
		-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 external IP" -m tcp -p tcp -d 192.168.99.33 --dport 80 -j KUBE-EXT-4SW47YFZTEDKD3PK
		-A KUBE-SERVICES -m comment --comment "ns5/svc5:p80 cluster IP" -m tcp -p tcp -d 172.30.0.45 --dport 80 -j KUBE-SVC-NUKIZ6OKUXPJNT4C
		-A KUBE-SERVICES -m comment --comment "ns5/svc5:p80 loadbalancer IP" -m tcp -p tcp -d 5.6.7.8 --dport 80 -j KUBE-FW-NUKIZ6OKUXPJNT4C
		-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
		-A KUBE-EXT-4SW47YFZTEDKD3PK -m comment --comment "masquerade traffic for ns4/svc4:p80 external destinations" -j KUBE-MARK-MASQ
		-A KUBE-EXT-4SW47YFZTEDKD3PK -j KUBE-SVC-4SW47YFZTEDKD3PK
		-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -m comment --comment "pod traffic for ns2/svc2:p80 external destinations" -s 10.0.0.0/8 -j KUBE-SVC-GNZBNJ2PO5MGZ6GT
		-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -m comment --comment "masquerade LOCAL traffic for ns2/svc2:p80 external destinations" -m addrtype --src-type LOCAL -j KUBE-MARK-MASQ
		-A KUBE-EXT-GNZBNJ2PO5MGZ6GT -m comment --comment "route LOCAL traffic for ns2/svc2:p80 external destinations" -m addrtype --src-type LOCAL -j KUBE-SVC-GNZBNJ2PO5MGZ6GT
		-A KUBE-EXT-NUKIZ6OKUXPJNT4C -m comment --comment "masquerade traffic for ns5/svc5:p80 external destinations" -j KUBE-MARK-MASQ
		-A KUBE-EXT-NUKIZ6OKUXPJNT4C -j KUBE-SVC-NUKIZ6OKUXPJNT4C
		-A KUBE-EXT-X27LE4BHSL4DOUIK -m comment --comment "masquerade traffic for ns3/svc3:p80 external destinations" -j KUBE-MARK-MASQ
		-A KUBE-EXT-X27LE4BHSL4DOUIK -j KUBE-SVC-X27LE4BHSL4DOUIK
		-A KUBE-FW-NUKIZ6OKUXPJNT4C -m comment --comment "ns5/svc5:p80 loadbalancer IP" -s 203.0.113.0/25 -j KUBE-EXT-NUKIZ6OKUXPJNT4C
		-A KUBE-FW-NUKIZ6OKUXPJNT4C -m comment --comment "other traffic to ns5/svc5:p80 will be dropped by KUBE-PROXY-FIREWALL"
		-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
		-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
		-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
		-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
		-A KUBE-SEP-C6EBXVWJJZMIWKLZ -m comment --comment ns4/svc4:p80 -s 10.180.0.5 -j KUBE-MARK-MASQ
		-A KUBE-SEP-C6EBXVWJJZMIWKLZ -m comment --comment ns4/svc4:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.5:80
		-A KUBE-SEP-I77PXRDZVX7PMWMN -m comment --comment ns5/svc5:p80 -s 10.180.0.3 -j KUBE-MARK-MASQ
		-A KUBE-SEP-I77PXRDZVX7PMWMN -m comment --comment ns5/svc5:p80 -m recent --name KUBE-SEP-I77PXRDZVX7PMWMN --set -m tcp -p tcp -j DNAT --to-destination 10.180.0.3:80
		-A KUBE-SEP-OYPFS5VJICHGATKP -m comment --comment ns3/svc3:p80 -s 10.180.0.3 -j KUBE-MARK-MASQ
		-A KUBE-SEP-OYPFS5VJICHGATKP -m comment --comment ns3/svc3:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.3:80
		-A KUBE-SEP-RS4RBKLTHTF2IUXJ -m comment --comment ns2/svc2:p80 -s 10.180.0.2 -j KUBE-MARK-MASQ
		-A KUBE-SEP-RS4RBKLTHTF2IUXJ -m comment --comment ns2/svc2:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.2:80
		-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -s 10.180.0.1 -j KUBE-MARK-MASQ
		-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.1:80
		-A KUBE-SEP-UKSFD7AGPMPPLUHC -m comment --comment ns4/svc4:p80 -s 10.180.0.4 -j KUBE-MARK-MASQ
		-A KUBE-SEP-UKSFD7AGPMPPLUHC -m comment --comment ns4/svc4:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.4:80
		-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment "ns4/svc4:p80 -> 10.180.0.4:80" -m statistic --mode random --probability 0.5000000000 -j KUBE-SEP-UKSFD7AGPMPPLUHC
		-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment "ns4/svc4:p80 -> 10.180.0.5:80" -j KUBE-SEP-C6EBXVWJJZMIWKLZ
		-A KUBE-SVC-GNZBNJ2PO5MGZ6GT -m comment --comment "ns2/svc2:p80 cluster IP" -m tcp -p tcp -d 172.30.0.42 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-GNZBNJ2PO5MGZ6GT -m comment --comment "ns2/svc2:p80 -> 10.180.0.2:80" -j KUBE-SEP-RS4RBKLTHTF2IUXJ
		-A KUBE-SVC-NUKIZ6OKUXPJNT4C -m comment --comment "ns5/svc5:p80 cluster IP" -m tcp -p tcp -d 172.30.0.45 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-NUKIZ6OKUXPJNT4C -m comment --comment "ns5/svc5:p80 -> 10.180.0.3:80" -m recent --name KUBE-SEP-I77PXRDZVX7PMWMN --rcheck --seconds 10800 --reap -j KUBE-SEP-I77PXRDZVX7PMWMN
		-A KUBE-SVC-NUKIZ6OKUXPJNT4C -m comment --comment "ns5/svc5:p80 -> 10.180.0.3:80" -j KUBE-SEP-I77PXRDZVX7PMWMN
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 -> 10.180.0.3:80" -j KUBE-SEP-OYPFS5VJICHGATKP
		-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 -> 10.180.0.1:80" -j KUBE-SEP-SXIVWICOYRO3J4NJ
		COMMIT
		`)

	assertIPTablesRulesEqual(t, getLine(), true, expected, fp.iptablesData.String())

	nNatRules := countRulesFromMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	expectedNatRules := countRules(logger, utiliptables.TableNAT, fp.iptablesData.String())

	if nNatRules != expectedNatRules {
		t.Fatalf("Wrong number of nat rules: expected %d received %d", expectedNatRules, nNatRules)
	}
}

// TestNoEndpointsReject tests that a service with no endpoints rejects connections to
// its ClusterIP, ExternalIPs, NodePort, and LoadBalancer IP.
func TestNoEndpointsReject(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)
	svcIP := "172.30.0.41"
	svcPort := 80
	svcNodePort := 3001
	svcExternalIPs := "192.168.99.11"
	svcLBIP := "1.2.3.4"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Protocol: v1.ProtocolTCP,
				Port:     int32(svcPort),
				NodePort: int32(svcNodePort),
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: svcLBIP,
			}}
		}),
	)
	fp.syncProxyRules()

	runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
		{
			name:     "pod to cluster IP with no endpoints",
			sourceIP: "10.0.0.2",
			destIP:   svcIP,
			destPort: svcPort,
			output:   "REJECT",
		},
		{
			name:     "external to external IP with no endpoints",
			sourceIP: testExternalClient,
			destIP:   svcExternalIPs,
			destPort: svcPort,
			output:   "REJECT",
		},
		{
			name:     "pod to NodePort with no endpoints",
			sourceIP: "10.0.0.2",
			destIP:   testNodeIP,
			destPort: svcNodePort,
			output:   "REJECT",
		},
		{
			name:     "external to NodePort with no endpoints",
			sourceIP: testExternalClient,
			destIP:   testNodeIP,
			destPort: svcNodePort,
			output:   "REJECT",
		},
		{
			name:     "pod to LoadBalancer IP with no endpoints",
			sourceIP: "10.0.0.2",
			destIP:   svcLBIP,
			destPort: svcPort,
			output:   "REJECT",
		},
		{
			name:     "external to LoadBalancer IP with no endpoints",
			sourceIP: testExternalClient,
			destIP:   svcLBIP,
			destPort: svcPort,
			output:   "REJECT",
		},
	})
}

// TestClusterIPGeneral tests various basic features of a ClusterIP service
func TestClusterIPGeneral(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)

	makeServiceMap(fp,
		makeTestService("ns1", "svc1", func(svc *v1.Service) {
			svc.Spec.ClusterIP = "172.30.0.41"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "http",
				Port:     80,
				Protocol: v1.ProtocolTCP,
			}}
		}),
		makeTestService("ns2", "svc2", func(svc *v1.Service) {
			svc.Spec.ClusterIP = "172.30.0.42"
			svc.Spec.Ports = []v1.ServicePort{
				{
					Name:     "http",
					Port:     80,
					Protocol: v1.ProtocolTCP,
				},
				{
					Name:       "https",
					Port:       443,
					Protocol:   v1.ProtocolTCP,
					TargetPort: intstr.FromInt32(8443),
				},
				{
					Name:     "dns-udp",
					Port:     53,
					Protocol: v1.ProtocolUDP,
				},
				{
					Name:     "dns-tcp",
					Port:     53,
					Protocol: v1.ProtocolTCP,
					// We use TargetPort on TCP but not UDP/SCTP to
					// help disambiguate the output.
					TargetPort: intstr.FromInt32(5353),
				},
				{
					Name:     "dns-sctp",
					Port:     53,
					Protocol: v1.ProtocolSCTP,
				},
			}
		}),
	)

	populateEndpointSlices(fp,
		makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{"10.180.0.1"},
				NodeName:  ptr.To(testNodeName),
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("http"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
		makeTestEndpointSlice("ns2", "svc2", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{
				{
					Addresses: []string{"10.180.0.1"},
					NodeName:  ptr.To(testNodeName),
				},
				{
					Addresses: []string{"10.180.2.1"},
					NodeName:  ptr.To("node2"),
				},
			}
			eps.Ports = []discovery.EndpointPort{
				{
					Name:     ptr.To("http"),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				},
				{
					Name:     ptr.To("https"),
					Port:     ptr.To[int32](8443),
					Protocol: ptr.To(v1.ProtocolTCP),
				},
				{
					Name:     ptr.To("dns-udp"),
					Port:     ptr.To[int32](53),
					Protocol: ptr.To(v1.ProtocolUDP),
				},
				{
					Name:     ptr.To("dns-tcp"),
					Port:     ptr.To[int32](5353),
					Protocol: ptr.To(v1.ProtocolTCP),
				},
				{
					Name:     ptr.To("dns-sctp"),
					Port:     ptr.To[int32](53),
					Protocol: ptr.To(v1.ProtocolSCTP),
				},
			}
		}),
	)

	fp.syncProxyRules()

	runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
		{
			name:     "simple clusterIP",
			sourceIP: "10.180.0.2",
			destIP:   "172.30.0.41",
			destPort: 80,
			output:   "10.180.0.1:80",
			masq:     false,
		},
		{
			name:     "hairpin to cluster IP",
			sourceIP: "10.180.0.1",
			destIP:   "172.30.0.41",
			destPort: 80,
			output:   "10.180.0.1:80",
			masq:     true,
		},
		{
			name:     "clusterIP with multiple endpoints",
			sourceIP: "10.180.0.2",
			destIP:   "172.30.0.42",
			destPort: 80,
			output:   "10.180.0.1:80, 10.180.2.1:80",
			masq:     false,
		},
		{
			name:     "clusterIP with TargetPort",
			sourceIP: "10.180.0.2",
			destIP:   "172.30.0.42",
			destPort: 443,
			output:   "10.180.0.1:8443, 10.180.2.1:8443",
			masq:     false,
		},
		{
			name:     "clusterIP with TCP, UDP, and SCTP on same port (TCP)",
			sourceIP: "10.180.0.2",
			protocol: v1.ProtocolTCP,
			destIP:   "172.30.0.42",
			destPort: 53,
			output:   "10.180.0.1:5353, 10.180.2.1:5353",
			masq:     false,
		},
		{
			name:     "clusterIP with TCP, UDP, and SCTP on same port (UDP)",
			sourceIP: "10.180.0.2",
			protocol: v1.ProtocolUDP,
			destIP:   "172.30.0.42",
			destPort: 53,
			output:   "10.180.0.1:53, 10.180.2.1:53",
			masq:     false,
		},
		{
			name:     "clusterIP with TCP, UDP, and SCTP on same port (SCTP)",
			sourceIP: "10.180.0.2",
			protocol: v1.ProtocolSCTP,
			destIP:   "172.30.0.42",
			destPort: 53,
			output:   "10.180.0.1:53, 10.180.2.1:53",
			masq:     false,
		},
		{
			name:     "TCP-only port does not match UDP traffic",
			sourceIP: "10.180.0.2",
			protocol: v1.ProtocolUDP,
			destIP:   "172.30.0.42",
			destPort: 80,
			output:   "",
		},
		{
			name:     "svc1 does not accept svc2's ports",
			sourceIP: "10.180.0.2",
			destIP:   "172.30.0.41",
			destPort: 443,
			output:   "",
		},
	})
}

func TestLoadBalancer(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)
	svcIP := "172.30.0.41"
	svcPort := 80
	svcNodePort := 3001
	svcLBIP1 := "1.2.3.4"
	svcLBIP2 := "5.6.7.8"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{
				{IP: svcLBIP1},
				{IP: svcLBIP2},
			}
			svc.Spec.LoadBalancerSourceRanges = []string{
				"192.168.0.0/24",

				// Regression test that excess whitespace gets ignored
				" 203.0.113.0/25",
			}
		}),
	)

	epIP := "10.180.0.1"
	populateEndpointSlices(fp,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIP},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	fp.syncProxyRules()

	runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
		{
			name:     "pod to cluster IP",
			sourceIP: "10.0.0.2",
			destIP:   svcIP,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d", epIP, svcPort),
			masq:     false,
		},
		{
			name:     "external to nodePort",
			sourceIP: testExternalClient,
			destIP:   testNodeIP,
			destPort: svcNodePort,
			output:   fmt.Sprintf("%s:%d", epIP, svcPort),
			masq:     true,
		},
		{
			name:     "nodePort bypasses LoadBalancerSourceRanges",
			sourceIP: testExternalClientBlocked,
			destIP:   testNodeIP,
			destPort: svcNodePort,
			output:   fmt.Sprintf("%s:%d", epIP, svcPort),
			masq:     true,
		},
		{
			name:     "accepted external to LB1",
			sourceIP: testExternalClient,
			destIP:   svcLBIP1,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d", epIP, svcPort),
			masq:     true,
		},
		{
			name:     "accepted external to LB2",
			sourceIP: testExternalClient,
			destIP:   svcLBIP2,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d", epIP, svcPort),
			masq:     true,
		},
		{
			name:     "blocked external to LB1",
			sourceIP: testExternalClientBlocked,
			destIP:   svcLBIP1,
			destPort: svcPort,
			output:   "DROP",
		},
		{
			name:     "blocked external to LB2",
			sourceIP: testExternalClientBlocked,
			destIP:   svcLBIP2,
			destPort: svcPort,
			output:   "DROP",
		},
		{
			name:     "pod to LB1 (blocked by LoadBalancerSourceRanges)",
			sourceIP: "10.0.0.2",
			destIP:   svcLBIP1,
			destPort: svcPort,
			output:   "DROP",
		},
		{
			name:     "pod to LB2 (blocked by LoadBalancerSourceRanges)",
			sourceIP: "10.0.0.2",
			destIP:   svcLBIP2,
			destPort: svcPort,
			output:   "DROP",
		},
		{
			name:     "node to LB1 (allowed by LoadBalancerSourceRanges)",
			sourceIP: testNodeIP,
			destIP:   svcLBIP1,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d", epIP, svcPort),
			masq:     true,
		},
		{
			name:     "node to LB2 (allowed by LoadBalancerSourceRanges)",
			sourceIP: testNodeIP,
			destIP:   svcLBIP2,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d", epIP, svcPort),
			masq:     true,
		},

		// The LB rules assume that when you connect from a node to a LB IP, that
		// something external to kube-proxy will cause the connection to be
		// SNATted to the LB IP, so if the LoadBalancerSourceRanges include the
		// node IP, then we add a rule allowing traffic from the LB IP as well...
		{
			name:     "same node to LB1, SNATted to LB1 (implicitly allowed)",
			sourceIP: svcLBIP1,
			destIP:   svcLBIP1,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d", epIP, svcPort),
			masq:     true,
		},
		{
			name:     "same node to LB2, SNATted to LB2 (implicitly allowed)",
			sourceIP: svcLBIP2,
			destIP:   svcLBIP2,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d", epIP, svcPort),
			masq:     true,
		},
	})
}

// TestNodePorts tests NodePort services under various combinations of the
// --nodeport-addresses and --localhost-nodeports flags.
func TestNodePorts(t *testing.T) {
	testCases := []struct {
		name string

		family             v1.IPFamily
		localhostNodePorts bool
		nodePortAddresses  []string

		// allowAltNodeIP is true if we expect NodePort traffic on the alternate
		// node IP to be accepted
		allowAltNodeIP bool

		// expectFirewall is true if we expect KUBE-FIREWALL to be filled in with
		// an anti-martian-packet rule
		expectFirewall bool
	}{
		{
			name: "ipv4, localhost-nodeports enabled",

			family:             v1.IPv4Protocol,
			localhostNodePorts: true,
			nodePortAddresses:  nil,

			allowAltNodeIP: true,
			expectFirewall: true,
		},
		{
			name: "ipv4, localhost-nodeports disabled",

			family:             v1.IPv4Protocol,
			localhostNodePorts: false,
			nodePortAddresses:  nil,

			allowAltNodeIP: true,
			expectFirewall: false,
		},
		{
			name: "ipv4, localhost-nodeports disabled, localhost in nodeport-addresses",

			family:             v1.IPv4Protocol,
			localhostNodePorts: false,
			nodePortAddresses:  []string{"192.168.0.0/24", "127.0.0.1/32"},

			allowAltNodeIP: false,
			expectFirewall: false,
		},
		{
			name: "ipv4, localhost-nodeports enabled, multiple nodeport-addresses",

			family:             v1.IPv4Protocol,
			localhostNodePorts: false,
			nodePortAddresses:  []string{"192.168.0.0/24", "192.168.1.0/24", "2001:db8::/64"},

			allowAltNodeIP: true,
			expectFirewall: false,
		},
		{
			name: "ipv6, localhost-nodeports enabled",

			family:             v1.IPv6Protocol,
			localhostNodePorts: true,
			nodePortAddresses:  nil,

			allowAltNodeIP: true,
			expectFirewall: false,
		},
		{
			name: "ipv6, localhost-nodeports disabled",

			family:             v1.IPv6Protocol,
			localhostNodePorts: false,
			nodePortAddresses:  nil,

			allowAltNodeIP: true,
			expectFirewall: false,
		},
		{
			name: "ipv6, localhost-nodeports disabled, multiple nodeport-addresses",

			family:             v1.IPv6Protocol,
			localhostNodePorts: false,
			nodePortAddresses:  []string{"192.168.0.0/24", "192.168.1.0/24", "2001:db8::/64"},

			allowAltNodeIP: false,
			expectFirewall: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var ipt *iptablestest.FakeIPTables
			var svcIP, epIP1, epIP2 string
			if tc.family == v1.IPv4Protocol {
				ipt = iptablestest.NewFake()
				svcIP = "172.30.0.41"
				epIP1 = "10.180.0.1"
				epIP2 = "10.180.2.1"
			} else {
				ipt = iptablestest.NewIPv6Fake()
				svcIP = "fd00:172:30::41"
				epIP1 = "fd00:10:180::1"
				epIP2 = "fd00:10:180::2:1"
			}
			fp := NewFakeProxier(ipt)
			fp.localhostNodePorts = tc.localhostNodePorts
			if tc.nodePortAddresses != nil {
				fp.nodePortAddresses = proxyutil.NewNodePortAddresses(tc.family, tc.nodePortAddresses)
			}

			makeServiceMap(fp,
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.Type = v1.ServiceTypeNodePort
					svc.Spec.ClusterIP = svcIP
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     80,
						Protocol: v1.ProtocolTCP,
						NodePort: 3001,
					}}
				}),
			)

			populateEndpointSlices(fp,
				makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
					if tc.family == v1.IPv4Protocol {
						eps.AddressType = discovery.AddressTypeIPv4
					} else {
						eps.AddressType = discovery.AddressTypeIPv6
					}
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{epIP1},
						NodeName:  nil,
					}, {
						Addresses: []string{epIP2},
						NodeName:  ptr.To(testNodeName),
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     ptr.To("p80"),
						Port:     ptr.To[int32](80),
						Protocol: ptr.To(v1.ProtocolTCP),
					}}
				}),
			)

			fp.syncProxyRules()

			var podIP, externalClientIP, nodeIP, altNodeIP, localhostIP string
			if tc.family == v1.IPv4Protocol {
				podIP = "10.0.0.2"
				externalClientIP = testExternalClient
				nodeIP = testNodeIP
				altNodeIP = testNodeIPAlt
				localhostIP = "127.0.0.1"
			} else {
				podIP = "fd00:10::2"
				externalClientIP = "2600:5200::1"
				nodeIP = testNodeIPv6
				altNodeIP = testNodeIPv6Alt
				localhostIP = "::1"
			}
			output := net.JoinHostPort(epIP1, "80") + ", " + net.JoinHostPort(epIP2, "80")

			// Basic tests are the same for all cases
			runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
				{
					name:     "pod to cluster IP",
					sourceIP: podIP,
					destIP:   svcIP,
					destPort: 80,
					output:   output,
					masq:     false,
				},
				{
					name:     "external to nodePort",
					sourceIP: externalClientIP,
					destIP:   nodeIP,
					destPort: 3001,
					output:   output,
					masq:     true,
				},
				{
					name:     "node to nodePort",
					sourceIP: nodeIP,
					destIP:   nodeIP,
					destPort: 3001,
					output:   output,
					masq:     true,
				},
			})

			// localhost to NodePort is only allowed in IPv4, and only if not disabled
			if tc.family == v1.IPv4Protocol && tc.localhostNodePorts {
				runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
					{
						name:     "localhost to nodePort gets masqueraded",
						sourceIP: localhostIP,
						destIP:   localhostIP,
						destPort: 3001,
						output:   output,
						masq:     true,
					},
				})
			} else {
				runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
					{
						name:     "localhost to nodePort is ignored",
						sourceIP: localhostIP,
						destIP:   localhostIP,
						destPort: 3001,
						output:   "",
					},
				})
			}

			// NodePort on altNodeIP should be allowed, unless
			// nodePortAddressess excludes altNodeIP
			if tc.allowAltNodeIP {
				runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
					{
						name:     "external to nodePort on secondary IP",
						sourceIP: externalClientIP,
						destIP:   altNodeIP,
						destPort: 3001,
						output:   output,
						masq:     true,
					},
				})
			} else {
				runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
					{
						name:     "secondary nodeIP ignores NodePorts",
						sourceIP: externalClientIP,
						destIP:   altNodeIP,
						destPort: 3001,
						output:   "",
					},
				})
			}

			// We have to check the firewall rule manually rather than via
			// runPacketFlowTests(), because the packet tracer doesn't
			// implement conntrack states.
			var expected string
			if tc.expectFirewall {
				expected = "-A KUBE-FIREWALL -m comment --comment \"block incoming localnet connections\" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP\n"
			}
			assertIPTablesChainEqual(t, getLine(), utiliptables.TableFilter, kubeletFirewallChain, expected, fp.iptablesData.String())
		})
	}
}

func TestHealthCheckNodePort(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)
	fp.nodePortAddresses = proxyutil.NewNodePortAddresses(v1.IPv4Protocol, []string{"127.0.0.0/8"})

	svcIP := "172.30.0.42"
	svcPort := 80
	svcNodePort := 3001
	svcHealthCheckNodePort := 30000
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	svc := makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
		svc.Spec.Type = "LoadBalancer"
		svc.Spec.ClusterIP = svcIP
		svc.Spec.Ports = []v1.ServicePort{{
			Name:     svcPortName.Port,
			Port:     int32(svcPort),
			Protocol: v1.ProtocolTCP,
			NodePort: int32(svcNodePort),
		}}
		svc.Spec.HealthCheckNodePort = int32(svcHealthCheckNodePort)
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
	})
	makeServiceMap(fp, svc)
	fp.syncProxyRules()

	runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
		{
			name:     "firewall accepts HealthCheckNodePort",
			sourceIP: "1.2.3.4",
			destIP:   testNodeIP,
			destPort: svcHealthCheckNodePort,
			output:   "ACCEPT",
			masq:     false,
		},
	})

	fp.OnServiceDelete(svc)
	fp.syncProxyRules()

	runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
		{
			name:     "HealthCheckNodePort no longer has any rule",
			sourceIP: "1.2.3.4",
			destIP:   testNodeIP,
			destPort: svcHealthCheckNodePort,
			output:   "",
		},
	})
}

func TestDropInvalidRule(t *testing.T) {
	kubeForwardChainRules := dedent.Dedent(`
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
	`)

	testCases := []struct {
		nfacctEnsured  bool
		tcpLiberal     bool
		dropRule       string
		nfAcctCounters map[string]bool
	}{
		{
			nfacctEnsured:  false,
			tcpLiberal:     false,
			nfAcctCounters: map[string]bool{},
			dropRule:       "-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP",
		},
		{
			nfacctEnsured: true,
			tcpLiberal:    false,
			nfAcctCounters: map[string]bool{
				metrics.IPTablesCTStateInvalidDroppedNFAcctCounter: true,
			},
			dropRule: fmt.Sprintf("-A KUBE-FORWARD -m conntrack --ctstate INVALID -m nfacct --nfacct-name %s -j DROP", metrics.IPTablesCTStateInvalidDroppedNFAcctCounter),
		},
		{
			nfacctEnsured:  false,
			tcpLiberal:     true,
			nfAcctCounters: map[string]bool{},
			dropRule:       "",
		},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("tcpLiberal is %t and nfacctEnsured is %t", tc.tcpLiberal, tc.nfacctEnsured), func(t *testing.T) {
			ipt := iptablestest.NewFake()
			fp := NewFakeProxier(ipt)
			fp.conntrackTCPLiberal = tc.tcpLiberal
			fp.nfAcctCounters = tc.nfAcctCounters
			fp.syncProxyRules()

			expected := tc.dropRule + kubeForwardChainRules
			assertIPTablesChainEqual(t, getLine(), utiliptables.TableFilter, kubeForwardChain, expected, fp.iptablesData.String())
		})
	}
}

func TestMasqueradeRule(t *testing.T) {
	for _, randomFully := range []bool{false, true} {
		t.Run(fmt.Sprintf("randomFully %t", randomFully), func(t *testing.T) {
			ipt := iptablestest.NewFake().SetHasRandomFully(randomFully)
			fp := NewFakeProxier(ipt)
			fp.syncProxyRules()

			expectedFmt := dedent.Dedent(`
				-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
				-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
				-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE%s
				`)
			var expected string
			if randomFully {
				expected = fmt.Sprintf(expectedFmt, " --random-fully")
			} else {
				expected = fmt.Sprintf(expectedFmt, "")
			}
			assertIPTablesChainEqual(t, getLine(), utiliptables.TableNAT, kubePostroutingChain, expected, fp.iptablesData.String())
		})
	}
}

// TestExternalTrafficPolicyLocal tests that traffic to externally-facing IPs does not get
// masqueraded when using Local traffic policy. For traffic from external sources, that
// means it can also only be routed to local endpoints, but for traffic from internal
// sources, it gets routed to all endpoints.
func TestExternalTrafficPolicyLocal(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)

	svcIP := "172.30.0.41"
	svcPort := 80
	svcNodePort := 3001
	svcHealthCheckNodePort := 30000
	svcExternalIPs := "192.168.99.11"
	svcLBIP := "1.2.3.4"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(svcPort),
				Protocol:   v1.ProtocolTCP,
				NodePort:   int32(svcNodePort),
				TargetPort: intstr.FromInt32(int32(svcPort)),
			}}
			svc.Spec.HealthCheckNodePort = int32(svcHealthCheckNodePort)
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: svcLBIP,
			}}
		}),
	)

	epIP1 := "10.180.0.1"
	epIP2 := "10.180.2.1"
	populateEndpointSlices(fp,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIP1},
			}, {
				Addresses: []string{epIP2},
				NodeName:  ptr.To(testNodeName),
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	fp.syncProxyRules()

	runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
		{
			name:     "pod to cluster IP hits both endpoints, unmasqueraded",
			sourceIP: "10.0.0.2",
			destIP:   svcIP,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d, %s:%d", epIP1, svcPort, epIP2, svcPort),
			masq:     false,
		},
		{
			name:     "pod to external IP hits both endpoints, unmasqueraded",
			sourceIP: "10.0.0.2",
			destIP:   svcExternalIPs,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d, %s:%d", epIP1, svcPort, epIP2, svcPort),
			masq:     false,
		},
		{
			name:     "external to external IP hits only local endpoint, unmasqueraded",
			sourceIP: testExternalClient,
			destIP:   svcExternalIPs,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d", epIP2, svcPort),
			masq:     false,
		},
		{
			name:     "pod to LB IP hits only both endpoints, unmasqueraded",
			sourceIP: "10.0.0.2",
			destIP:   svcLBIP,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d, %s:%d", epIP1, svcPort, epIP2, svcPort),
			masq:     false,
		},
		{
			name:     "external to LB IP hits only local endpoint, unmasqueraded",
			sourceIP: testExternalClient,
			destIP:   svcLBIP,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d", epIP2, svcPort),
			masq:     false,
		},
		{
			name:     "pod to NodePort hits both endpoints, unmasqueraded",
			sourceIP: "10.0.0.2",
			destIP:   testNodeIP,
			destPort: svcNodePort,
			output:   fmt.Sprintf("%s:%d, %s:%d", epIP1, svcPort, epIP2, svcPort),
			masq:     false,
		},
		{
			name:     "external to NodePort hits only local endpoint, unmasqueraded",
			sourceIP: testExternalClient,
			destIP:   testNodeIP,
			destPort: svcNodePort,
			output:   fmt.Sprintf("%s:%d", epIP2, svcPort),
			masq:     false,
		},
	})
}

// TestExternalTrafficPolicyCluster tests that traffic to an externally-facing IP gets
// masqueraded when using Cluster traffic policy.
func TestExternalTrafficPolicyCluster(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)

	svcIP := "172.30.0.41"
	svcPort := 80
	svcNodePort := 3001
	svcExternalIPs := "192.168.99.11"
	svcLBIP := "1.2.3.4"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(svcPort),
				Protocol:   v1.ProtocolTCP,
				NodePort:   int32(svcNodePort),
				TargetPort: intstr.FromInt32(int32(svcPort)),
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: svcLBIP,
			}}
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyCluster
		}),
	)

	epIP1 := "10.180.0.1"
	epIP2 := "10.180.2.1"
	populateEndpointSlices(fp,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIP1},
				NodeName:  nil,
			}, {
				Addresses: []string{epIP2},
				NodeName:  ptr.To(testNodeName),
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	fp.syncProxyRules()

	runPacketFlowTests(t, getLine(), ipt, testNodeIPs, []packetFlowTest{
		{
			name:     "pod to cluster IP hits both endpoints, unmasqueraded",
			sourceIP: "10.0.0.2",
			destIP:   svcIP,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d, %s:%d", epIP1, svcPort, epIP2, svcPort),
			masq:     false,
		},
		{
			name:     "pod to external IP hits both endpoints, masqueraded",
			sourceIP: "10.0.0.2",
			destIP:   svcExternalIPs,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d, %s:%d", epIP1, svcPort, epIP2, svcPort),
			masq:     true,
		},
		{
			name:     "external to external IP hits both endpoints, masqueraded",
			sourceIP: testExternalClient,
			destIP:   svcExternalIPs,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d, %s:%d", epIP1, svcPort, epIP2, svcPort),
			masq:     true,
		},
		{
			name:     "pod to LB IP hits both endpoints, masqueraded",
			sourceIP: "10.0.0.2",
			destIP:   svcLBIP,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d, %s:%d", epIP1, svcPort, epIP2, svcPort),
			masq:     true,
		},
		{
			name:     "external to LB IP hits both endpoints, masqueraded",
			sourceIP: testExternalClient,
			destIP:   svcLBIP,
			destPort: svcPort,
			output:   fmt.Sprintf("%s:%d, %s:%d", epIP1, svcPort, epIP2, svcPort),
			masq:     true,
		},
		{
			name:     "pod to NodePort hits both endpoints, masqueraded",
			sourceIP: "10.0.0.2",
			destIP:   testNodeIP,
			destPort: svcNodePort,
			output:   fmt.Sprintf("%s:%d, %s:%d", epIP1, svcPort, epIP2, svcPort),
			masq:     true,
		},
		{
			name:     "external to NodePort hits both endpoints, masqueraded",
			sourceIP: testExternalClient,
			destIP:   testNodeIP,
			destPort: svcNodePort,
			output:   fmt.Sprintf("%s:%d, %s:%d", epIP1, svcPort, epIP2, svcPort),
			masq:     true,
		},
	})
}

func TestComputeProbability(t *testing.T) {
	expectedProbabilities := map[int]string{
		1:      "1.0000000000",
		2:      "0.5000000000",
		10:     "0.1000000000",
		100:    "0.0100000000",
		1000:   "0.0010000000",
		10000:  "0.0001000000",
		100000: "0.0000100000",
		100001: "0.0000099999",
	}

	for num, expected := range expectedProbabilities {
		actual := computeProbability(num)
		if actual != expected {
			t.Errorf("Expected computeProbability(%d) to be %s, got: %s", num, expected, actual)
		}
	}

	prevProbability := float64(0)
	for i := 100000; i > 1; i-- {
		currProbability, err := strconv.ParseFloat(computeProbability(i), 64)
		if err != nil {
			t.Fatalf("Error parsing float probability for %d: %v", i, err)
		}
		if currProbability <= prevProbability {
			t.Fatalf("Probability unexpectedly <= to previous probability for %d: (%0.10f <= %0.10f)", i, currProbability, prevProbability)
		}
		prevProbability = currProbability
	}
}

func makeTestService(namespace, name string, svcFunc func(*v1.Service)) *v1.Service {
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Annotations: map[string]string{},
		},
		Spec:   v1.ServiceSpec{},
		Status: v1.ServiceStatus{},
	}
	svcFunc(svc)
	return svc
}

func addTestPort(array []v1.ServicePort, name string, protocol v1.Protocol, port, nodeport int32, targetPort int) []v1.ServicePort {
	svcPort := v1.ServicePort{
		Name:       name,
		Protocol:   protocol,
		Port:       port,
		NodePort:   nodeport,
		TargetPort: intstr.FromInt32(int32(targetPort)),
	}
	return append(array, svcPort)
}

func TestBuildServiceMapAddRemove(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)

	services := []*v1.Service{
		makeTestService("somewhere-else", "cluster-ip", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "172.30.55.4"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "something", "UDP", 1234, 4321, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "UDP", 1235, 5321, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "sctpport", "SCTP", 1236, 6321, 0)
		}),
		makeTestService("somewhere-else", "node-port", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.ClusterIP = "172.30.55.10"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "blahblah", "UDP", 345, 678, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "moreblahblah", "TCP", 344, 677, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "muchmoreblah", "SCTP", 343, 676, 0)
		}),
		makeTestService("somewhere", "load-balancer", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = "172.30.55.11"
			svc.Spec.LoadBalancerIP = "1.2.3.4"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "foobar", "UDP", 8675, 30061, 7000)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "baz", "UDP", 8676, 30062, 7001)
			svc.Status.LoadBalancer = v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{
					{IP: "1.2.3.4"},
				},
			}
		}),
		makeTestService("somewhere", "only-local-load-balancer", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = "172.30.55.12"
			svc.Spec.LoadBalancerIP = "5.6.7.8"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "foobar2", "UDP", 8677, 30063, 7002)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "baz", "UDP", 8678, 30064, 7003)
			svc.Status.LoadBalancer = v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{
					{IP: "5.6.7.8"},
				},
			}
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
			svc.Spec.HealthCheckNodePort = 345
		}),
	}

	for i := range services {
		fp.OnServiceAdd(services[i])
	}
	fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 10 {
		t.Errorf("expected service map length 10, got %v", fp.svcPortMap)
	}

	// The only-local-loadbalancer ones get added
	healthCheckNodePorts := fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 1 {
		t.Errorf("expected 1 healthcheck port, got %v", healthCheckNodePorts)
	} else {
		nsn := makeNSN("somewhere", "only-local-load-balancer")
		if port, found := healthCheckNodePorts[nsn]; !found || port != 345 {
			t.Errorf("expected healthcheck port [%q]=345: got %v", nsn, healthCheckNodePorts)
		}
	}

	// Remove some stuff
	// oneService is a modification of services[0] with removed first port.
	oneService := makeTestService("somewhere-else", "cluster-ip", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.30.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "UDP", 1235, 5321, 0)
	})

	fp.OnServiceUpdate(services[0], oneService)
	fp.OnServiceDelete(services[1])
	fp.OnServiceDelete(services[2])
	fp.OnServiceDelete(services[3])

	fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 1 {
		t.Errorf("expected service map length 1, got %v", fp.svcPortMap)
	}

	healthCheckNodePorts = fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 0 {
		t.Errorf("expected 0 healthcheck ports, got %v", healthCheckNodePorts)
	}
}

func TestBuildServiceMapServiceHeadless(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)

	makeServiceMap(fp,
		makeTestService("somewhere-else", "headless", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = v1.ClusterIPNone
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "rpc", "UDP", 1234, 0, 0)
		}),
		makeTestService("somewhere-else", "headless-without-port", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = v1.ClusterIPNone
		}),
	)

	// Headless service should be ignored
	fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 0 {
		t.Errorf("expected service map length 0, got %d", len(fp.svcPortMap))
	}

	// No proxied services, so no healthchecks
	healthCheckNodePorts := fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %d", len(healthCheckNodePorts))
	}
}

func TestBuildServiceMapServiceTypeExternalName(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)

	makeServiceMap(fp,
		makeTestService("somewhere-else", "external-name", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeExternalName
			svc.Spec.ClusterIP = "172.30.55.4" // Should be ignored
			svc.Spec.ExternalName = "foo2.bar.com"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "blah", "UDP", 1235, 5321, 0)
		}),
	)

	fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 0 {
		t.Errorf("expected service map length 0, got %v", fp.svcPortMap)
	}
	// No proxied services, so no healthchecks
	healthCheckNodePorts := fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", healthCheckNodePorts)
	}
}

func TestBuildServiceMapServiceUpdate(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)

	servicev1 := makeTestService("somewhere", "some-service", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.30.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "something", "UDP", 1234, 4321, 0)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "TCP", 1235, 5321, 0)
	})
	servicev2 := makeTestService("somewhere", "some-service", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ClusterIP = "172.30.55.4"
		svc.Spec.LoadBalancerIP = "1.2.3.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "something", "UDP", 1234, 4321, 7002)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "TCP", 1235, 5321, 7003)
		svc.Status.LoadBalancer = v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{
				{IP: "1.2.3.4"},
			},
		}
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
		svc.Spec.HealthCheckNodePort = 345
	})

	fp.OnServiceAdd(servicev1)

	fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.svcPortMap)
	}
	healthCheckNodePorts := fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", healthCheckNodePorts)
	}

	// Change service to load-balancer
	fp.OnServiceUpdate(servicev1, servicev2)
	fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.svcPortMap)
	}
	healthCheckNodePorts = fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", healthCheckNodePorts)
	}

	// No change; make sure the service map stays the same and there are
	// no health-check changes
	fp.OnServiceUpdate(servicev2, servicev2)
	fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.svcPortMap)
	}
	healthCheckNodePorts = fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", healthCheckNodePorts)
	}

	// And back to ClusterIP
	fp.OnServiceUpdate(servicev2, servicev1)
	fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.svcPortMap)
	}
	healthCheckNodePorts = fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", healthCheckNodePorts)
	}
}

func populateEndpointSlices(proxier *Proxier, allEndpointSlices ...*discovery.EndpointSlice) {
	for i := range allEndpointSlices {
		proxier.OnEndpointSliceAdd(allEndpointSlices[i])
	}
}

func makeTestEndpointSlice(namespace, name string, sliceNum int, epsFunc func(*discovery.EndpointSlice)) *discovery.EndpointSlice {
	eps := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%d", name, sliceNum),
			Namespace: namespace,
			Labels:    map[string]string{discovery.LabelServiceName: name},
		},
	}
	epsFunc(eps)
	return eps
}

func makeNSN(namespace, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: namespace, Name: name}
}

func makeServicePortName(ns, name, port string, protocol v1.Protocol) proxy.ServicePortName {
	return proxy.ServicePortName{
		NamespacedName: makeNSN(ns, name),
		Port:           port,
		Protocol:       protocol,
	}
}

func makeServiceMap(proxier *Proxier, allServices ...*v1.Service) {
	for i := range allServices {
		proxier.OnServiceAdd(allServices[i])
	}

	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.servicesSynced = true
}

type endpointExpectation struct {
	endpoint string
	isLocal  bool
}

func checkEndpointExpectations(t *testing.T, tci int, newMap proxy.EndpointsMap, expected map[proxy.ServicePortName][]endpointExpectation) {
	if len(newMap) != len(expected) {
		t.Errorf("[%d] expected %d results, got %d: %v", tci, len(expected), len(newMap), newMap)
	}
	for x := range expected {
		if len(newMap[x]) != len(expected[x]) {
			t.Errorf("[%d] expected %d endpoints for %v, got %d", tci, len(expected[x]), x, len(newMap[x]))
		} else {
			for i := range expected[x] {
				newEp := newMap[x][i]
				if newEp.String() != expected[x][i].endpoint ||
					newEp.IsLocal() != expected[x][i].isLocal {
					t.Errorf("[%d] expected new[%v][%d] to be %v, got %v", tci, x, i, expected[x][i], newEp)
				}
			}
		}
	}
}

func TestUpdateEndpointsMap(t *testing.T) {
	emptyEndpointSlices := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, func(*discovery.EndpointSlice) {}),
	}
	subset1 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.1"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p11"),
			Port:     ptr.To[int32](11),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	subset2 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.2"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p12"),
			Port:     ptr.To[int32](12),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	namedPortLocal := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1,
			func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{"10.1.1.1"},
					NodeName:  ptr.To(testNodeName),
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     ptr.To("p11"),
					Port:     ptr.To[int32](11),
					Protocol: ptr.To(v1.ProtocolUDP),
				}}
			}),
	}
	namedPort := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subset1),
	}
	namedPortRenamed := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1,
			func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{"10.1.1.1"},
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     ptr.To("p11-2"),
					Port:     ptr.To[int32](11),
					Protocol: ptr.To(v1.ProtocolUDP),
				}}
			}),
	}
	namedPortRenumbered := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1,
			func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{"10.1.1.1"},
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     ptr.To("p11"),
					Port:     ptr.To[int32](22),
					Protocol: ptr.To(v1.ProtocolUDP),
				}}
			}),
	}
	namedPortsLocalNoLocal := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1,
			func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{"10.1.1.1"},
				}, {
					Addresses: []string{"10.1.1.2"},
					NodeName:  ptr.To(testNodeName),
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     ptr.To("p11"),
					Port:     ptr.To[int32](11),
					Protocol: ptr.To(v1.ProtocolUDP),
				}, {
					Name:     ptr.To("p12"),
					Port:     ptr.To[int32](12),
					Protocol: ptr.To(v1.ProtocolUDP),
				}}
			}),
	}
	multipleSubsets := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subset1),
		makeTestEndpointSlice("ns1", "ep1", 2, subset2),
	}
	subsetLocal := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.2"},
			NodeName:  ptr.To(testNodeName),
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p12"),
			Port:     ptr.To[int32](12),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	multipleSubsetsWithLocal := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subset1),
		makeTestEndpointSlice("ns1", "ep1", 2, subsetLocal),
	}
	subsetMultiplePortsLocal := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.1"},
			NodeName:  ptr.To(testNodeName),
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p11"),
			Port:     ptr.To[int32](11),
			Protocol: ptr.To(v1.ProtocolUDP),
		}, {
			Name:     ptr.To("p12"),
			Port:     ptr.To[int32](12),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	subset3 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.3"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p13"),
			Port:     ptr.To[int32](13),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	multipleSubsetsMultiplePortsLocal := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subsetMultiplePortsLocal),
		makeTestEndpointSlice("ns1", "ep1", 2, subset3),
	}
	subsetMultipleIPsPorts1 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.1"},
		}, {
			Addresses: []string{"10.1.1.2"},
			NodeName:  ptr.To(testNodeName),
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p11"),
			Port:     ptr.To[int32](11),
			Protocol: ptr.To(v1.ProtocolUDP),
		}, {
			Name:     ptr.To("p12"),
			Port:     ptr.To[int32](12),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	subsetMultipleIPsPorts2 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.3"},
		}, {
			Addresses: []string{"10.1.1.4"},
			NodeName:  ptr.To(testNodeName),
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p13"),
			Port:     ptr.To[int32](13),
			Protocol: ptr.To(v1.ProtocolUDP),
		}, {
			Name:     ptr.To("p14"),
			Port:     ptr.To[int32](14),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	subsetMultipleIPsPorts3 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.2.2.1"},
		}, {
			Addresses: []string{"10.2.2.2"},
			NodeName:  ptr.To(testNodeName),
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p21"),
			Port:     ptr.To[int32](21),
			Protocol: ptr.To(v1.ProtocolUDP),
		}, {
			Name:     ptr.To("p22"),
			Port:     ptr.To[int32](22),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	multipleSubsetsIPsPorts := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subsetMultipleIPsPorts1),
		makeTestEndpointSlice("ns1", "ep1", 2, subsetMultipleIPsPorts2),
		makeTestEndpointSlice("ns2", "ep2", 1, subsetMultipleIPsPorts3),
	}
	complexSubset1 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.2.2.2"},
			NodeName:  ptr.To(testNodeName),
		}, {
			Addresses: []string{"10.2.2.22"},
			NodeName:  ptr.To(testNodeName),
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p22"),
			Port:     ptr.To[int32](22),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	complexSubset2 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.2.2.3"},
			NodeName:  ptr.To(testNodeName),
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p23"),
			Port:     ptr.To[int32](23),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	complexSubset3 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.4.4.4"},
			NodeName:  ptr.To(testNodeName),
		}, {
			Addresses: []string{"10.4.4.5"},
			NodeName:  ptr.To(testNodeName),
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p44"),
			Port:     ptr.To[int32](44),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	complexSubset4 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.4.4.6"},
			NodeName:  ptr.To(testNodeName),
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p45"),
			Port:     ptr.To[int32](45),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	complexSubset5 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.1"},
		}, {
			Addresses: []string{"10.1.1.11"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p11"),
			Port:     ptr.To[int32](11),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	complexSubset6 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.2"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p12"),
			Port:     ptr.To[int32](12),
			Protocol: ptr.To(v1.ProtocolUDP),
		}, {
			Name:     ptr.To("p122"),
			Port:     ptr.To[int32](122),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	complexSubset7 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.3.3.3"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p33"),
			Port:     ptr.To[int32](33),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	complexSubset8 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.4.4.4"},
			NodeName:  ptr.To(testNodeName),
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p44"),
			Port:     ptr.To[int32](44),
			Protocol: ptr.To(v1.ProtocolUDP),
		}}
	}
	complexBefore := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subset1),
		nil,
		makeTestEndpointSlice("ns2", "ep2", 1, complexSubset1),
		makeTestEndpointSlice("ns2", "ep2", 2, complexSubset2),
		nil,
		makeTestEndpointSlice("ns4", "ep4", 1, complexSubset3),
		makeTestEndpointSlice("ns4", "ep4", 2, complexSubset4),
	}
	complexAfter := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, complexSubset5),
		makeTestEndpointSlice("ns1", "ep1", 2, complexSubset6),
		nil,
		nil,
		makeTestEndpointSlice("ns3", "ep3", 1, complexSubset7),
		makeTestEndpointSlice("ns4", "ep4", 1, complexSubset8),
		nil,
	}

	testCases := []struct {
		// previousEndpoints and currentEndpoints are used to call appropriate
		// handlers OnEndpoints* (based on whether corresponding values are nil
		// or non-nil) and must be of equal length.
		name                             string
		previousEndpoints                []*discovery.EndpointSlice
		currentEndpoints                 []*discovery.EndpointSlice
		oldEndpoints                     map[proxy.ServicePortName][]endpointExpectation
		expectedResult                   map[proxy.ServicePortName][]endpointExpectation
		expectedConntrackCleanupRequired bool
		expectedLocalEndpoints           map[types.NamespacedName]int
	}{{
		// Case[0]: nothing
		name:                             "nothing",
		oldEndpoints:                     map[proxy.ServicePortName][]endpointExpectation{},
		expectedResult:                   map[proxy.ServicePortName][]endpointExpectation{},
		expectedConntrackCleanupRequired: false,
		expectedLocalEndpoints:           map[types.NamespacedName]int{},
	}, {
		// Case[1]: no change, named port, local
		name:              "no change, named port, local",
		previousEndpoints: namedPortLocal,
		currentEndpoints:  namedPortLocal,
		oldEndpoints: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: true},
			},
		},
		expectedConntrackCleanupRequired: false,
		expectedLocalEndpoints: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[2]: no change, multiple subsets
		name:              "no change, multiple subsets",
		previousEndpoints: multipleSubsets,
		currentEndpoints:  multipleSubsets,
		oldEndpoints: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{endpoint: "10.1.1.2:12", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{endpoint: "10.1.1.2:12", isLocal: false},
			},
		},
		expectedConntrackCleanupRequired: false,
		expectedLocalEndpoints:           map[types.NamespacedName]int{},
	}, {
		// Case[3]: no change, multiple subsets, multiple ports, local
		name:              "no change, multiple subsets, multiple ports, local",
		previousEndpoints: multipleSubsetsMultiplePortsLocal,
		currentEndpoints:  multipleSubsetsMultiplePortsLocal,
		oldEndpoints: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:12", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{endpoint: "10.1.1.3:13", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:12", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{endpoint: "10.1.1.3:13", isLocal: false},
			},
		},
		expectedConntrackCleanupRequired: false,
		expectedLocalEndpoints: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[4]: no change, multiple endpoints, subsets, IPs, and ports
		name:              "no change, multiple endpoints, subsets, IPs, and ports",
		previousEndpoints: multipleSubsetsIPsPorts,
		currentEndpoints:  multipleSubsetsIPsPorts,
		oldEndpoints: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
				{endpoint: "10.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:12", isLocal: false},
				{endpoint: "10.1.1.2:12", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{endpoint: "10.1.1.3:13", isLocal: false},
				{endpoint: "10.1.1.4:13", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p14", v1.ProtocolUDP): {
				{endpoint: "10.1.1.3:14", isLocal: false},
				{endpoint: "10.1.1.4:14", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p21", v1.ProtocolUDP): {
				{endpoint: "10.2.2.1:21", isLocal: false},
				{endpoint: "10.2.2.2:21", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{endpoint: "10.2.2.1:22", isLocal: false},
				{endpoint: "10.2.2.2:22", isLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
				{endpoint: "10.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:12", isLocal: false},
				{endpoint: "10.1.1.2:12", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{endpoint: "10.1.1.3:13", isLocal: false},
				{endpoint: "10.1.1.4:13", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p14", v1.ProtocolUDP): {
				{endpoint: "10.1.1.3:14", isLocal: false},
				{endpoint: "10.1.1.4:14", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p21", v1.ProtocolUDP): {
				{endpoint: "10.2.2.1:21", isLocal: false},
				{endpoint: "10.2.2.2:21", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{endpoint: "10.2.2.1:22", isLocal: false},
				{endpoint: "10.2.2.2:22", isLocal: true},
			},
		},
		expectedConntrackCleanupRequired: false,
		expectedLocalEndpoints: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 2,
			makeNSN("ns2", "ep2"): 1,
		},
	}, {
		// Case[5]: add an Endpoints
		name:              "add an Endpoints",
		previousEndpoints: []*discovery.EndpointSlice{nil},
		currentEndpoints:  namedPortLocal,
		oldEndpoints:      map[proxy.ServicePortName][]endpointExpectation{},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: true},
			},
		},
		expectedConntrackCleanupRequired: true,
		expectedLocalEndpoints: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[6]: remove an Endpoints
		name:              "remove an Endpoints",
		previousEndpoints: namedPortLocal,
		currentEndpoints:  []*discovery.EndpointSlice{nil},
		oldEndpoints: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: true},
			},
		},
		expectedConntrackCleanupRequired: true,
		expectedResult:                   map[proxy.ServicePortName][]endpointExpectation{},
		expectedLocalEndpoints:           map[types.NamespacedName]int{},
	}, {
		// Case[7]: add an IP and port
		name:              "add an IP and port",
		previousEndpoints: namedPort,
		currentEndpoints:  namedPortsLocalNoLocal,
		oldEndpoints: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
				{endpoint: "10.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:12", isLocal: false},
				{endpoint: "10.1.1.2:12", isLocal: true},
			},
		},
		expectedConntrackCleanupRequired: true,
		expectedLocalEndpoints: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[8]: remove an IP and port
		name:              "remove an IP and port",
		previousEndpoints: namedPortsLocalNoLocal,
		currentEndpoints:  namedPort,
		oldEndpoints: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
				{endpoint: "10.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:12", isLocal: false},
				{endpoint: "10.1.1.2:12", isLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
		},
		expectedConntrackCleanupRequired: true,
		expectedLocalEndpoints:           map[types.NamespacedName]int{},
	}, {
		// Case[9]: add a subset
		name:              "add a subset",
		previousEndpoints: []*discovery.EndpointSlice{namedPort[0], nil},
		currentEndpoints:  multipleSubsetsWithLocal,
		oldEndpoints: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{endpoint: "10.1.1.2:12", isLocal: true},
			},
		},
		expectedConntrackCleanupRequired: true,
		expectedLocalEndpoints: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[10]: remove a subset
		name:              "remove a subset",
		previousEndpoints: multipleSubsets,
		currentEndpoints:  []*discovery.EndpointSlice{namedPort[0], nil},
		oldEndpoints: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{endpoint: "10.1.1.2:12", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
		},
		expectedConntrackCleanupRequired: true,
		expectedLocalEndpoints:           map[types.NamespacedName]int{},
	}, {
		// Case[11]: rename a port
		name:              "rename a port",
		previousEndpoints: namedPort,
		currentEndpoints:  namedPortRenamed,
		oldEndpoints: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11-2", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
		},
		expectedConntrackCleanupRequired: true,
		expectedLocalEndpoints:           map[types.NamespacedName]int{},
	}, {
		// Case[12]: renumber a port
		name:              "renumber a port",
		previousEndpoints: namedPort,
		currentEndpoints:  namedPortRenumbered,
		oldEndpoints: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:22", isLocal: false},
			},
		},
		expectedConntrackCleanupRequired: true,
		expectedLocalEndpoints:           map[types.NamespacedName]int{},
	}, {
		// Case[13]: complex add and remove
		name:              "complex add and remove",
		previousEndpoints: complexBefore,
		currentEndpoints:  complexAfter,
		oldEndpoints: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{endpoint: "10.2.2.22:22", isLocal: true},
				{endpoint: "10.2.2.2:22", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p23", v1.ProtocolUDP): {
				{endpoint: "10.2.2.3:23", isLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP): {
				{endpoint: "10.4.4.4:44", isLocal: true},
				{endpoint: "10.4.4.5:44", isLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p45", v1.ProtocolUDP): {
				{endpoint: "10.4.4.6:45", isLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.11:11", isLocal: false},
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{endpoint: "10.1.1.2:12", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p122", v1.ProtocolUDP): {
				{endpoint: "10.1.1.2:122", isLocal: false},
			},
			makeServicePortName("ns3", "ep3", "p33", v1.ProtocolUDP): {
				{endpoint: "10.3.3.3:33", isLocal: false},
			},
			makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP): {
				{endpoint: "10.4.4.4:44", isLocal: true},
			},
		},
		expectedConntrackCleanupRequired: true,
		expectedLocalEndpoints: map[types.NamespacedName]int{
			makeNSN("ns4", "ep4"): 1,
		},
	}, {
		// Case[14]: change from 0 endpoint address to 1 unnamed port
		name:              "change from 0 endpoint address to 1 unnamed port",
		previousEndpoints: emptyEndpointSlices,
		currentEndpoints:  namedPort,
		oldEndpoints:      map[proxy.ServicePortName][]endpointExpectation{},
		expectedResult: map[proxy.ServicePortName][]endpointExpectation{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{endpoint: "10.1.1.1:11", isLocal: false},
			},
		},
		expectedConntrackCleanupRequired: true,
		expectedLocalEndpoints:           map[types.NamespacedName]int{},
	},
	}

	for tci, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ipt := iptablestest.NewFake()
			fp := NewFakeProxier(ipt)

			// First check that after adding all previous versions of endpoints,
			// the fp.oldEndpoints is as we expect.
			for i := range tc.previousEndpoints {
				if tc.previousEndpoints[i] != nil {
					fp.OnEndpointSliceAdd(tc.previousEndpoints[i])
				}
			}
			fp.endpointsMap.Update(fp.endpointsChanges)
			checkEndpointExpectations(t, tci, fp.endpointsMap, tc.oldEndpoints)

			// Now let's call appropriate handlers to get to state we want to be.
			if len(tc.previousEndpoints) != len(tc.currentEndpoints) {
				t.Fatalf("[%d] different lengths of previous and current endpoints", tci)
			}

			for i := range tc.previousEndpoints {
				prev, curr := tc.previousEndpoints[i], tc.currentEndpoints[i]
				switch {
				case prev == nil:
					fp.OnEndpointSliceAdd(curr)
				case curr == nil:
					fp.OnEndpointSliceDelete(prev)
				default:
					fp.OnEndpointSliceUpdate(prev, curr)
				}
			}
			result := fp.endpointsMap.Update(fp.endpointsChanges)
			newMap := fp.endpointsMap
			checkEndpointExpectations(t, tci, newMap, tc.expectedResult)
			if result.ConntrackCleanupRequired != tc.expectedConntrackCleanupRequired {
				t.Errorf("[%d] expected conntrackCleanupRequired to be %t, got %t", tci, tc.expectedConntrackCleanupRequired, result.ConntrackCleanupRequired)
			}
			localReadyEndpoints := fp.endpointsMap.LocalReadyEndpoints()
			if !reflect.DeepEqual(localReadyEndpoints, tc.expectedLocalEndpoints) {
				t.Errorf("[%d] expected local endpoints %v, got %v", tci, tc.expectedLocalEndpoints, localReadyEndpoints)
			}
		})
	}
}

// TestHealthCheckNodePortWhenTerminating tests that health check node ports are not enabled when all local endpoints are terminating
func TestHealthCheckNodePortWhenTerminating(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)
	fp.OnServiceSynced()
	fp.OnEndpointSlicesSynced()

	serviceName := "svc1"
	namespaceName := "ns1"

	fp.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
		Spec: v1.ServiceSpec{
			ClusterIP: "172.30.1.1",
			Selector:  map[string]string{"foo": "bar"},
			Ports:     []v1.ServicePort{{Name: "", TargetPort: intstr.FromInt32(80), Protocol: v1.ProtocolTCP}},
		},
	})

	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", serviceName),
			Namespace: namespaceName,
			Labels:    map[string]string{discovery.LabelServiceName: serviceName},
		},
		Ports: []discovery.EndpointPort{{
			Name:     ptr.To(""),
			Port:     ptr.To[int32](80),
			Protocol: ptr.To(v1.ProtocolTCP),
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{{
			Addresses:  []string{"10.0.1.1"},
			Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
			NodeName:   ptr.To(testNodeName),
		}, {
			Addresses:  []string{"10.0.1.2"},
			Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
			NodeName:   ptr.To(testNodeName),
		}, {
			Addresses:  []string{"10.0.1.3"},
			Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
			NodeName:   ptr.To(testNodeName),
		}, { // not ready endpoints should be ignored
			Addresses:  []string{"10.0.1.4"},
			Conditions: discovery.EndpointConditions{Ready: ptr.To(false)},
			NodeName:   ptr.To(testNodeName),
		}},
	}

	fp.OnEndpointSliceAdd(endpointSlice)
	_ = fp.endpointsMap.Update(fp.endpointsChanges)
	localReadyEndpoints := fp.endpointsMap.LocalReadyEndpoints()
	if len(localReadyEndpoints) != 1 {
		t.Errorf("unexpected number of local ready endpoints, expected 1 but got: %d", len(localReadyEndpoints))
	}

	// set all endpoints to terminating
	endpointSliceTerminating := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", serviceName),
			Namespace: namespaceName,
			Labels:    map[string]string{discovery.LabelServiceName: serviceName},
		},
		Ports: []discovery.EndpointPort{{
			Name:     ptr.To(""),
			Port:     ptr.To[int32](80),
			Protocol: ptr.To(v1.ProtocolTCP),
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{{
			Addresses: []string{"10.0.1.1"},
			Conditions: discovery.EndpointConditions{
				Ready:       ptr.To(false),
				Serving:     ptr.To(true),
				Terminating: ptr.To(false),
			},
			NodeName: ptr.To(testNodeName),
		}, {
			Addresses: []string{"10.0.1.2"},
			Conditions: discovery.EndpointConditions{
				Ready:       ptr.To(false),
				Serving:     ptr.To(true),
				Terminating: ptr.To(true),
			},
			NodeName: ptr.To(testNodeName),
		}, {
			Addresses: []string{"10.0.1.3"},
			Conditions: discovery.EndpointConditions{
				Ready:       ptr.To(false),
				Serving:     ptr.To(true),
				Terminating: ptr.To(true),
			},
			NodeName: ptr.To(testNodeName),
		}, { // not ready endpoints should be ignored
			Addresses: []string{"10.0.1.4"},
			Conditions: discovery.EndpointConditions{
				Ready:       ptr.To(false),
				Serving:     ptr.To(false),
				Terminating: ptr.To(true),
			},
			NodeName: ptr.To(testNodeName),
		}},
	}

	fp.OnEndpointSliceUpdate(endpointSlice, endpointSliceTerminating)
	_ = fp.endpointsMap.Update(fp.endpointsChanges)
	localReadyEndpoints = fp.endpointsMap.LocalReadyEndpoints()
	if len(localReadyEndpoints) != 0 {
		t.Errorf("unexpected number of local ready endpoints, expected 0 but got: %d", len(localReadyEndpoints))
	}
}

func TestProxierMetricsIPTablesTotalRules(t *testing.T) {
	logger, _ := klogtesting.NewTestContext(t)
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)

	metrics.RegisterMetrics(kubeproxyconfig.ProxyModeIPTables)

	svcIP := "172.30.0.41"
	svcPort := 80
	nodePort := 31201
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(nodePort),
			}}
		}),
	)
	fp.syncProxyRules()
	iptablesData := fp.iptablesData.String()

	nFilterRules := countRulesFromMetric(logger, utiliptables.TableFilter, string(fp.ipFamily))
	expectedFilterRules := countRules(logger, utiliptables.TableFilter, iptablesData)

	if nFilterRules != expectedFilterRules {
		t.Fatalf("Wrong number of filter rule: expected %d got %d\n%s", expectedFilterRules, nFilterRules, iptablesData)
	}

	nNatRules := countRulesFromMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	expectedNatRules := countRules(logger, utiliptables.TableNAT, iptablesData)

	if nNatRules != expectedNatRules {
		t.Fatalf("Wrong number of nat rules: expected %d got %d\n%s", expectedNatRules, nNatRules, iptablesData)
	}

	populateEndpointSlices(fp,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{"10.0.0.2"},
			}, {
				Addresses: []string{"10.0.0.5"},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	fp.syncProxyRules()
	iptablesData = fp.iptablesData.String()

	nFilterRules = countRulesFromMetric(logger, utiliptables.TableFilter, string(fp.ipFamily))
	expectedFilterRules = countRules(logger, utiliptables.TableFilter, iptablesData)

	if nFilterRules != expectedFilterRules {
		t.Fatalf("Wrong number of filter rule: expected %d got %d\n%s", expectedFilterRules, nFilterRules, iptablesData)
	}

	nNatRules = countRulesFromMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	expectedNatRules = countRules(logger, utiliptables.TableNAT, iptablesData)

	if nNatRules != expectedNatRules {
		t.Fatalf("Wrong number of nat rules: expected %d got %d\n%s", expectedNatRules, nNatRules, iptablesData)
	}
}

// TODO(thockin): add *more* tests for syncProxyRules() or break it down further and test the pieces.

// This test ensures that the iptables proxier supports translating Endpoints to
// iptables output when internalTrafficPolicy is specified
func TestInternalTrafficPolicy(t *testing.T) {
	type endpoint struct {
		ip       string
		nodeName string
	}

	testCases := []struct {
		name                  string
		line                  int
		internalTrafficPolicy *v1.ServiceInternalTrafficPolicy
		endpoints             []endpoint
		flowTests             []packetFlowTest
	}{
		{
			name:                  "internalTrafficPolicy is cluster",
			line:                  getLine(),
			internalTrafficPolicy: ptr.To(v1.ServiceInternalTrafficPolicyCluster),
			endpoints: []endpoint{
				{"10.0.1.1", testNodeName},
				{"10.0.1.2", "node1"},
				{"10.0.1.3", "node2"},
			},
			flowTests: []packetFlowTest{
				{
					name:     "pod to ClusterIP hits all endpoints",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "10.0.1.1:80, 10.0.1.2:80, 10.0.1.3:80",
					masq:     false,
				},
			},
		},
		{
			name:                  "internalTrafficPolicy is local and there is one local endpoint",
			line:                  getLine(),
			internalTrafficPolicy: ptr.To(v1.ServiceInternalTrafficPolicyLocal),
			endpoints: []endpoint{
				{"10.0.1.1", testNodeName},
				{"10.0.1.2", "node1"},
				{"10.0.1.3", "node2"},
			},
			flowTests: []packetFlowTest{
				{
					name:     "pod to ClusterIP hits only local endpoint",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "10.0.1.1:80",
					masq:     false,
				},
			},
		},
		{
			name:                  "internalTrafficPolicy is local and there are multiple local endpoints",
			line:                  getLine(),
			internalTrafficPolicy: ptr.To(v1.ServiceInternalTrafficPolicyLocal),
			endpoints: []endpoint{
				{"10.0.1.1", testNodeName},
				{"10.0.1.2", testNodeName},
				{"10.0.1.3", "node2"},
			},
			flowTests: []packetFlowTest{
				{
					name:     "pod to ClusterIP hits all local endpoints",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "10.0.1.1:80, 10.0.1.2:80",
					masq:     false,
				},
			},
		},
		{
			name:                  "internalTrafficPolicy is local and there are no local endpoints",
			line:                  getLine(),
			internalTrafficPolicy: ptr.To(v1.ServiceInternalTrafficPolicyLocal),
			endpoints: []endpoint{
				{"10.0.1.1", "node0"},
				{"10.0.1.2", "node1"},
				{"10.0.1.3", "node2"},
			},
			flowTests: []packetFlowTest{
				{
					name:     "no endpoints",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "DROP",
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ipt := iptablestest.NewFake()
			fp := NewFakeProxier(ipt)
			fp.OnServiceSynced()
			fp.OnEndpointSlicesSynced()

			serviceName := "svc1"
			namespaceName := "ns1"

			svc := &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
				Spec: v1.ServiceSpec{
					ClusterIP: "172.30.1.1",
					Selector:  map[string]string{"foo": "bar"},
					Ports:     []v1.ServicePort{{Name: "", Port: 80, Protocol: v1.ProtocolTCP}},
				},
			}
			if tc.internalTrafficPolicy != nil {
				svc.Spec.InternalTrafficPolicy = tc.internalTrafficPolicy
			}

			fp.OnServiceAdd(svc)

			endpointSlice := &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("%s-1", serviceName),
					Namespace: namespaceName,
					Labels:    map[string]string{discovery.LabelServiceName: serviceName},
				},
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}},
				AddressType: discovery.AddressTypeIPv4,
			}
			for _, ep := range tc.endpoints {
				endpointSlice.Endpoints = append(endpointSlice.Endpoints, discovery.Endpoint{
					Addresses:  []string{ep.ip},
					Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
					NodeName:   ptr.To(ep.nodeName),
				})
			}

			fp.OnEndpointSliceAdd(endpointSlice)
			fp.syncProxyRules()
			runPacketFlowTests(t, tc.line, ipt, testNodeIPs, tc.flowTests)

			fp.OnEndpointSliceDelete(endpointSlice)
			fp.syncProxyRules()
			runPacketFlowTests(t, tc.line, ipt, testNodeIPs, []packetFlowTest{
				{
					name:     "endpoints deleted",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "REJECT",
				},
			})
		})
	}
}

// TestTerminatingEndpointsTrafficPolicyLocal tests that when there are local ready and
// ready + terminating endpoints, only the ready endpoints are used.
func TestTerminatingEndpointsTrafficPolicyLocal(t *testing.T) {
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "svc1", Namespace: "ns1"},
		Spec: v1.ServiceSpec{
			ClusterIP:             "172.30.1.1",
			Type:                  v1.ServiceTypeLoadBalancer,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyLocal,
			Ports: []v1.ServicePort{
				{
					Name:       "",
					TargetPort: intstr.FromInt32(80),
					Port:       80,
					Protocol:   v1.ProtocolTCP,
				},
			},
			HealthCheckNodePort: 30000,
		},
		Status: v1.ServiceStatus{
			LoadBalancer: v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{
					{IP: "1.2.3.4"},
				},
			},
		},
	}

	testcases := []struct {
		name          string
		line          int
		endpointslice *discovery.EndpointSlice
		flowTests     []packetFlowTest
	}{
		{
			name: "ready endpoints exist",
			line: getLine(),
			endpointslice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("%s-1", "svc1"),
					Namespace: "ns1",
					Labels:    map[string]string{discovery.LabelServiceName: "svc1"},
				},
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}},
				AddressType: discovery.AddressTypeIPv4,
				Endpoints: []discovery.Endpoint{
					{
						Addresses: []string{"10.0.1.1"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(true),
							Serving:     ptr.To(true),
							Terminating: ptr.To(false),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						Addresses: []string{"10.0.1.2"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(true),
							Serving:     ptr.To(true),
							Terminating: ptr.To(false),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						// this endpoint should be ignored for external since there are ready non-terminating endpoints
						Addresses: []string{"10.0.1.3"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(true),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						// this endpoint should be ignored for external since there are ready non-terminating endpoints
						Addresses: []string{"10.0.1.4"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(false),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						// this endpoint should be ignored for external since it's not local
						Addresses: []string{"10.0.1.5"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(true),
							Serving:     ptr.To(true),
							Terminating: ptr.To(false),
						},
						NodeName: ptr.To("node-1"),
					},
				},
			},
			flowTests: []packetFlowTest{
				{
					name:     "pod to clusterIP",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "10.0.1.1:80, 10.0.1.2:80, 10.0.1.5:80",
					masq:     false,
				},
				{
					name:     "external to LB",
					sourceIP: testExternalClient,
					destIP:   "1.2.3.4",
					destPort: 80,
					output:   "10.0.1.1:80, 10.0.1.2:80",
					masq:     false,
				},
			},
		},
		{
			name: "only terminating endpoints exist",
			line: getLine(),
			endpointslice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("%s-1", "svc1"),
					Namespace: "ns1",
					Labels:    map[string]string{discovery.LabelServiceName: "svc1"},
				},
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}},
				AddressType: discovery.AddressTypeIPv4,
				Endpoints: []discovery.Endpoint{
					{
						// this endpoint should be used since there are only ready terminating endpoints
						Addresses: []string{"10.0.1.2"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(true),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						// this endpoint should be used since there are only ready terminating endpoints
						Addresses: []string{"10.0.1.3"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(true),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						// this endpoint should not be used since it is both terminating and not ready.
						Addresses: []string{"10.0.1.4"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(false),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						// this endpoint should be ignored for external since it's not local
						Addresses: []string{"10.0.1.5"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(true),
							Serving:     ptr.To(true),
							Terminating: ptr.To(false),
						},
						NodeName: ptr.To("node-1"),
					},
				},
			},
			flowTests: []packetFlowTest{
				{
					name:     "pod to clusterIP",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "10.0.1.5:80",
					masq:     false,
				},
				{
					name:     "external to LB",
					sourceIP: testExternalClient,
					destIP:   "1.2.3.4",
					destPort: 80,
					output:   "10.0.1.2:80, 10.0.1.3:80",
					masq:     false,
				},
			},
		},
		{
			name: "terminating endpoints on remote node",
			line: getLine(),
			endpointslice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("%s-1", "svc1"),
					Namespace: "ns1",
					Labels:    map[string]string{discovery.LabelServiceName: "svc1"},
				},
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}},
				AddressType: discovery.AddressTypeIPv4,
				Endpoints: []discovery.Endpoint{
					{
						// this endpoint won't be used because it's not local,
						// but it will prevent a REJECT rule from being created
						Addresses: []string{"10.0.1.5"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(true),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To("node-1"),
					},
				},
			},
			flowTests: []packetFlowTest{
				{
					name:     "pod to clusterIP",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "10.0.1.5:80",
				},
				{
					name:     "external to LB, no locally-usable endpoints",
					sourceIP: testExternalClient,
					destIP:   "1.2.3.4",
					destPort: 80,
					output:   "DROP",
				},
			},
		},
		{
			name: "no usable endpoints on any node",
			line: getLine(),
			endpointslice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("%s-1", "svc1"),
					Namespace: "ns1",
					Labels:    map[string]string{discovery.LabelServiceName: "svc1"},
				},
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}},
				AddressType: discovery.AddressTypeIPv4,
				Endpoints: []discovery.Endpoint{
					{
						// Local but not ready or serving
						Addresses: []string{"10.0.1.5"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(false),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						// Remote and not ready or serving
						Addresses: []string{"10.0.1.5"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(false),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To("node-1"),
					},
				},
			},
			flowTests: []packetFlowTest{
				{
					name:     "pod to clusterIP, no usable endpoints",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "REJECT",
				},
				{
					name:     "external to LB, no usable endpoints",
					sourceIP: testExternalClient,
					destIP:   "1.2.3.4",
					destPort: 80,
					output:   "REJECT",
				},
			},
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			ipt := iptablestest.NewFake()
			fp := NewFakeProxier(ipt)
			fp.OnServiceSynced()
			fp.OnEndpointSlicesSynced()

			fp.OnServiceAdd(service)

			fp.OnEndpointSliceAdd(testcase.endpointslice)
			fp.syncProxyRules()
			runPacketFlowTests(t, testcase.line, ipt, testNodeIPs, testcase.flowTests)

			fp.OnEndpointSliceDelete(testcase.endpointslice)
			fp.syncProxyRules()
			runPacketFlowTests(t, testcase.line, ipt, testNodeIPs, []packetFlowTest{
				{
					name:     "pod to clusterIP after endpoints deleted",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "REJECT",
				},
				{
					name:     "external to LB after endpoints deleted",
					sourceIP: testExternalClient,
					destIP:   "1.2.3.4",
					destPort: 80,
					output:   "REJECT",
				},
			})
		})
	}
}

// TestTerminatingEndpointsTrafficPolicyCluster tests that when there are cluster-wide
// ready and ready + terminating endpoints, only the ready endpoints are used.
func TestTerminatingEndpointsTrafficPolicyCluster(t *testing.T) {
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "svc1", Namespace: "ns1"},
		Spec: v1.ServiceSpec{
			ClusterIP:             "172.30.1.1",
			Type:                  v1.ServiceTypeLoadBalancer,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyCluster,
			Ports: []v1.ServicePort{
				{
					Name:       "",
					TargetPort: intstr.FromInt32(80),
					Port:       80,
					Protocol:   v1.ProtocolTCP,
				},
			},
			HealthCheckNodePort: 30000,
		},
		Status: v1.ServiceStatus{
			LoadBalancer: v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{
					{IP: "1.2.3.4"},
				},
			},
		},
	}

	testcases := []struct {
		name          string
		line          int
		endpointslice *discovery.EndpointSlice
		flowTests     []packetFlowTest
	}{
		{
			name: "ready endpoints exist",
			line: getLine(),
			endpointslice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("%s-1", "svc1"),
					Namespace: "ns1",
					Labels:    map[string]string{discovery.LabelServiceName: "svc1"},
				},
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}},
				AddressType: discovery.AddressTypeIPv4,
				Endpoints: []discovery.Endpoint{
					{
						Addresses: []string{"10.0.1.1"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(true),
							Serving:     ptr.To(true),
							Terminating: ptr.To(false),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						Addresses: []string{"10.0.1.2"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(true),
							Serving:     ptr.To(true),
							Terminating: ptr.To(false),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						// this endpoint should be ignored since there are ready non-terminating endpoints
						Addresses: []string{"10.0.1.3"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(true),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To("another-node"),
					},
					{
						// this endpoint should be ignored since it is not "serving"
						Addresses: []string{"10.0.1.4"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(false),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To("another-node"),
					},
					{
						Addresses: []string{"10.0.1.5"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(true),
							Serving:     ptr.To(true),
							Terminating: ptr.To(false),
						},
						NodeName: ptr.To("another-node"),
					},
				},
			},
			flowTests: []packetFlowTest{
				{
					name:     "pod to clusterIP",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "10.0.1.1:80, 10.0.1.2:80, 10.0.1.5:80",
					masq:     false,
				},
				{
					name:     "external to LB",
					sourceIP: testExternalClient,
					destIP:   "1.2.3.4",
					destPort: 80,
					output:   "10.0.1.1:80, 10.0.1.2:80, 10.0.1.5:80",
					masq:     true,
				},
			},
		},
		{
			name: "only terminating endpoints exist",
			line: getLine(),
			endpointslice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("%s-1", "svc1"),
					Namespace: "ns1",
					Labels:    map[string]string{discovery.LabelServiceName: "svc1"},
				},
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}},
				AddressType: discovery.AddressTypeIPv4,
				Endpoints: []discovery.Endpoint{
					{
						// this endpoint should be used since there are only ready terminating endpoints
						Addresses: []string{"10.0.1.2"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(true),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						// this endpoint should be used since there are only ready terminating endpoints
						Addresses: []string{"10.0.1.3"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(true),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						// this endpoint should not be used since it is both terminating and not ready.
						Addresses: []string{"10.0.1.4"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(false),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To("another-node"),
					},
					{
						// this endpoint should be used since there are only ready terminating endpoints
						Addresses: []string{"10.0.1.5"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(true),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To("another-node"),
					},
				},
			},
			flowTests: []packetFlowTest{
				{
					name:     "pod to clusterIP",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "10.0.1.2:80, 10.0.1.3:80, 10.0.1.5:80",
					masq:     false,
				},
				{
					name:     "external to LB",
					sourceIP: testExternalClient,
					destIP:   "1.2.3.4",
					destPort: 80,
					output:   "10.0.1.2:80, 10.0.1.3:80, 10.0.1.5:80",
					masq:     true,
				},
			},
		},
		{
			name: "terminating endpoints on remote node",
			line: getLine(),
			endpointslice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("%s-1", "svc1"),
					Namespace: "ns1",
					Labels:    map[string]string{discovery.LabelServiceName: "svc1"},
				},
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}},
				AddressType: discovery.AddressTypeIPv4,
				Endpoints: []discovery.Endpoint{
					{
						Addresses: []string{"10.0.1.5"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(true),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To("node-1"),
					},
				},
			},
			flowTests: []packetFlowTest{
				{
					name:     "pod to clusterIP",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "10.0.1.5:80",
					masq:     false,
				},
				{
					name:     "external to LB",
					sourceIP: testExternalClient,
					destIP:   "1.2.3.4",
					destPort: 80,
					output:   "10.0.1.5:80",
					masq:     true,
				},
			},
		},
		{
			name: "no usable endpoints on any node",
			line: getLine(),
			endpointslice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("%s-1", "svc1"),
					Namespace: "ns1",
					Labels:    map[string]string{discovery.LabelServiceName: "svc1"},
				},
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}},
				AddressType: discovery.AddressTypeIPv4,
				Endpoints: []discovery.Endpoint{
					{
						// Local, not ready or serving
						Addresses: []string{"10.0.1.5"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(false),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To(testNodeName),
					},
					{
						// Remote, not ready or serving
						Addresses: []string{"10.0.1.5"},
						Conditions: discovery.EndpointConditions{
							Ready:       ptr.To(false),
							Serving:     ptr.To(false),
							Terminating: ptr.To(true),
						},
						NodeName: ptr.To("node-1"),
					},
				},
			},
			flowTests: []packetFlowTest{
				{
					name:     "pod to clusterIP",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "REJECT",
				},
				{
					name:     "external to LB",
					sourceIP: testExternalClient,
					destIP:   "1.2.3.4",
					destPort: 80,
					output:   "REJECT",
				},
			},
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {

			ipt := iptablestest.NewFake()
			fp := NewFakeProxier(ipt)
			fp.OnServiceSynced()
			fp.OnEndpointSlicesSynced()

			fp.OnServiceAdd(service)

			fp.OnEndpointSliceAdd(testcase.endpointslice)
			fp.syncProxyRules()
			runPacketFlowTests(t, testcase.line, ipt, testNodeIPs, testcase.flowTests)

			fp.OnEndpointSliceDelete(testcase.endpointslice)
			fp.syncProxyRules()
			runPacketFlowTests(t, testcase.line, ipt, testNodeIPs, []packetFlowTest{
				{
					name:     "pod to clusterIP after endpoints deleted",
					sourceIP: "10.0.0.2",
					destIP:   "172.30.1.1",
					destPort: 80,
					output:   "REJECT",
				},
				{
					name:     "external to LB after endpoints deleted",
					sourceIP: testExternalClient,
					destIP:   "1.2.3.4",
					destPort: 80,
					output:   "REJECT",
				},
			})
		})
	}
}

func TestInternalExternalMasquerade(t *testing.T) {
	// (Put the test setup code in an internal function so we can have it here at the
	// top, before the test cases that will be run against it.)
	setupTest := func(fp *Proxier) {
		makeServiceMap(fp,
			makeTestService("ns1", "svc1", func(svc *v1.Service) {
				svc.Spec.Type = "LoadBalancer"
				svc.Spec.ClusterIP = "172.30.0.41"
				svc.Spec.Ports = []v1.ServicePort{{
					Name:     "p80",
					Port:     80,
					Protocol: v1.ProtocolTCP,
					NodePort: int32(3001),
				}}
				svc.Spec.HealthCheckNodePort = 30001
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
					IP: "1.2.3.4",
				}}
			}),
			makeTestService("ns2", "svc2", func(svc *v1.Service) {
				svc.Spec.Type = "LoadBalancer"
				svc.Spec.ClusterIP = "172.30.0.42"
				svc.Spec.Ports = []v1.ServicePort{{
					Name:     "p80",
					Port:     80,
					Protocol: v1.ProtocolTCP,
					NodePort: int32(3002),
				}}
				svc.Spec.HealthCheckNodePort = 30002
				svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
					IP: "5.6.7.8",
				}}
			}),
			makeTestService("ns3", "svc3", func(svc *v1.Service) {
				svc.Spec.Type = "LoadBalancer"
				svc.Spec.ClusterIP = "172.30.0.43"
				svc.Spec.Ports = []v1.ServicePort{{
					Name:     "p80",
					Port:     80,
					Protocol: v1.ProtocolTCP,
					NodePort: int32(3003),
				}}
				svc.Spec.HealthCheckNodePort = 30003
				svc.Spec.InternalTrafficPolicy = ptr.To(v1.ServiceInternalTrafficPolicyLocal)
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
					IP: "9.10.11.12",
				}}
			}),
		)

		populateEndpointSlices(fp,
			makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{
					{
						Addresses: []string{"10.180.0.1"},
						NodeName:  ptr.To(testNodeName),
					},
					{
						Addresses: []string{"10.180.1.1"},
						NodeName:  ptr.To("remote"),
					},
				}
				eps.Ports = []discovery.EndpointPort{{
					Name:     ptr.To("p80"),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}}
			}),
			makeTestEndpointSlice("ns2", "svc2", 1, func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{
					{
						Addresses: []string{"10.180.0.2"},
						NodeName:  ptr.To(testNodeName),
					},
					{
						Addresses: []string{"10.180.1.2"},
						NodeName:  ptr.To("remote"),
					},
				}
				eps.Ports = []discovery.EndpointPort{{
					Name:     ptr.To("p80"),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}}
			}),
			makeTestEndpointSlice("ns3", "svc3", 1, func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{
					{
						Addresses: []string{"10.180.0.3"},
						NodeName:  ptr.To(testNodeName),
					},
					{
						Addresses: []string{"10.180.1.3"},
						NodeName:  ptr.To("remote"),
					},
				}
				eps.Ports = []discovery.EndpointPort{{
					Name:     ptr.To("p80"),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}}
			}),
		)

		fp.syncProxyRules()
	}

	// We use the same flowTests for all of the testCases. The "output" and "masq"
	// values here represent the normal case (working localDetector, no masqueradeAll)
	flowTests := []packetFlowTest{
		{
			name:     "pod to ClusterIP",
			sourceIP: "10.0.0.2",
			destIP:   "172.30.0.41",
			destPort: 80,
			output:   "10.180.0.1:80, 10.180.1.1:80",
			masq:     false,
		},
		{
			name:     "pod to NodePort",
			sourceIP: "10.0.0.2",
			destIP:   testNodeIP,
			destPort: 3001,
			output:   "10.180.0.1:80, 10.180.1.1:80",
			masq:     true,
		},
		{
			name:     "pod to LB",
			sourceIP: "10.0.0.2",
			destIP:   "1.2.3.4",
			destPort: 80,
			output:   "10.180.0.1:80, 10.180.1.1:80",
			masq:     true,
		},
		{
			name:     "node to ClusterIP",
			sourceIP: testNodeIP,
			destIP:   "172.30.0.41",
			destPort: 80,
			output:   "10.180.0.1:80, 10.180.1.1:80",
			masq:     true,
		},
		{
			name:     "node to NodePort",
			sourceIP: testNodeIP,
			destIP:   testNodeIP,
			destPort: 3001,
			output:   "10.180.0.1:80, 10.180.1.1:80",
			masq:     true,
		},
		{
			name:     "localhost to NodePort",
			sourceIP: "127.0.0.1",
			destIP:   "127.0.0.1",
			destPort: 3001,
			output:   "10.180.0.1:80, 10.180.1.1:80",
			masq:     true,
		},
		{
			name:     "node to LB",
			sourceIP: testNodeIP,
			destIP:   "1.2.3.4",
			destPort: 80,
			output:   "10.180.0.1:80, 10.180.1.1:80",
			masq:     true,
		},
		{
			name:     "external to ClusterIP",
			sourceIP: testExternalClient,
			destIP:   "172.30.0.41",
			destPort: 80,
			output:   "10.180.0.1:80, 10.180.1.1:80",
			masq:     true,
		},
		{
			name:     "external to NodePort",
			sourceIP: testExternalClient,
			destIP:   testNodeIP,
			destPort: 3001,
			output:   "10.180.0.1:80, 10.180.1.1:80",
			masq:     true,
		},
		{
			name:     "external to LB",
			sourceIP: testExternalClient,
			destIP:   "1.2.3.4",
			destPort: 80,
			output:   "10.180.0.1:80, 10.180.1.1:80",
			masq:     true,
		},
		{
			name:     "pod to ClusterIP with eTP:Local",
			sourceIP: "10.0.0.2",
			destIP:   "172.30.0.42",
			destPort: 80,

			// externalTrafficPolicy does not apply to ClusterIP traffic, so same
			// as "Pod to ClusterIP"
			output: "10.180.0.2:80, 10.180.1.2:80",
			masq:   false,
		},
		{
			name:     "pod to NodePort with eTP:Local",
			sourceIP: "10.0.0.2",
			destIP:   testNodeIP,
			destPort: 3002,

			// See the comment below in the "pod to LB with eTP:Local" case.
			// It doesn't actually make sense to short-circuit here, since if
			// you connect directly to a NodePort from outside the cluster,
			// you only get the local endpoints. But it's simpler for us and
			// slightly more convenient for users to have this case get
			// short-circuited too.
			output: "10.180.0.2:80, 10.180.1.2:80",
			masq:   false,
		},
		{
			name:     "pod to LB with eTP:Local",
			sourceIP: "10.0.0.2",
			destIP:   "5.6.7.8",
			destPort: 80,

			// The short-circuit rule is supposed to make this behave the same
			// way it would if the packet actually went out to the LB and then
			// came back into the cluster. So it gets routed to all endpoints,
			// not just local ones. In reality, if the packet actually left
			// the cluster, it would have to get masqueraded, but since we can
			// avoid doing that in the short-circuit case, and not masquerading
			// is more useful, we avoid masquerading.
			output: "10.180.0.2:80, 10.180.1.2:80",
			masq:   false,
		},
		{
			name:     "node to ClusterIP with eTP:Local",
			sourceIP: testNodeIP,
			destIP:   "172.30.0.42",
			destPort: 80,

			// externalTrafficPolicy does not apply to ClusterIP traffic, so same
			// as "node to ClusterIP"
			output: "10.180.0.2:80, 10.180.1.2:80",
			masq:   true,
		},
		{
			name:     "node to NodePort with eTP:Local",
			sourceIP: testNodeIP,
			destIP:   testNodeIP,
			destPort: 3001,

			// The traffic gets short-circuited, ignoring externalTrafficPolicy, so
			// same as "node to NodePort" above.
			output: "10.180.0.1:80, 10.180.1.1:80",
			masq:   true,
		},
		{
			name:     "localhost to NodePort with eTP:Local",
			sourceIP: "127.0.0.1",
			destIP:   "127.0.0.1",
			destPort: 3002,

			// The traffic gets short-circuited, ignoring externalTrafficPolicy, so
			// same as "localhost to NodePort" above.
			output: "10.180.0.2:80, 10.180.1.2:80",
			masq:   true,
		},
		{
			name:     "node to LB with eTP:Local",
			sourceIP: testNodeIP,
			destIP:   "5.6.7.8",
			destPort: 80,

			// The traffic gets short-circuited, ignoring externalTrafficPolicy, so
			// same as "node to LB" above.
			output: "10.180.0.2:80, 10.180.1.2:80",
			masq:   true,
		},
		{
			name:     "external to ClusterIP with eTP:Local",
			sourceIP: testExternalClient,
			destIP:   "172.30.0.42",
			destPort: 80,

			// externalTrafficPolicy does not apply to ClusterIP traffic, so same
			// as "external to ClusterIP" above.
			output: "10.180.0.2:80, 10.180.1.2:80",
			masq:   true,
		},
		{
			name:     "external to NodePort with eTP:Local",
			sourceIP: testExternalClient,
			destIP:   testNodeIP,
			destPort: 3002,

			// externalTrafficPolicy applies; only the local endpoint is
			// selected, and we don't masquerade.
			output: "10.180.0.2:80",
			masq:   false,
		},
		{
			name:     "external to LB with eTP:Local",
			sourceIP: testExternalClient,
			destIP:   "5.6.7.8",
			destPort: 80,

			// externalTrafficPolicy applies; only the local endpoint is
			// selected, and we don't masquerade.
			output: "10.180.0.2:80",
			masq:   false,
		},
		{
			name:     "pod to ClusterIP with iTP:Local",
			sourceIP: "10.0.0.2",
			destIP:   "172.30.0.43",
			destPort: 80,

			// internalTrafficPolicy applies; only the local endpoint is
			// selected.
			output: "10.180.0.3:80",
			masq:   false,
		},
		{
			name:     "pod to NodePort with iTP:Local",
			sourceIP: "10.0.0.2",
			destIP:   testNodeIP,
			destPort: 3003,

			// internalTrafficPolicy does not apply to NodePort traffic, so same as
			// "pod to NodePort" above.
			output: "10.180.0.3:80, 10.180.1.3:80",
			masq:   true,
		},
		{
			name:     "pod to LB with iTP:Local",
			sourceIP: "10.0.0.2",
			destIP:   "9.10.11.12",
			destPort: 80,

			// internalTrafficPolicy does not apply to LoadBalancer traffic, so
			// same as "pod to LB" above.
			output: "10.180.0.3:80, 10.180.1.3:80",
			masq:   true,
		},
		{
			name:     "node to ClusterIP with iTP:Local",
			sourceIP: testNodeIP,
			destIP:   "172.30.0.43",
			destPort: 80,

			// internalTrafficPolicy applies; only the local endpoint is selected.
			// Traffic is masqueraded as in the "node to ClusterIP" case because
			// internalTrafficPolicy does not affect masquerading.
			output: "10.180.0.3:80",
			masq:   true,
		},
		{
			name:     "node to NodePort with iTP:Local",
			sourceIP: testNodeIP,
			destIP:   testNodeIP,
			destPort: 3003,

			// internalTrafficPolicy does not apply to NodePort traffic, so same as
			// "node to NodePort" above.
			output: "10.180.0.3:80, 10.180.1.3:80",
			masq:   true,
		},
		{
			name:     "localhost to NodePort with iTP:Local",
			sourceIP: "127.0.0.1",
			destIP:   "127.0.0.1",
			destPort: 3003,

			// internalTrafficPolicy does not apply to NodePort traffic, so same as
			// "localhost to NodePort" above.
			output: "10.180.0.3:80, 10.180.1.3:80",
			masq:   true,
		},
		{
			name:     "node to LB with iTP:Local",
			sourceIP: testNodeIP,
			destIP:   "9.10.11.12",
			destPort: 80,

			// internalTrafficPolicy does not apply to LoadBalancer traffic, so
			// same as "node to LB" above.
			output: "10.180.0.3:80, 10.180.1.3:80",
			masq:   true,
		},
		{
			name:     "external to ClusterIP with iTP:Local",
			sourceIP: testExternalClient,
			destIP:   "172.30.0.43",
			destPort: 80,

			// internalTrafficPolicy applies; only the local endpoint is selected.
			// Traffic is masqueraded as in the "external to ClusterIP" case
			// because internalTrafficPolicy does not affect masquerading.
			output: "10.180.0.3:80",
			masq:   true,
		},
		{
			name:     "external to NodePort with iTP:Local",
			sourceIP: testExternalClient,
			destIP:   testNodeIP,
			destPort: 3003,

			// internalTrafficPolicy does not apply to NodePort traffic, so same as
			// "external to NodePort" above.
			output: "10.180.0.3:80, 10.180.1.3:80",
			masq:   true,
		},
		{
			name:     "external to LB with iTP:Local",
			sourceIP: testExternalClient,
			destIP:   "9.10.11.12",
			destPort: 80,

			// internalTrafficPolicy does not apply to LoadBalancer traffic, so
			// same as "external to LB" above.
			output: "10.180.0.3:80, 10.180.1.3:80",
			masq:   true,
		},
	}

	type packetFlowTestOverride struct {
		output *string
		masq   *bool
	}

	testCases := []struct {
		name          string
		line          int
		masqueradeAll bool
		localDetector bool
		overrides     map[string]packetFlowTestOverride
	}{
		{
			name:          "base",
			line:          getLine(),
			masqueradeAll: false,
			localDetector: true,
			overrides:     nil,
		},
		{
			name:          "no LocalTrafficDetector",
			line:          getLine(),
			masqueradeAll: false,
			localDetector: false,
			overrides: map[string]packetFlowTestOverride{
				// With no LocalTrafficDetector, all traffic to a
				// ClusterIP is assumed to be from a pod, and thus to not
				// require masquerading.
				"node to ClusterIP": {
					masq: ptr.To(false),
				},
				"node to ClusterIP with eTP:Local": {
					masq: ptr.To(false),
				},
				"node to ClusterIP with iTP:Local": {
					masq: ptr.To(false),
				},
				"external to ClusterIP": {
					masq: ptr.To(false),
				},
				"external to ClusterIP with eTP:Local": {
					masq: ptr.To(false),
				},
				"external to ClusterIP with iTP:Local": {
					masq: ptr.To(false),
				},

				// And there's no eTP:Local short-circuit for pod traffic,
				// so pods get only the local endpoints.
				"pod to NodePort with eTP:Local": {
					output: ptr.To("10.180.0.2:80"),
				},
				"pod to LB with eTP:Local": {
					output: ptr.To("10.180.0.2:80"),
				},
			},
		},
		{
			name:          "masqueradeAll",
			line:          getLine(),
			masqueradeAll: true,
			localDetector: true,
			overrides: map[string]packetFlowTestOverride{
				// All "to ClusterIP" traffic gets masqueraded when using
				// --masquerade-all.
				"pod to ClusterIP": {
					masq: ptr.To(true),
				},
				"pod to ClusterIP with eTP:Local": {
					masq: ptr.To(true),
				},
				"pod to ClusterIP with iTP:Local": {
					masq: ptr.To(true),
				},
			},
		},
		{
			name:          "masqueradeAll, no LocalTrafficDetector",
			line:          getLine(),
			masqueradeAll: true,
			localDetector: false,
			overrides: map[string]packetFlowTestOverride{
				// As in "masqueradeAll"
				"pod to ClusterIP": {
					masq: ptr.To(true),
				},
				"pod to ClusterIP with eTP:Local": {
					masq: ptr.To(true),
				},
				"pod to ClusterIP with iTP:Local": {
					masq: ptr.To(true),
				},

				// As in "no LocalTrafficDetector"
				"pod to NodePort with eTP:Local": {
					output: ptr.To("10.180.0.2:80"),
				},
				"pod to LB with eTP:Local": {
					output: ptr.To("10.180.0.2:80"),
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ipt := iptablestest.NewFake()
			fp := NewFakeProxier(ipt)
			fp.masqueradeAll = tc.masqueradeAll
			if !tc.localDetector {
				fp.localDetector = proxyutil.NewNoOpLocalDetector()
			}
			setupTest(fp)

			// Merge base flowTests with per-test-case overrides
			tcFlowTests := make([]packetFlowTest, len(flowTests))
			overridesApplied := 0
			for i := range flowTests {
				tcFlowTests[i] = flowTests[i]
				if overrides, set := tc.overrides[flowTests[i].name]; set {
					overridesApplied++
					if overrides.masq != nil {
						if tcFlowTests[i].masq == *overrides.masq {
							t.Errorf("%q override value for masq is same as base value", flowTests[i].name)
						}
						tcFlowTests[i].masq = *overrides.masq
					}
					if overrides.output != nil {
						if tcFlowTests[i].output == *overrides.output {
							t.Errorf("%q override value for output is same as base value", flowTests[i].name)
						}
						tcFlowTests[i].output = *overrides.output
					}
				}
			}
			if overridesApplied != len(tc.overrides) {
				t.Errorf("%d overrides did not match any test case name!", len(tc.overrides)-overridesApplied)
			}
			runPacketFlowTests(t, tc.line, ipt, testNodeIPs, tcFlowTests)
		})
	}
}

func countEndpointsAndComments(iptablesData string, matchEndpoint string) (string, int, int) {
	var numEndpoints, numComments int
	var matched string
	for _, line := range strings.Split(iptablesData, "\n") {
		if strings.HasPrefix(line, "-A KUBE-SEP-") && strings.Contains(line, "-j DNAT") {
			numEndpoints++
			if strings.Contains(line, "--comment") {
				numComments++
			}
			if strings.Contains(line, matchEndpoint) {
				matched = line
			}
		}
	}
	return matched, numEndpoints, numComments
}

func TestSyncProxyRulesLargeClusterMode(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)
	fp.masqueradeAll = true
	fp.syncPeriod = 30 * time.Second

	makeServiceMap(fp,
		makeTestService("ns1", "svc1", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "172.30.0.41"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p80",
				Port:     80,
				Protocol: v1.ProtocolTCP,
			}}
		}),
		makeTestService("ns2", "svc2", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "172.30.0.42"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p8080",
				Port:     8080,
				Protocol: v1.ProtocolTCP,
			}}
		}),
		makeTestService("ns3", "svc3", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "172.30.0.43"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p8081",
				Port:     8081,
				Protocol: v1.ProtocolTCP,
			}}
		}),
	)

	populateEndpointSlices(fp,
		makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = make([]discovery.Endpoint, largeClusterEndpointsThreshold/2-1)
			for i := range eps.Endpoints {
				eps.Endpoints[i].Addresses = []string{fmt.Sprintf("10.0.%d.%d", i%256, i/256)}
			}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p80"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
		makeTestEndpointSlice("ns2", "svc2", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = make([]discovery.Endpoint, largeClusterEndpointsThreshold/2-1)
			for i := range eps.Endpoints {
				eps.Endpoints[i].Addresses = []string{fmt.Sprintf("10.1.%d.%d", i%256, i/256)}
			}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p8080"),
				Port:     ptr.To[int32](8080),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	fp.syncProxyRules()
	expectedEndpoints := 2 * (largeClusterEndpointsThreshold/2 - 1)

	firstEndpoint, numEndpoints, numComments := countEndpointsAndComments(fp.iptablesData.String(), "10.0.0.0")
	assert.Equal(t, "-A KUBE-SEP-DKGQUZGBKLTPAR56 -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.0.0.0:80", firstEndpoint)
	if numEndpoints != expectedEndpoints {
		t.Errorf("Found wrong number of endpoints: expected %d, got %d", expectedEndpoints, numEndpoints)
	}
	if numComments != numEndpoints {
		t.Errorf("numComments (%d) != numEndpoints (%d) when numEndpoints < threshold (%d)", numComments, numEndpoints, largeClusterEndpointsThreshold)
	}

	fp.OnEndpointSliceAdd(makeTestEndpointSlice("ns3", "svc3", 1, func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"203.0.113.4"},
		}, {
			Addresses: []string{"203.0.113.8"},
		}, {
			Addresses: []string{"203.0.113.12"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p8081"),
			Port:     ptr.To[int32](8081),
			Protocol: ptr.To(v1.ProtocolTCP),
		}}
	}))
	fp.syncProxyRules()

	firstEndpoint, numEndpoints, numComments = countEndpointsAndComments(fp.iptablesData.String(), "203.0.113.4")
	assert.Equal(t, "-A KUBE-SEP-RUVVH7YV3PHQBDOS -m tcp -p tcp -j DNAT --to-destination 203.0.113.4:8081", firstEndpoint)
	// syncProxyRules will only have output the endpoints for svc3, since the others
	// didn't change (and syncProxyRules doesn't automatically do a full resync when you
	// cross the largeClusterEndpointsThreshold).
	if numEndpoints != 3 {
		t.Errorf("Found wrong number of endpoints on partial resync: expected %d, got %d", 3, numEndpoints)
	}
	if numComments != 0 {
		t.Errorf("numComments (%d) != 0 after partial resync when numEndpoints (%d) > threshold (%d)", numComments, expectedEndpoints+3, largeClusterEndpointsThreshold)
	}

	// Now force a full resync and confirm that it rewrites the older services with
	// no comments as well.
	fp.forceSyncProxyRules()
	expectedEndpoints += 3

	firstEndpoint, numEndpoints, numComments = countEndpointsAndComments(fp.iptablesData.String(), "10.0.0.0")
	assert.Equal(t, "-A KUBE-SEP-DKGQUZGBKLTPAR56 -m tcp -p tcp -j DNAT --to-destination 10.0.0.0:80", firstEndpoint)
	if numEndpoints != expectedEndpoints {
		t.Errorf("Found wrong number of endpoints: expected %d, got %d", expectedEndpoints, numEndpoints)
	}
	if numComments != 0 {
		t.Errorf("numComments (%d) != 0 when numEndpoints (%d) > threshold (%d)", numComments, numEndpoints, largeClusterEndpointsThreshold)
	}

	// Now test service deletion; we have to create another service to do this though,
	// because if we deleted any of the existing services, we'd fall back out of large
	// cluster mode.
	svc4 := makeTestService("ns4", "svc4", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.30.0.44"
		svc.Spec.Ports = []v1.ServicePort{{
			Name:     "p8082",
			Port:     8082,
			Protocol: v1.ProtocolTCP,
		}}
	})
	fp.OnServiceAdd(svc4)
	fp.OnEndpointSliceAdd(makeTestEndpointSlice("ns4", "svc4", 1, func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.4.0.1"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p8082"),
			Port:     ptr.To[int32](8082),
			Protocol: ptr.To(v1.ProtocolTCP),
		}}
	}))
	fp.syncProxyRules()

	svc4Endpoint, numEndpoints, _ := countEndpointsAndComments(fp.iptablesData.String(), "10.4.0.1")
	assert.Equal(t, "-A KUBE-SEP-SU5STNODRYEWJAUF -m tcp -p tcp -j DNAT --to-destination 10.4.0.1:8082", svc4Endpoint, "svc4 endpoint was not created")
	// should only sync svc4
	if numEndpoints != 1 {
		t.Errorf("Found wrong number of endpoints after svc4 creation: expected %d, got %d", 1, numEndpoints)
	}

	// In large-cluster mode, if we delete a service, it will not re-sync its chains
	// but it will not delete them immediately either.
	fp.lastIPTablesCleanup = time.Now()
	fp.OnServiceDelete(svc4)
	fp.syncProxyRules()

	svc4Endpoint, numEndpoints, _ = countEndpointsAndComments(fp.iptablesData.String(), "10.4.0.1")
	assert.Equal(t, "", svc4Endpoint, "svc4 endpoint was still created!")
	// should only sync svc4, and shouldn't output its endpoints
	if numEndpoints != 0 {
		t.Errorf("Found wrong number of endpoints after service deletion: expected %d, got %d", 0, numEndpoints)
	}
	assert.NotContains(t, fp.iptablesData.String(), "-X ", "iptables data unexpectedly contains chain deletions")

	// But resyncing after a long-enough delay will delete the stale chains
	fp.lastIPTablesCleanup = time.Now().Add(-fp.syncPeriod).Add(-1)
	fp.syncProxyRules()

	svc4Endpoint, numEndpoints, _ = countEndpointsAndComments(fp.iptablesData.String(), "10.4.0.1")
	assert.Equal(t, "", svc4Endpoint, "svc4 endpoint was still created!")
	if numEndpoints != 0 {
		t.Errorf("Found wrong number of endpoints after delayed resync: expected %d, got %d", 0, numEndpoints)
	}
	assert.Contains(t, fp.iptablesData.String(), "-X KUBE-SVC-EBDQOQU5SJFXRIL3", "iptables data does not contain chain deletion")
	assert.Contains(t, fp.iptablesData.String(), "-X KUBE-SEP-SU5STNODRYEWJAUF", "iptables data does not contain endpoint deletions")

	// force a full sync and count
	fp.forceSyncProxyRules()
	_, numEndpoints, _ = countEndpointsAndComments(fp.iptablesData.String(), "10.0.0.0")
	if numEndpoints != expectedEndpoints {
		t.Errorf("Found wrong number of endpoints: expected %d, got %d", expectedEndpoints, numEndpoints)
	}
}

// Test calling syncProxyRules() multiple times with various changes
func TestSyncProxyRulesRepeated(t *testing.T) {
	logger, _ := klogtesting.NewTestContext(t)
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)
	metrics.RegisterMetrics(kubeproxyconfig.ProxyModeIPTables)
	defer legacyregistry.Reset()

	// Create initial state
	var svc2 *v1.Service

	makeServiceMap(fp,
		makeTestService("ns1", "svc1", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "172.30.0.41"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p80",
				Port:     80,
				Protocol: v1.ProtocolTCP,
			}}
		}),
		makeTestService("ns2", "svc2", func(svc *v1.Service) {
			svc2 = svc
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "172.30.0.42"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p8080",
				Port:     8080,
				Protocol: v1.ProtocolTCP,
			}}
		}),
	)

	populateEndpointSlices(fp,
		makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{"10.0.1.1"},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p80"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
		makeTestEndpointSlice("ns2", "svc2", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{"10.0.2.1"},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p8080"),
				Port:     ptr.To[int32](8080),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	fp.syncProxyRules()

	expected := dedent.Dedent(`
		*filter
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A KUBE-FORWARD -m conntrack --ctstate INVALID -m nfacct --nfacct-name ct_state_invalid_dropped_pkts -j DROP
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
		COMMIT
		*nat
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-POSTROUTING - [0:0]
		:KUBE-SEP-SNQ3ZNILQDEJNDQO - [0:0]
		:KUBE-SEP-UHEGFW77JX3KXTOV - [0:0]
		:KUBE-SVC-2VJB64SDSIJUP5T6 - [0:0]
		:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns2/svc2:p8080 cluster IP" -m tcp -p tcp -d 172.30.0.42 --dport 8080 -j KUBE-SVC-2VJB64SDSIJUP5T6
		-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
		-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
		-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
		-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
		-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
		-A KUBE-SEP-SNQ3ZNILQDEJNDQO -m comment --comment ns1/svc1:p80 -s 10.0.1.1 -j KUBE-MARK-MASQ
		-A KUBE-SEP-SNQ3ZNILQDEJNDQO -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.0.1.1:80
		-A KUBE-SEP-UHEGFW77JX3KXTOV -m comment --comment ns2/svc2:p8080 -s 10.0.2.1 -j KUBE-MARK-MASQ
		-A KUBE-SEP-UHEGFW77JX3KXTOV -m comment --comment ns2/svc2:p8080 -m tcp -p tcp -j DNAT --to-destination 10.0.2.1:8080
		-A KUBE-SVC-2VJB64SDSIJUP5T6 -m comment --comment "ns2/svc2:p8080 cluster IP" -m tcp -p tcp -d 172.30.0.42 --dport 8080 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-2VJB64SDSIJUP5T6 -m comment --comment "ns2/svc2:p8080 -> 10.0.2.1:8080" -j KUBE-SEP-UHEGFW77JX3KXTOV
		-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 -> 10.0.1.1:80" -j KUBE-SEP-SNQ3ZNILQDEJNDQO
		COMMIT
		`)
	assertIPTablesRulesEqual(t, getLine(), true, expected, fp.iptablesData.String())

	rulesSynced := countRules(logger, utiliptables.TableNAT, expected)
	rulesSyncedMetric := countRulesFromLastSyncMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesSyncedMetric != rulesSynced {
		t.Errorf("metric shows %d rules synced but iptables data shows %d", rulesSyncedMetric, rulesSynced)
	}

	rulesTotal := rulesSynced
	rulesTotalMetric := countRulesFromMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesTotalMetric != rulesTotal {
		t.Errorf("metric shows %d rules total but expected %d", rulesTotalMetric, rulesTotal)
	}

	// Add a new service and its endpoints. (This will only sync the SVC and SEP rules
	// for the new service, not the existing ones.)
	makeServiceMap(fp,
		makeTestService("ns3", "svc3", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "172.30.0.43"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p80",
				Port:     80,
				Protocol: v1.ProtocolTCP,
			}}
		}),
	)
	var eps3 *discovery.EndpointSlice
	populateEndpointSlices(fp,
		makeTestEndpointSlice("ns3", "svc3", 1, func(eps *discovery.EndpointSlice) {
			eps3 = eps
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{"10.0.3.1"},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p80"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)
	fp.syncProxyRules()

	expected = dedent.Dedent(`
		*filter
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A KUBE-FORWARD -m conntrack --ctstate INVALID -m nfacct --nfacct-name ct_state_invalid_dropped_pkts -j DROP
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
		COMMIT
		*nat
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-POSTROUTING - [0:0]
		:KUBE-SEP-BSWRHOQ77KEXZLNL - [0:0]
		:KUBE-SVC-X27LE4BHSL4DOUIK - [0:0]
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns2/svc2:p8080 cluster IP" -m tcp -p tcp -d 172.30.0.42 --dport 8080 -j KUBE-SVC-2VJB64SDSIJUP5T6
		-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 -j KUBE-SVC-X27LE4BHSL4DOUIK
		-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
		-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
		-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
		-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
		-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
		-A KUBE-SEP-BSWRHOQ77KEXZLNL -m comment --comment ns3/svc3:p80 -s 10.0.3.1 -j KUBE-MARK-MASQ
		-A KUBE-SEP-BSWRHOQ77KEXZLNL -m comment --comment ns3/svc3:p80 -m tcp -p tcp -j DNAT --to-destination 10.0.3.1:80
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 -> 10.0.3.1:80" -j KUBE-SEP-BSWRHOQ77KEXZLNL
		COMMIT
		`)
	assertIPTablesRulesEqual(t, getLine(), false, expected, fp.iptablesData.String())

	rulesSynced = countRules(logger, utiliptables.TableNAT, expected)
	rulesSyncedMetric = countRulesFromLastSyncMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesSyncedMetric != rulesSynced {
		t.Errorf("metric shows %d rules synced but iptables data shows %d", rulesSyncedMetric, rulesSynced)
	}

	// We added 1 KUBE-SERVICES rule, 2 KUBE-SVC-X27LE4BHSL4DOUIK rules, and 2
	// KUBE-SEP-BSWRHOQ77KEXZLNL rules.
	rulesTotal += 5
	rulesTotalMetric = countRulesFromMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesTotalMetric != rulesTotal {
		t.Errorf("metric shows %d rules total but expected %d", rulesTotalMetric, rulesTotal)
	}

	// Delete a service. (Won't update the other services.)
	fp.OnServiceDelete(svc2)
	fp.syncProxyRules()

	expected = dedent.Dedent(`
		*filter
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A KUBE-FORWARD -m conntrack --ctstate INVALID -m nfacct --nfacct-name ct_state_invalid_dropped_pkts -j DROP
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
		COMMIT
		*nat
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-POSTROUTING - [0:0]
		:KUBE-SEP-UHEGFW77JX3KXTOV - [0:0]
		:KUBE-SVC-2VJB64SDSIJUP5T6 - [0:0]
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 -j KUBE-SVC-X27LE4BHSL4DOUIK
		-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
		-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
		-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
		-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
		-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
		-X KUBE-SEP-UHEGFW77JX3KXTOV
		-X KUBE-SVC-2VJB64SDSIJUP5T6
		COMMIT
		`)
	assertIPTablesRulesEqual(t, getLine(), false, expected, fp.iptablesData.String())

	rulesSynced = countRules(logger, utiliptables.TableNAT, expected)
	rulesSyncedMetric = countRulesFromLastSyncMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesSyncedMetric != rulesSynced {
		t.Errorf("metric shows %d rules synced but iptables data shows %d", rulesSyncedMetric, rulesSynced)
	}

	// We deleted 1 KUBE-SERVICES rule, 2 KUBE-SVC-2VJB64SDSIJUP5T6 rules, and 2
	// KUBE-SEP-UHEGFW77JX3KXTOV rules
	rulesTotal -= 5
	rulesTotalMetric = countRulesFromMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesTotalMetric != rulesTotal {
		t.Errorf("metric shows %d rules total but expected %d", rulesTotalMetric, rulesTotal)
	}

	// Add a service, sync, then add its endpoints. (The first sync will be a no-op other
	// than adding the REJECT rule. The second sync will create the new service.)
	var svc4 *v1.Service
	makeServiceMap(fp,
		makeTestService("ns4", "svc4", func(svc *v1.Service) {
			svc4 = svc
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "172.30.0.44"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p80",
				Port:     80,
				Protocol: v1.ProtocolTCP,
			}}
		}),
	)
	fp.syncProxyRules()
	expected = dedent.Dedent(`
		*filter
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 has no endpoints" -m tcp -p tcp -d 172.30.0.44 --dport 80 -j REJECT
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A KUBE-FORWARD -m conntrack --ctstate INVALID -m nfacct --nfacct-name ct_state_invalid_dropped_pkts -j DROP
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
		COMMIT
		*nat
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-POSTROUTING - [0:0]
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 -j KUBE-SVC-X27LE4BHSL4DOUIK
		-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
		-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
		-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
		-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
		-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
		COMMIT
		`)
	assertIPTablesRulesEqual(t, getLine(), false, expected, fp.iptablesData.String())

	rulesSynced = countRules(logger, utiliptables.TableNAT, expected)
	rulesSyncedMetric = countRulesFromLastSyncMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesSyncedMetric != rulesSynced {
		t.Errorf("metric shows %d rules synced but iptables data shows %d", rulesSyncedMetric, rulesSynced)
	}

	// The REJECT rule is in "filter", not NAT, so the number of NAT rules hasn't
	// changed.
	rulesTotalMetric = countRulesFromMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesTotalMetric != rulesTotal {
		t.Errorf("metric shows %d rules total but expected %d", rulesTotalMetric, rulesTotal)
	}

	populateEndpointSlices(fp,
		makeTestEndpointSlice("ns4", "svc4", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{"10.0.4.1"},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p80"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)
	fp.syncProxyRules()
	expected = dedent.Dedent(`
		*filter
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A KUBE-FORWARD -m conntrack --ctstate INVALID -m nfacct --nfacct-name ct_state_invalid_dropped_pkts -j DROP
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
		COMMIT
		*nat
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-POSTROUTING - [0:0]
		:KUBE-SEP-AYCN5HPXMIRJNJXU - [0:0]
		:KUBE-SVC-4SW47YFZTEDKD3PK - [0:0]
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 -j KUBE-SVC-X27LE4BHSL4DOUIK
		-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 -j KUBE-SVC-4SW47YFZTEDKD3PK
		-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
		-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
		-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
		-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
		-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
		-A KUBE-SEP-AYCN5HPXMIRJNJXU -m comment --comment ns4/svc4:p80 -s 10.0.4.1 -j KUBE-MARK-MASQ
		-A KUBE-SEP-AYCN5HPXMIRJNJXU -m comment --comment ns4/svc4:p80 -m tcp -p tcp -j DNAT --to-destination 10.0.4.1:80
		-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment "ns4/svc4:p80 -> 10.0.4.1:80" -j KUBE-SEP-AYCN5HPXMIRJNJXU
		COMMIT
		`)
	assertIPTablesRulesEqual(t, getLine(), false, expected, fp.iptablesData.String())

	rulesSynced = countRules(logger, utiliptables.TableNAT, expected)
	rulesSyncedMetric = countRulesFromLastSyncMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesSyncedMetric != rulesSynced {
		t.Errorf("metric shows %d rules synced but iptables data shows %d", rulesSyncedMetric, rulesSynced)
	}

	// We added 1 KUBE-SERVICES rule, 2 KUBE-SVC-4SW47YFZTEDKD3PK rules, and
	// 2 KUBE-SEP-AYCN5HPXMIRJNJXU rules
	rulesTotal += 5
	rulesTotalMetric = countRulesFromMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesTotalMetric != rulesTotal {
		t.Errorf("metric shows %d rules total but expected %d", rulesTotalMetric, rulesTotal)
	}

	// Change an endpoint of an existing service. This will cause its SVC and SEP
	// chains to be rewritten.
	eps3update := eps3.DeepCopy()
	eps3update.Endpoints[0].Addresses[0] = "10.0.3.2"
	fp.OnEndpointSliceUpdate(eps3, eps3update)
	fp.syncProxyRules()

	expected = dedent.Dedent(`
		*filter
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A KUBE-FORWARD -m conntrack --ctstate INVALID -m nfacct --nfacct-name ct_state_invalid_dropped_pkts -j DROP
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
		COMMIT
		*nat
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-POSTROUTING - [0:0]
		:KUBE-SEP-BSWRHOQ77KEXZLNL - [0:0]
		:KUBE-SEP-DKCFIS26GWF2WLWC - [0:0]
		:KUBE-SVC-X27LE4BHSL4DOUIK - [0:0]
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 -j KUBE-SVC-X27LE4BHSL4DOUIK
		-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 -j KUBE-SVC-4SW47YFZTEDKD3PK
		-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
		-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
		-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
		-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
		-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
		-A KUBE-SEP-DKCFIS26GWF2WLWC -m comment --comment ns3/svc3:p80 -s 10.0.3.2 -j KUBE-MARK-MASQ
		-A KUBE-SEP-DKCFIS26GWF2WLWC -m comment --comment ns3/svc3:p80 -m tcp -p tcp -j DNAT --to-destination 10.0.3.2:80
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 -> 10.0.3.2:80" -j KUBE-SEP-DKCFIS26GWF2WLWC
		-X KUBE-SEP-BSWRHOQ77KEXZLNL
		COMMIT
		`)
	assertIPTablesRulesEqual(t, getLine(), false, expected, fp.iptablesData.String())

	rulesSynced = countRules(logger, utiliptables.TableNAT, expected)
	rulesSyncedMetric = countRulesFromLastSyncMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesSyncedMetric != rulesSynced {
		t.Errorf("metric shows %d rules synced but iptables data shows %d", rulesSyncedMetric, rulesSynced)
	}

	// We rewrote existing rules but did not change the overall number of rules.
	rulesTotalMetric = countRulesFromMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesTotalMetric != rulesTotal {
		t.Errorf("metric shows %d rules total but expected %d", rulesTotalMetric, rulesTotal)
	}

	// Add an endpoint to a service. This will cause its SVC and SEP chains to be rewritten.
	eps3update2 := eps3update.DeepCopy()
	eps3update2.Endpoints = append(eps3update2.Endpoints, discovery.Endpoint{Addresses: []string{"10.0.3.3"}})
	fp.OnEndpointSliceUpdate(eps3update, eps3update2)
	fp.syncProxyRules()

	expected = dedent.Dedent(`
		*filter
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A KUBE-FORWARD -m conntrack --ctstate INVALID -m nfacct --nfacct-name ct_state_invalid_dropped_pkts -j DROP
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
		COMMIT
		*nat
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-POSTROUTING - [0:0]
		:KUBE-SEP-DKCFIS26GWF2WLWC - [0:0]
		:KUBE-SEP-JVVZVJ7BSEPPRNBS - [0:0]
		:KUBE-SVC-X27LE4BHSL4DOUIK - [0:0]
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 -j KUBE-SVC-X27LE4BHSL4DOUIK
		-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 -j KUBE-SVC-4SW47YFZTEDKD3PK
		-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
		-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
		-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
		-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
		-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
		-A KUBE-SEP-DKCFIS26GWF2WLWC -m comment --comment ns3/svc3:p80 -s 10.0.3.2 -j KUBE-MARK-MASQ
		-A KUBE-SEP-DKCFIS26GWF2WLWC -m comment --comment ns3/svc3:p80 -m tcp -p tcp -j DNAT --to-destination 10.0.3.2:80
		-A KUBE-SEP-JVVZVJ7BSEPPRNBS -m comment --comment ns3/svc3:p80 -s 10.0.3.3 -j KUBE-MARK-MASQ
		-A KUBE-SEP-JVVZVJ7BSEPPRNBS -m comment --comment ns3/svc3:p80 -m tcp -p tcp -j DNAT --to-destination 10.0.3.3:80
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 -> 10.0.3.2:80" -m statistic --mode random --probability 0.5000000000 -j KUBE-SEP-DKCFIS26GWF2WLWC
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 -> 10.0.3.3:80" -j KUBE-SEP-JVVZVJ7BSEPPRNBS
		COMMIT
		`)
	assertIPTablesRulesEqual(t, getLine(), false, expected, fp.iptablesData.String())

	rulesSynced = countRules(logger, utiliptables.TableNAT, expected)
	rulesSyncedMetric = countRulesFromLastSyncMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesSyncedMetric != rulesSynced {
		t.Errorf("metric shows %d rules synced but iptables data shows %d", rulesSyncedMetric, rulesSynced)
	}

	// We added 2 KUBE-SEP-JVVZVJ7BSEPPRNBS rules and 1 KUBE-SVC-X27LE4BHSL4DOUIK rule
	// jumping to the new SEP chain. The other rules related to svc3 got rewritten,
	// but that does not change the count of rules.
	rulesTotal += 3
	rulesTotalMetric = countRulesFromMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesTotalMetric != rulesTotal {
		t.Errorf("metric shows %d rules total but expected %d", rulesTotalMetric, rulesTotal)
	}

	// Sync with no new changes... This will not rewrite any SVC or SEP chains
	fp.syncProxyRules()

	expected = dedent.Dedent(`
		*filter
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A KUBE-FORWARD -m conntrack --ctstate INVALID -m nfacct --nfacct-name ct_state_invalid_dropped_pkts -j DROP
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
		COMMIT
		*nat
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-POSTROUTING - [0:0]
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 -j KUBE-SVC-X27LE4BHSL4DOUIK
		-A KUBE-SERVICES -m comment --comment "ns4/svc4:p80 cluster IP" -m tcp -p tcp -d 172.30.0.44 --dport 80 -j KUBE-SVC-4SW47YFZTEDKD3PK
		-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
		-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
		-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
		-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
		-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
		COMMIT
		`)
	assertIPTablesRulesEqual(t, getLine(), false, expected, fp.iptablesData.String())

	rulesSynced = countRules(logger, utiliptables.TableNAT, expected)
	rulesSyncedMetric = countRulesFromLastSyncMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesSyncedMetric != rulesSynced {
		t.Errorf("metric shows %d rules synced but iptables data shows %d", rulesSyncedMetric, rulesSynced)
	}

	// (No changes)
	rulesTotalMetric = countRulesFromMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesTotalMetric != rulesTotal {
		t.Errorf("metric shows %d rules total but expected %d", rulesTotalMetric, rulesTotal)
	}

	// Now force a partial resync error and ensure that it recovers correctly
	if fp.needFullSync {
		t.Fatalf("Proxier unexpectedly already needs a full sync?")
	}
	partialRestoreFailures, err := testutil.GetCounterMetricValue(metrics.IPTablesPartialRestoreFailuresTotal.WithLabelValues(string(fp.ipFamily)))
	if err != nil {
		t.Fatalf("Could not get partial restore failures metric: %v", err)
	}
	if partialRestoreFailures != 0.0 {
		t.Errorf("Already did a partial resync? Something failed earlier!")
	}

	// Add a rule jumping from svc3's service chain to svc4's endpoint, then try to
	// delete svc4. This will fail because the partial resync won't rewrite svc3's
	// rules and so the partial restore would leave a dangling jump from there to
	// svc4's endpoint. The proxier will then queue a full resync in response to the
	// partial resync failure, and the full resync will succeed (since it will rewrite
	// svc3's rules as well).
	//
	// This is an absurd scenario, but it has to be; partial resync failures are
	// supposed to be impossible; if we knew of any non-absurd scenario that would
	// cause such a failure, then that would be a bug and we would fix it.
	if _, err := fp.iptables.ChainExists(utiliptables.TableNAT, utiliptables.Chain("KUBE-SEP-AYCN5HPXMIRJNJXU")); err != nil {
		t.Fatalf("svc4's endpoint chain unexpected already does not exist!")
	}
	if _, err := fp.iptables.EnsureRule(utiliptables.Append, utiliptables.TableNAT, utiliptables.Chain("KUBE-SVC-X27LE4BHSL4DOUIK"), "-j", "KUBE-SEP-AYCN5HPXMIRJNJXU"); err != nil {
		t.Fatalf("Could not add bad iptables rule: %v", err)
	}

	fp.OnServiceDelete(svc4)
	fp.syncProxyRules()

	if _, err := fp.iptables.ChainExists(utiliptables.TableNAT, utiliptables.Chain("KUBE-SEP-AYCN5HPXMIRJNJXU")); err != nil {
		t.Errorf("svc4's endpoint chain was successfully deleted despite dangling references!")
	}
	if !fp.needFullSync {
		t.Errorf("Proxier did not fail on previous partial resync?")
	}
	updatedPartialRestoreFailures, err := testutil.GetCounterMetricValue(metrics.IPTablesPartialRestoreFailuresTotal.WithLabelValues(string(fp.ipFamily)))
	if err != nil {
		t.Errorf("Could not get partial restore failures metric: %v", err)
	}
	if updatedPartialRestoreFailures != partialRestoreFailures+1.0 {
		t.Errorf("Partial restore failures metric was not incremented after failed partial resync (expected %.02f, got %.02f)", partialRestoreFailures+1.0, updatedPartialRestoreFailures)
	}

	// On retry we should do a full resync, which should succeed (and delete svc4)
	fp.syncProxyRules()

	expected = dedent.Dedent(`
		*filter
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A KUBE-FORWARD -m conntrack --ctstate INVALID -m nfacct --nfacct-name ct_state_invalid_dropped_pkts -j DROP
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
		COMMIT
		*nat
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-POSTROUTING - [0:0]
		:KUBE-SEP-AYCN5HPXMIRJNJXU - [0:0]
		:KUBE-SEP-DKCFIS26GWF2WLWC - [0:0]
		:KUBE-SEP-JVVZVJ7BSEPPRNBS - [0:0]
		:KUBE-SEP-SNQ3ZNILQDEJNDQO - [0:0]
		:KUBE-SVC-4SW47YFZTEDKD3PK - [0:0]
		:KUBE-SVC-X27LE4BHSL4DOUIK - [0:0]
		:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 -j KUBE-SVC-X27LE4BHSL4DOUIK
		-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
		-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
		-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
		-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
		-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
		-A KUBE-SEP-DKCFIS26GWF2WLWC -m comment --comment ns3/svc3:p80 -s 10.0.3.2 -j KUBE-MARK-MASQ
		-A KUBE-SEP-DKCFIS26GWF2WLWC -m comment --comment ns3/svc3:p80 -m tcp -p tcp -j DNAT --to-destination 10.0.3.2:80
		-A KUBE-SEP-JVVZVJ7BSEPPRNBS -m comment --comment ns3/svc3:p80 -s 10.0.3.3 -j KUBE-MARK-MASQ
		-A KUBE-SEP-JVVZVJ7BSEPPRNBS -m comment --comment ns3/svc3:p80 -m tcp -p tcp -j DNAT --to-destination 10.0.3.3:80
		-A KUBE-SEP-SNQ3ZNILQDEJNDQO -m comment --comment ns1/svc1:p80 -s 10.0.1.1 -j KUBE-MARK-MASQ
		-A KUBE-SEP-SNQ3ZNILQDEJNDQO -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.0.1.1:80
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 cluster IP" -m tcp -p tcp -d 172.30.0.43 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 -> 10.0.3.2:80" -m statistic --mode random --probability 0.5000000000 -j KUBE-SEP-DKCFIS26GWF2WLWC
		-A KUBE-SVC-X27LE4BHSL4DOUIK -m comment --comment "ns3/svc3:p80 -> 10.0.3.3:80" -j KUBE-SEP-JVVZVJ7BSEPPRNBS
		-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 -> 10.0.1.1:80" -j KUBE-SEP-SNQ3ZNILQDEJNDQO
		-X KUBE-SEP-AYCN5HPXMIRJNJXU
		-X KUBE-SVC-4SW47YFZTEDKD3PK
		COMMIT
		`)
	assertIPTablesRulesEqual(t, getLine(), false, expected, fp.iptablesData.String())

	rulesSynced = countRules(logger, utiliptables.TableNAT, expected)
	rulesSyncedMetric = countRulesFromLastSyncMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesSyncedMetric != rulesSynced {
		t.Errorf("metric shows %d rules synced but iptables data shows %d", rulesSyncedMetric, rulesSynced)
	}

	// We deleted 1 KUBE-SERVICES rule, 2 KUBE-SVC-4SW47YFZTEDKD3PK rules, and 2
	// KUBE-SEP-AYCN5HPXMIRJNJXU rules
	rulesTotal -= 5
	rulesTotalMetric = countRulesFromMetric(logger, utiliptables.TableNAT, string(fp.ipFamily))
	if rulesTotalMetric != rulesTotal {
		t.Errorf("metric shows %d rules total but expected %d", rulesTotalMetric, rulesTotal)
	}
}

func TestNoEndpointsMetric(t *testing.T) {
	type endpoint struct {
		ip       string
		nodeName string
	}

	metrics.RegisterMetrics(kubeproxyconfig.ProxyModeIPTables)
	testCases := []struct {
		name                                                string
		internalTrafficPolicy                               *v1.ServiceInternalTrafficPolicy
		externalTrafficPolicy                               v1.ServiceExternalTrafficPolicy
		endpoints                                           []endpoint
		expectedSyncProxyRulesNoLocalEndpointsTotalInternal int
		expectedSyncProxyRulesNoLocalEndpointsTotalExternal int
	}{
		{
			name:                  "internalTrafficPolicy is set and there are local endpoints",
			internalTrafficPolicy: ptr.To(v1.ServiceInternalTrafficPolicyLocal),
			endpoints: []endpoint{
				{"10.0.1.1", testNodeName},
				{"10.0.1.2", "node1"},
				{"10.0.1.3", "node2"},
			},
		},
		{
			name:                  "externalTrafficPolicy is set and there are local endpoints",
			externalTrafficPolicy: v1.ServiceExternalTrafficPolicyLocal,
			endpoints: []endpoint{
				{"10.0.1.1", testNodeName},
				{"10.0.1.2", "node1"},
				{"10.0.1.3", "node2"},
			},
		},
		{
			name:                  "both policies are set and there are local endpoints",
			internalTrafficPolicy: ptr.To(v1.ServiceInternalTrafficPolicyLocal),
			externalTrafficPolicy: v1.ServiceExternalTrafficPolicyLocal,
			endpoints: []endpoint{
				{"10.0.1.1", testNodeName},
				{"10.0.1.2", "node1"},
				{"10.0.1.3", "node2"},
			},
		},
		{
			name:                  "internalTrafficPolicy is set and there are no local endpoints",
			internalTrafficPolicy: ptr.To(v1.ServiceInternalTrafficPolicyLocal),
			endpoints: []endpoint{
				{"10.0.1.1", "node0"},
				{"10.0.1.2", "node1"},
				{"10.0.1.3", "node2"},
			},
			expectedSyncProxyRulesNoLocalEndpointsTotalInternal: 1,
		},
		{
			name:                  "externalTrafficPolicy is set and there are no local endpoints",
			externalTrafficPolicy: v1.ServiceExternalTrafficPolicyLocal,
			endpoints: []endpoint{
				{"10.0.1.1", "node0"},
				{"10.0.1.2", "node1"},
				{"10.0.1.3", "node2"},
			},
			expectedSyncProxyRulesNoLocalEndpointsTotalExternal: 1,
		},
		{
			name:                  "both policies are set and there are no local endpoints",
			internalTrafficPolicy: ptr.To(v1.ServiceInternalTrafficPolicyLocal),
			externalTrafficPolicy: v1.ServiceExternalTrafficPolicyLocal,
			endpoints: []endpoint{
				{"10.0.1.1", "node0"},
				{"10.0.1.2", "node1"},
				{"10.0.1.3", "node2"},
			},
			expectedSyncProxyRulesNoLocalEndpointsTotalInternal: 1,
			expectedSyncProxyRulesNoLocalEndpointsTotalExternal: 1,
		},
		{
			name:                  "both policies are set and there are no endpoints at all",
			internalTrafficPolicy: ptr.To(v1.ServiceInternalTrafficPolicyLocal),
			externalTrafficPolicy: v1.ServiceExternalTrafficPolicyLocal,
			endpoints:             []endpoint{},
			expectedSyncProxyRulesNoLocalEndpointsTotalInternal: 0,
			expectedSyncProxyRulesNoLocalEndpointsTotalExternal: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ipt := iptablestest.NewFake()
			fp := NewFakeProxier(ipt)
			fp.OnServiceSynced()
			fp.OnEndpointSlicesSynced()

			serviceName := "svc1"
			namespaceName := "ns1"

			svc := &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
				Spec: v1.ServiceSpec{
					ClusterIP: "172.30.1.1",
					Selector:  map[string]string{"foo": "bar"},
					Ports:     []v1.ServicePort{{Name: "", Port: 80, Protocol: v1.ProtocolTCP, NodePort: 123}},
				},
			}
			if tc.internalTrafficPolicy != nil {
				svc.Spec.InternalTrafficPolicy = tc.internalTrafficPolicy
			}
			if tc.externalTrafficPolicy != "" {
				svc.Spec.Type = v1.ServiceTypeNodePort
				svc.Spec.ExternalTrafficPolicy = tc.externalTrafficPolicy
			}

			fp.OnServiceAdd(svc)

			endpointSlice := &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("%s-1", serviceName),
					Namespace: namespaceName,
					Labels:    map[string]string{discovery.LabelServiceName: serviceName},
				},
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Port:     ptr.To[int32](80),
					Protocol: ptr.To(v1.ProtocolTCP),
				}},
				AddressType: discovery.AddressTypeIPv4,
			}
			for _, ep := range tc.endpoints {
				endpointSlice.Endpoints = append(endpointSlice.Endpoints, discovery.Endpoint{
					Addresses:  []string{ep.ip},
					Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
					NodeName:   ptr.To(ep.nodeName),
				})
			}

			fp.OnEndpointSliceAdd(endpointSlice)
			fp.syncProxyRules()
			syncProxyRulesNoLocalEndpointsTotalInternal, err := testutil.GetGaugeMetricValue(metrics.SyncProxyRulesNoLocalEndpointsTotal.WithLabelValues("internal", string(fp.ipFamily)))
			if err != nil {
				t.Errorf("failed to get %s value, err: %v", metrics.SyncProxyRulesNoLocalEndpointsTotal.Name, err)
			}

			if tc.expectedSyncProxyRulesNoLocalEndpointsTotalInternal != int(syncProxyRulesNoLocalEndpointsTotalInternal) {
				t.Errorf("sync_proxy_rules_no_endpoints_total metric mismatch(internal): got=%d, expected %d", int(syncProxyRulesNoLocalEndpointsTotalInternal), tc.expectedSyncProxyRulesNoLocalEndpointsTotalInternal)
			}

			syncProxyRulesNoLocalEndpointsTotalExternal, err := testutil.GetGaugeMetricValue(metrics.SyncProxyRulesNoLocalEndpointsTotal.WithLabelValues("external", string(fp.ipFamily)))
			if err != nil {
				t.Errorf("failed to get %s value(external), err: %v", metrics.SyncProxyRulesNoLocalEndpointsTotal.Name, err)
			}

			if tc.expectedSyncProxyRulesNoLocalEndpointsTotalExternal != int(syncProxyRulesNoLocalEndpointsTotalExternal) {
				t.Errorf("sync_proxy_rules_no_endpoints_total metric mismatch(internal): got=%d, expected %d", int(syncProxyRulesNoLocalEndpointsTotalExternal), tc.expectedSyncProxyRulesNoLocalEndpointsTotalExternal)
			}
		})
	}
}

func TestLoadBalancerIngressRouteTypeProxy(t *testing.T) {
	testCases := []struct {
		name          string
		ipModeEnabled bool
		svcIP         string
		svcLBIP       string
		ipMode        *v1.LoadBalancerIPMode
		expectedRule  bool
	}{
		/* LoadBalancerIPMode disabled */
		{
			name:          "LoadBalancerIPMode disabled, ipMode Proxy",
			ipModeEnabled: false,
			svcIP:         "10.20.30.41",
			svcLBIP:       "1.2.3.4",
			ipMode:        ptr.To(v1.LoadBalancerIPModeProxy),
			expectedRule:  true,
		},
		{
			name:          "LoadBalancerIPMode disabled, ipMode VIP",
			ipModeEnabled: false,
			svcIP:         "10.20.30.42",
			svcLBIP:       "1.2.3.5",
			ipMode:        ptr.To(v1.LoadBalancerIPModeVIP),
			expectedRule:  true,
		},
		{
			name:          "LoadBalancerIPMode disabled, ipMode nil",
			ipModeEnabled: false,
			svcIP:         "10.20.30.43",
			svcLBIP:       "1.2.3.6",
			ipMode:        nil,
			expectedRule:  true,
		},
		/* LoadBalancerIPMode enabled */
		{
			name:          "LoadBalancerIPMode enabled, ipMode Proxy",
			ipModeEnabled: true,
			svcIP:         "10.20.30.41",
			svcLBIP:       "1.2.3.4",
			ipMode:        ptr.To(v1.LoadBalancerIPModeProxy),
			expectedRule:  false,
		},
		{
			name:          "LoadBalancerIPMode enabled, ipMode VIP",
			ipModeEnabled: true,
			svcIP:         "10.20.30.42",
			svcLBIP:       "1.2.3.5",
			ipMode:        ptr.To(v1.LoadBalancerIPModeVIP),
			expectedRule:  true,
		},
		{
			name:          "LoadBalancerIPMode enabled, ipMode nil",
			ipModeEnabled: true,
			svcIP:         "10.20.30.43",
			svcLBIP:       "1.2.3.6",
			ipMode:        nil,
			expectedRule:  true,
		},
	}

	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			if !testCase.ipModeEnabled {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.31"))
			}
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.LoadBalancerIPMode, testCase.ipModeEnabled)
			ipt := iptablestest.NewFake()
			fp := NewFakeProxier(ipt)
			makeServiceMap(fp,
				makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
					svc.Spec.Type = "LoadBalancer"
					svc.Spec.ClusterIP = testCase.svcIP
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     svcPortName.Port,
						Port:     int32(svcPort),
						Protocol: v1.ProtocolTCP,
						NodePort: int32(svcNodePort),
					}}
					svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
						IP:     testCase.svcLBIP,
						IPMode: testCase.ipMode,
					}}
				}),
			)

			populateEndpointSlices(fp,
				makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv4
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"10.180.0.1"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     ptr.To("p80"),
						Port:     ptr.To[int32](80),
						Protocol: ptr.To(v1.ProtocolTCP),
					}}
				}),
			)

			fp.syncProxyRules()

			c, _ := ipt.Dump.GetChain(utiliptables.TableNAT, kubeServicesChain)
			ruleExists := false
			for _, r := range c.Rules {
				if r.DestinationAddress != nil && r.DestinationAddress.Value == testCase.svcLBIP {
					ruleExists = true
				}
			}
			if ruleExists != testCase.expectedRule {
				t.Errorf("unexpected rule for %s", testCase.svcLBIP)
			}
		})
	}
}

// TestBadIPs tests that "bad" IPs and CIDRs in Services/Endpoints are rewritten to
// be "good" in the input provided to iptables-restore
func TestBadIPs(t *testing.T) {
	ipt := iptablestest.NewFake()
	fp := NewFakeProxier(ipt)
	metrics.RegisterMetrics(kubeproxyconfig.ProxyModeIPTables)

	makeServiceMap(fp,
		makeTestService("ns1", "svc1", func(svc *v1.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ClusterIP = "172.30.0.041"
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p80",
				Port:     80,
				Protocol: v1.ProtocolTCP,
				NodePort: 3001,
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: "1.2.3.004",
			}}
			svc.Spec.ExternalIPs = []string{"192.168.099.022"}
			svc.Spec.LoadBalancerSourceRanges = []string{"203.0.113.000/025"}
		}),
	)
	populateEndpointSlices(fp,
		makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{"10.180.00.001"},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p80"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	fp.syncProxyRules()

	expected := dedent.Dedent(`
		*filter
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A KUBE-FORWARD -m conntrack --ctstate INVALID -m nfacct --nfacct-name ct_state_invalid_dropped_pkts -j DROP
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
		-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
		-A KUBE-PROXY-FIREWALL -m comment --comment "ns1/svc1:p80 traffic not accepted by KUBE-FW-XPGD46QRK7WJZT7O" -m tcp -p tcp -d 1.2.3.4 --dport 80 -j DROP
		COMMIT
		*nat
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-EXT-XPGD46QRK7WJZT7O - [0:0]
		:KUBE-FW-XPGD46QRK7WJZT7O - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-POSTROUTING - [0:0]
		:KUBE-SEP-SXIVWICOYRO3J4NJ - [0:0]
		:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
		-A KUBE-NODEPORTS -m comment --comment ns1/svc1:p80 -m tcp -p tcp -d 127.0.0.0/8 --dport 3001 -m nfacct --nfacct-name localhost_nps_accepted_pkts -j KUBE-EXT-XPGD46QRK7WJZT7O
		-A KUBE-NODEPORTS -m comment --comment ns1/svc1:p80 -m tcp -p tcp --dport 3001 -j KUBE-EXT-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 external IP" -m tcp -p tcp -d 192.168.99.22 --dport 80 -j KUBE-EXT-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 loadbalancer IP" -m tcp -p tcp -d 1.2.3.4 --dport 80 -j KUBE-FW-XPGD46QRK7WJZT7O
		-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
		-A KUBE-EXT-XPGD46QRK7WJZT7O -m comment --comment "masquerade traffic for ns1/svc1:p80 external destinations" -j KUBE-MARK-MASQ
		-A KUBE-EXT-XPGD46QRK7WJZT7O -j KUBE-SVC-XPGD46QRK7WJZT7O
		-A KUBE-FW-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 loadbalancer IP" -s 203.0.113.0/25 -j KUBE-EXT-XPGD46QRK7WJZT7O
		-A KUBE-FW-XPGD46QRK7WJZT7O -m comment --comment "other traffic to ns1/svc1:p80 will be dropped by KUBE-PROXY-FIREWALL"
		-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000
		-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN
		-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000
		-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE
		-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -s 10.180.0.1 -j KUBE-MARK-MASQ
		-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.1:80
		-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ
		-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 -> 10.180.0.1:80" -j KUBE-SEP-SXIVWICOYRO3J4NJ
		COMMIT
		`)

	assertIPTablesRulesEqual(t, getLine(), true, expected, fp.iptablesData.String())
}
