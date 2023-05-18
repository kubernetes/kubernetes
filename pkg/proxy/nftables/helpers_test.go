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

package nftables

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/danwinship/knftables"
	"github.com/google/go-cmp/cmp"
	"github.com/lithammer/dedent"

	"k8s.io/api/core/v1"
)

// getLine returns a string containing the file and line number of the caller, if
// possible. This is useful in tests with a large number of cases - when something goes
// wrong you can find which case more easily.
func getLine() string {
	_, file, line, ok := runtime.Caller(1)
	if !ok {
		return ""
	}
	return fmt.Sprintf(" (from %s:%d)", file, line)
}

// objectOrder defines the order we sort different types into (higher = earlier); while
// not necessary just for comparison purposes, it's more intuitive in the Diff output to
// see rules/sets/maps before chains/elements.
var objectOrder = map[string]int{
	"table":   10,
	"chain":   9,
	"rule":    8,
	"set":     7,
	"map":     6,
	"element": 5,
	// anything else: 0
}

// sortNFTablesTransaction sorts an nftables transaction into a standard order for comparison
func sortNFTablesTransaction(tx string) string {
	lines := strings.Split(tx, "\n")

	// strip blank lines and comments
	for i := 0; i < len(lines); {
		if lines[i] == "" || lines[i][0] == '#' {
			lines = append(lines[:i], lines[i+1:]...)
		} else {
			i++
		}
	}

	// sort remaining lines
	sort.SliceStable(lines, func(i, j int) bool {
		li := lines[i]
		wi := strings.Split(li, " ")
		lj := lines[j]
		wj := strings.Split(lj, " ")

		// All lines will start with "add OBJECTTYPE ip kube-proxy". Everything
		// except "add table" will have an object name after the table name, and
		// "add table" will have a comment after the table name. So every line
		// should have at least 5 words.
		if len(wi) < 5 || len(wj) < 5 {
			return false
		}

		// Sort by object type first.
		if wi[1] != wj[1] {
			return objectOrder[wi[1]] >= objectOrder[wj[1]]
		}

		// Sort by object name when object type is identical.
		if wi[4] != wj[4] {
			return wi[4] < wj[4]
		}

		// Leave rules in the order they were added in
		if wi[1] == "rule" {
			return false
		}

		// Sort by the whole line when object type and name is identical. (e.g.,
		// individual "add rule" and "add element" lines in a chain/set/map.)
		return li < lj
	})
	return strings.Join(lines, "\n")
}

// diffNFTablesTransaction is a (testable) helper function for assertNFTablesTransactionEqual
func diffNFTablesTransaction(expected, result string) string {
	expected = sortNFTablesTransaction(expected)
	result = sortNFTablesTransaction(result)

	return cmp.Diff(expected, result)
}

// assertNFTablesTransactionEqual asserts that expected and result are equal, ignoring
// irrelevant differences.
func assertNFTablesTransactionEqual(t *testing.T, line string, expected, result string) {
	diff := diffNFTablesTransaction(expected, result)
	if diff != "" {
		t.Errorf("tables do not match%s:\ndiff:\n%s\nfull result: %+v", line, diff, result)
	}
}

// diffNFTablesChain is a (testable) helper function for assertNFTablesChainEqual
func diffNFTablesChain(nft *knftables.Fake, chain, expected string) string {
	expected = strings.TrimSpace(expected)
	result := ""
	if ch := nft.Table.Chains[chain]; ch != nil {
		for i, rule := range ch.Rules {
			if i > 0 {
				result += "\n"
			}
			result += rule.Rule
		}
	}

	return cmp.Diff(expected, result)
}

// assertNFTablesChainEqual asserts that the indicated chain in nft's table contains
// exactly the rules in expected (in that order).
func assertNFTablesChainEqual(t *testing.T, line string, nft *knftables.Fake, chain, expected string) {
	if diff := diffNFTablesChain(nft, chain, expected); diff != "" {
		t.Errorf("rules do not match%s:\ndiff:\n%s", line, diff)
	}
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

func runPacketFlowTests(t *testing.T, line string, nft *knftables.Fake, nodeIPs []string, testCases []packetFlowTest) {
	for _, tc := range testCases {
		t.Logf("Skipping test %s which doesn't work yet", tc.name)
	}
}

// helpers_test unit tests

var testInput = dedent.Dedent(`
	add table ip testing { comment "rules for kube-proxy" ; }

	add chain ip testing forward
	add rule ip testing forward ct state invalid drop
	add chain ip testing mark-for-masquerade
	add rule ip testing mark-for-masquerade mark set mark or 0x4000
	add chain ip testing masquerading
	add rule ip testing masquerading mark and 0x4000 == 0 return
	add rule ip testing masquerading mark set mark xor 0x4000
	add rule ip testing masquerading masquerade fully-random

	add set ip testing firewall { type ipv4_addr . inet_proto . inet_service ; comment "destinations that are subject to LoadBalancerSourceRanges" ; }
	add set ip testing firewall-allow { type ipv4_addr . inet_proto . inet_service . ipv4_addr ; flags interval ; comment "destinations+sources that are allowed by LoadBalancerSourceRanges" ; }
	add chain ip testing firewall-check
	add chain ip testing firewall-allow-check
	add rule ip testing firewall-allow-check ip daddr . meta l4proto . th dport . ip saddr @firewall-allow return
	add rule ip testing firewall-allow-check drop
	add rule ip testing firewall-check ip daddr . meta l4proto . th dport @firewall jump firewall-allow-check

	# svc1
	add chain ip testing service-ULMVA6XW-ns1/svc1/tcp/p80
	add rule ip testing service-ULMVA6XW-ns1/svc1/tcp/p80 ip daddr 172.30.0.41 tcp dport 80 ip saddr != 10.0.0.0/8 jump mark-for-masquerade
	add rule ip testing service-ULMVA6XW-ns1/svc1/tcp/p80 numgen random mod 1 vmap { 0 : goto endpoint-5OJB2KTY-ns1/svc1/tcp/p80__10.180.0.1/80 }

	add chain ip testing endpoint-5OJB2KTY-ns1/svc1/tcp/p80__10.180.0.1/80
	add rule ip testing endpoint-5OJB2KTY-ns1/svc1/tcp/p80__10.180.0.1/80 ip saddr 10.180.0.1 jump mark-for-masquerade
	add rule ip testing endpoint-5OJB2KTY-ns1/svc1/tcp/p80__10.180.0.1/80 meta l4proto tcp dnat to 10.180.0.1:80

	add element ip testing service-ips { 172.30.0.41 . tcp . 80 : goto service-ULMVA6XW-ns1/svc1/tcp/p80 }

	# svc2
	add chain ip testing service-42NFTM6N-ns2/svc2/tcp/p80
	add rule ip testing service-42NFTM6N-ns2/svc2/tcp/p80 ip daddr 172.30.0.42 tcp dport 80 ip saddr != 10.0.0.0/8 jump mark-for-masquerade
	add rule ip testing service-42NFTM6N-ns2/svc2/tcp/p80 numgen random mod 1 vmap { 0 : goto endpoint-SGOXE6O3-ns2/svc2/tcp/p80__10.180.0.2/80 }
	add chain ip testing external-42NFTM6N-ns2/svc2/tcp/p80
	add rule ip testing external-42NFTM6N-ns2/svc2/tcp/p80 ip saddr 10.0.0.0/8 goto service-42NFTM6N-ns2/svc2/tcp/p80 comment "short-circuit pod traffic"
	add rule ip testing external-42NFTM6N-ns2/svc2/tcp/p80 fib saddr type local jump mark-for-masquerade comment "masquerade local traffic"
	add rule ip testing external-42NFTM6N-ns2/svc2/tcp/p80 fib saddr type local goto service-42NFTM6N-ns2/svc2/tcp/p80 comment "short-circuit local traffic"
	add chain ip testing endpoint-SGOXE6O3-ns2/svc2/tcp/p80__10.180.0.2/80
	add rule ip testing endpoint-SGOXE6O3-ns2/svc2/tcp/p80__10.180.0.2/80 ip saddr 10.180.0.2 jump mark-for-masquerade
	add rule ip testing endpoint-SGOXE6O3-ns2/svc2/tcp/p80__10.180.0.2/80 meta l4proto tcp dnat to 10.180.0.2:80

	add element ip testing service-ips { 172.30.0.42 . tcp . 80 : goto service-42NFTM6N-ns2/svc2/tcp/p80 }
	add element ip testing service-ips { 192.168.99.22 . tcp . 80 : goto external-42NFTM6N-ns2/svc2/tcp/p80 }
	add element ip testing service-ips { 1.2.3.4 . tcp . 80 : goto external-42NFTM6N-ns2/svc2/tcp/p80 }
	add element ip testing service-nodeports { tcp . 3001 : goto external-42NFTM6N-ns2/svc2/tcp/p80 }

	add element ip testing no-endpoint-nodeports { tcp . 3001 comment "ns2/svc2:p80" : drop }
	add element ip testing no-endpoint-services { 1.2.3.4 . tcp . 80 comment "ns2/svc2:p80" : drop }
	add element ip testing no-endpoint-services { 192.168.99.22 . tcp . 80 comment "ns2/svc2:p80" : drop }
	`)

var testExpected = dedent.Dedent(`
	add table ip testing { comment "rules for kube-proxy" ; }
	add chain ip testing endpoint-5OJB2KTY-ns1/svc1/tcp/p80__10.180.0.1/80
	add chain ip testing endpoint-SGOXE6O3-ns2/svc2/tcp/p80__10.180.0.2/80
	add chain ip testing external-42NFTM6N-ns2/svc2/tcp/p80
	add chain ip testing firewall-allow-check
	add chain ip testing firewall-check
	add chain ip testing forward
	add chain ip testing mark-for-masquerade
	add chain ip testing masquerading
	add chain ip testing service-42NFTM6N-ns2/svc2/tcp/p80
	add chain ip testing service-ULMVA6XW-ns1/svc1/tcp/p80
	add rule ip testing endpoint-5OJB2KTY-ns1/svc1/tcp/p80__10.180.0.1/80 ip saddr 10.180.0.1 jump mark-for-masquerade
	add rule ip testing endpoint-5OJB2KTY-ns1/svc1/tcp/p80__10.180.0.1/80 meta l4proto tcp dnat to 10.180.0.1:80
	add rule ip testing endpoint-SGOXE6O3-ns2/svc2/tcp/p80__10.180.0.2/80 ip saddr 10.180.0.2 jump mark-for-masquerade
	add rule ip testing endpoint-SGOXE6O3-ns2/svc2/tcp/p80__10.180.0.2/80 meta l4proto tcp dnat to 10.180.0.2:80
	add rule ip testing external-42NFTM6N-ns2/svc2/tcp/p80 ip saddr 10.0.0.0/8 goto service-42NFTM6N-ns2/svc2/tcp/p80 comment "short-circuit pod traffic"
	add rule ip testing external-42NFTM6N-ns2/svc2/tcp/p80 fib saddr type local jump mark-for-masquerade comment "masquerade local traffic"
	add rule ip testing external-42NFTM6N-ns2/svc2/tcp/p80 fib saddr type local goto service-42NFTM6N-ns2/svc2/tcp/p80 comment "short-circuit local traffic"
	add rule ip testing firewall-allow-check ip daddr . meta l4proto . th dport . ip saddr @firewall-allow return
	add rule ip testing firewall-allow-check drop
	add rule ip testing firewall-check ip daddr . meta l4proto . th dport @firewall jump firewall-allow-check
	add rule ip testing forward ct state invalid drop
	add rule ip testing mark-for-masquerade mark set mark or 0x4000
	add rule ip testing masquerading mark and 0x4000 == 0 return
	add rule ip testing masquerading mark set mark xor 0x4000
	add rule ip testing masquerading masquerade fully-random
	add rule ip testing service-42NFTM6N-ns2/svc2/tcp/p80 ip daddr 172.30.0.42 tcp dport 80 ip saddr != 10.0.0.0/8 jump mark-for-masquerade
	add rule ip testing service-42NFTM6N-ns2/svc2/tcp/p80 numgen random mod 1 vmap { 0 : goto endpoint-SGOXE6O3-ns2/svc2/tcp/p80__10.180.0.2/80 }
	add rule ip testing service-ULMVA6XW-ns1/svc1/tcp/p80 ip daddr 172.30.0.41 tcp dport 80 ip saddr != 10.0.0.0/8 jump mark-for-masquerade
	add rule ip testing service-ULMVA6XW-ns1/svc1/tcp/p80 numgen random mod 1 vmap { 0 : goto endpoint-5OJB2KTY-ns1/svc1/tcp/p80__10.180.0.1/80 }
	add set ip testing firewall { type ipv4_addr . inet_proto . inet_service ; comment "destinations that are subject to LoadBalancerSourceRanges" ; }
	add set ip testing firewall-allow { type ipv4_addr . inet_proto . inet_service . ipv4_addr ; flags interval ; comment "destinations+sources that are allowed by LoadBalancerSourceRanges" ; }
	add element ip testing no-endpoint-nodeports { tcp . 3001 comment "ns2/svc2:p80" : drop }
	add element ip testing no-endpoint-services { 1.2.3.4 . tcp . 80 comment "ns2/svc2:p80" : drop }
	add element ip testing no-endpoint-services { 192.168.99.22 . tcp . 80 comment "ns2/svc2:p80" : drop }
	add element ip testing service-ips { 1.2.3.4 . tcp . 80 : goto external-42NFTM6N-ns2/svc2/tcp/p80 }
	add element ip testing service-ips { 172.30.0.41 . tcp . 80 : goto service-ULMVA6XW-ns1/svc1/tcp/p80 }
	add element ip testing service-ips { 172.30.0.42 . tcp . 80 : goto service-42NFTM6N-ns2/svc2/tcp/p80 }
	add element ip testing service-ips { 192.168.99.22 . tcp . 80 : goto external-42NFTM6N-ns2/svc2/tcp/p80 }
	add element ip testing service-nodeports { tcp . 3001 : goto external-42NFTM6N-ns2/svc2/tcp/p80 }
	`)

func Test_sortNFTablesTransaction(t *testing.T) {
	output := sortNFTablesTransaction(testInput)
	expected := strings.TrimSpace(testExpected)

	diff := cmp.Diff(expected, output)
	if diff != "" {
		t.Errorf("output does not match expected:\n%s", diff)
	}
}

func Test_diffNFTablesTransaction(t *testing.T) {
	diff := diffNFTablesTransaction(testInput, testExpected)
	if diff != "" {
		t.Errorf("found diff in inputs that should have been equal:\n%s", diff)
	}

	notExpected := strings.Join(strings.Split(testExpected, "\n")[2:], "\n")
	diff = diffNFTablesTransaction(testInput, notExpected)
	if diff == "" {
		t.Errorf("found no diff in inputs that should have been different")
	}
}

func Test_diffNFTablesChain(t *testing.T) {
	fake := knftables.NewFake(knftables.IPv4Family, "testing")
	tx := fake.NewTransaction()

	tx.Add(&knftables.Table{})
	tx.Add(&knftables.Chain{
		Name: "mark-masq-chain",
	})
	tx.Add(&knftables.Chain{
		Name: "masquerade-chain",
	})
	tx.Add(&knftables.Chain{
		Name: "empty-chain",
	})

	tx.Add(&knftables.Rule{
		Chain: "mark-masq-chain",
		Rule:  "mark set mark or 0x4000",
	})

	tx.Add(&knftables.Rule{
		Chain: "masquerade-chain",
		Rule:  "mark and 0x4000 == 0 return",
	})
	tx.Add(&knftables.Rule{
		Chain: "masquerade-chain",
		Rule:  "mark set mark xor 0x4000",
	})
	tx.Add(&knftables.Rule{
		Chain: "masquerade-chain",
		Rule:  "masquerade fully-random",
	})

	err := fake.Run(context.Background(), tx)
	if err != nil {
		t.Fatalf("Unexpected error running transaction: %v", err)
	}

	diff := diffNFTablesChain(fake, "mark-masq-chain", "mark set mark or 0x4000")
	if diff != "" {
		t.Errorf("unexpected difference in mark-masq-chain:\n%s", diff)
	}
	diff = diffNFTablesChain(fake, "mark-masq-chain", "mark set mark or 0x4000\n")
	if diff != "" {
		t.Errorf("unexpected difference in mark-masq-chain with trailing newline:\n%s", diff)
	}

	diff = diffNFTablesChain(fake, "masquerade-chain", "mark and 0x4000 == 0 return\nmark set mark xor 0x4000\nmasquerade fully-random")
	if diff != "" {
		t.Errorf("unexpected difference in masquerade-chain:\n%s", diff)
	}
	diff = diffNFTablesChain(fake, "masquerade-chain", "mark set mark xor 0x4000\nmasquerade fully-random")
	if diff == "" {
		t.Errorf("unexpected lack of difference in wrong masquerade-chain")
	}

	diff = diffNFTablesChain(fake, "empty-chain", "")
	if diff != "" {
		t.Errorf("unexpected difference in empty-chain:\n%s", diff)
	}
	diff = diffNFTablesChain(fake, "empty-chain", "\n")
	if diff != "" {
		t.Errorf("unexpected difference in empty-chain with trailing newline:\n%s", diff)
	}
}
