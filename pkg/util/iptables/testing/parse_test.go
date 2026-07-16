//go:build linux

/*
Copyright 2022 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/lithammer/dedent"

	"k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/utils/ptr"
)

func TestParseRule(t *testing.T) {
	testCases := []struct {
		name      string
		rule      string
		parsed    *Rule
		nonStrict bool
		err       string
	}{
		{
			name: "basic rule",
			rule: `-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT`,
			parsed: &Rule{
				Raw:             `-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT`,
				Chain:           iptables.Chain("KUBE-NODEPORTS"),
				Comment:         &IPTablesValue{Value: "ns2/svc2:p80 health check node port"},
				Protocol:        &IPTablesValue{Value: "tcp"},
				DestinationPort: &IPTablesValue{Value: "30000"},
				Jump:            &IPTablesValue{Value: "ACCEPT"},
			},
		},
		{
			name: "addRuleToChainRegex requires an actual rule, not just a chain name",
			rule: `-A KUBE-NODEPORTS`,
			err:  `(no match rules)`,
		},
		{
			name: "ParseRule only parses adds",
			rule: `-D KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT`,
			err:  `(does not start with "-A CHAIN")`,
		},
		{
			name: "unquoted comment",
			rule: `-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment ns1/svc1:p80 -j KUBE-SEP-SXIVWICOYRO3J4NJ`,
			parsed: &Rule{
				Raw:     `-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment ns1/svc1:p80 -j KUBE-SEP-SXIVWICOYRO3J4NJ`,
				Chain:   iptables.Chain("KUBE-SVC-XPGD46QRK7WJZT7O"),
				Comment: &IPTablesValue{Value: "ns1/svc1:p80"},
				Jump:    &IPTablesValue{Value: "KUBE-SEP-SXIVWICOYRO3J4NJ"},
			},
		},
		{
			name: "local source",
			rule: `-A KUBE-XLB-GNZBNJ2PO5MGZ6GT -m comment --comment "masquerade LOCAL traffic for ns2/svc2:p80 LB IP" -m addrtype --src-type LOCAL -j KUBE-MARK-MASQ`,
			parsed: &Rule{
				Raw:        `-A KUBE-XLB-GNZBNJ2PO5MGZ6GT -m comment --comment "masquerade LOCAL traffic for ns2/svc2:p80 LB IP" -m addrtype --src-type LOCAL -j KUBE-MARK-MASQ`,
				Chain:      iptables.Chain("KUBE-XLB-GNZBNJ2PO5MGZ6GT"),
				Comment:    &IPTablesValue{Value: "masquerade LOCAL traffic for ns2/svc2:p80 LB IP"},
				SourceType: &IPTablesValue{Value: "LOCAL"},
				Jump:       &IPTablesValue{Value: "KUBE-MARK-MASQ"},
			},
		},
		{
			name: "not local destination",
			rule: `-A RULE-TYPE-NOT-CURRENTLY-USED-BY-KUBE-PROXY -m addrtype ! --dst-type LOCAL -j KUBE-MARK-MASQ`,
			parsed: &Rule{
				Raw:             `-A RULE-TYPE-NOT-CURRENTLY-USED-BY-KUBE-PROXY -m addrtype ! --dst-type LOCAL -j KUBE-MARK-MASQ`,
				Chain:           iptables.Chain("RULE-TYPE-NOT-CURRENTLY-USED-BY-KUBE-PROXY"),
				DestinationType: &IPTablesValue{Negated: true, Value: "LOCAL"},
				Jump:            &IPTablesValue{Value: "KUBE-MARK-MASQ"},
			},
		},
		{
			name: "destination IP/port",
			rule: `-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O`,
			parsed: &Rule{
				Raw:                `-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O`,
				Chain:              iptables.Chain("KUBE-SERVICES"),
				Comment:            &IPTablesValue{Value: "ns1/svc1:p80 cluster IP"},
				Protocol:           &IPTablesValue{Value: "tcp"},
				DestinationAddress: &IPTablesValue{Value: "172.30.0.41"},
				DestinationPort:    &IPTablesValue{Value: "80"},
				Jump:               &IPTablesValue{Value: "KUBE-SVC-XPGD46QRK7WJZT7O"},
			},
		},
		{
			name: "source IP",
			rule: `-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -s 10.180.0.1 -j KUBE-MARK-MASQ`,
			parsed: &Rule{
				Raw:           `-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -s 10.180.0.1 -j KUBE-MARK-MASQ`,
				Chain:         iptables.Chain("KUBE-SEP-SXIVWICOYRO3J4NJ"),
				Comment:       &IPTablesValue{Value: "ns1/svc1:p80"},
				SourceAddress: &IPTablesValue{Value: "10.180.0.1"},
				Jump:          &IPTablesValue{Value: "KUBE-MARK-MASQ"},
			},
		},
		{
			name: "not source IP",
			rule: `-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ`,
			parsed: &Rule{
				Raw:                `-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 172.30.0.41 --dport 80 ! -s 10.0.0.0/8 -j KUBE-MARK-MASQ`,
				Chain:              iptables.Chain("KUBE-SVC-XPGD46QRK7WJZT7O"),
				Comment:            &IPTablesValue{Value: "ns1/svc1:p80 cluster IP"},
				Protocol:           &IPTablesValue{Value: "tcp"},
				DestinationAddress: &IPTablesValue{Value: "172.30.0.41"},
				DestinationPort:    &IPTablesValue{Value: "80"},
				SourceAddress:      &IPTablesValue{Negated: true, Value: "10.0.0.0/8"},
				Jump:               &IPTablesValue{Value: "KUBE-MARK-MASQ"},
			},
		},
		{
			name: "affinity",
			rule: `-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment ns1/svc1:p80 -m recent --name KUBE-SEP-SXIVWICOYRO3J4NJ --rcheck --seconds 10800 --reap -j KUBE-SEP-SXIVWICOYRO3J4NJ`,
			parsed: &Rule{
				Raw:             `-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment ns1/svc1:p80 -m recent --name KUBE-SEP-SXIVWICOYRO3J4NJ --rcheck --seconds 10800 --reap -j KUBE-SEP-SXIVWICOYRO3J4NJ`,
				Chain:           iptables.Chain("KUBE-SVC-XPGD46QRK7WJZT7O"),
				Comment:         &IPTablesValue{Value: "ns1/svc1:p80"},
				AffinityName:    &IPTablesValue{Value: "KUBE-SEP-SXIVWICOYRO3J4NJ"},
				AffinitySeconds: &IPTablesValue{Value: "10800"},
				AffinityCheck:   ptr.To(true),
				AffinityReap:    ptr.To(true),
				Jump:            &IPTablesValue{Value: "KUBE-SEP-SXIVWICOYRO3J4NJ"},
			},
		},
		{
			name: "jump to DNAT",
			rule: `-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.1:80`,
			parsed: &Rule{
				Raw:             `-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.1:80`,
				Chain:           iptables.Chain("KUBE-SEP-SXIVWICOYRO3J4NJ"),
				Comment:         &IPTablesValue{Value: "ns1/svc1:p80"},
				Protocol:        &IPTablesValue{Value: "tcp"},
				Jump:            &IPTablesValue{Value: "DNAT"},
				DNATDestination: &IPTablesValue{Value: "10.180.0.1:80"},
			},
		},
		{
			name: "jump to endpoint",
			rule: `-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment ns4/svc4:p80 -m statistic --mode random --probability 0.5000000000 -j KUBE-SEP-UKSFD7AGPMPPLUHC`,
			parsed: &Rule{
				Raw:           `-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment ns4/svc4:p80 -m statistic --mode random --probability 0.5000000000 -j KUBE-SEP-UKSFD7AGPMPPLUHC`,
				Chain:         iptables.Chain("KUBE-SVC-4SW47YFZTEDKD3PK"),
				Comment:       &IPTablesValue{Value: "ns4/svc4:p80"},
				Probability:   &IPTablesValue{Value: "0.5000000000"},
				StatisticMode: &IPTablesValue{Value: "random"},
				Jump:          &IPTablesValue{Value: "KUBE-SEP-UKSFD7AGPMPPLUHC"},
			},
		},
		{
			name: "unrecognized arguments",
			rule: `-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment ns4/svc4:p80 -i eth0 -j KUBE-SEP-UKSFD7AGPMPPLUHC`,
			err:  `unrecognized parameter "-i"`,
		},
		{
			name:      "unrecognized arguments with strict=false",
			rule:      `-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment ns4/svc4:p80 -i eth0 -j KUBE-SEP-UKSFD7AGPMPPLUHC`,
			nonStrict: true,
			parsed: &Rule{
				Raw:     `-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment ns4/svc4:p80 -i eth0 -j KUBE-SEP-UKSFD7AGPMPPLUHC`,
				Chain:   iptables.Chain("KUBE-SVC-4SW47YFZTEDKD3PK"),
				Comment: &IPTablesValue{Value: "ns4/svc4:p80"},
				Jump:    &IPTablesValue{Value: "KUBE-SEP-UKSFD7AGPMPPLUHC"},
			},
		},
		{
			name: "bad use of !",
			rule: `-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment ns4/svc4:p80 ! -j KUBE-SEP-UKSFD7AGPMPPLUHC`,
			err:  `cannot negate parameter "-j"`,
		},
		{
			name: "missing argument",
			rule: `-A KUBE-SVC-4SW47YFZTEDKD3PK -m comment --comment ns4/svc4:p80 -j`,
			err:  `parameter "-j" requires an argument`,
		},
		{
			name: "negated bool arg",
			rule: `-A TEST -m recent ! --rcheck -j KUBE-SEP-SXIVWICOYRO3J4NJ`,
			parsed: &Rule{
				Raw:           `-A TEST -m recent ! --rcheck -j KUBE-SEP-SXIVWICOYRO3J4NJ`,
				Chain:         iptables.Chain("TEST"),
				AffinityCheck: ptr.To(false),
				Jump:          &IPTablesValue{Value: "KUBE-SEP-SXIVWICOYRO3J4NJ"},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			rule, err := ParseRule(testCase.rule, !testCase.nonStrict)
			if err != nil {
				if testCase.err == "" {
					t.Errorf("expected %+v, got error %q", testCase.parsed, err)
				} else if !strings.Contains(err.Error(), testCase.err) {
					t.Errorf("wrong error, expected %q got %q", testCase.err, err)
				}
			} else {
				if testCase.err != "" {
					t.Errorf("expected error %q, got %+v", testCase.err, rule)
				} else if !reflect.DeepEqual(rule, testCase.parsed) {
					t.Errorf("bad match: expected\n%+v\ngot\n%+v", testCase.parsed, rule)
				}
			}
		})
	}
}

// Helper for TestParseIPTablesDump. Obviously it should not be used in TestParseRule...
func mustParseRule(rule string) *Rule {
	parsed, err := ParseRule(rule, false)
	if err != nil {
		panic(fmt.Sprintf("failed to parse test case rule %q: %v", rule, err))
	}
	return parsed
}

func TestParseIPTablesDump(t *testing.T) {
	for _, tc := range []struct {
		name   string
		input  string
		output *IPTablesDump
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
			output: &IPTablesDump{
				Tables: []Table{{
					Name: iptables.TableFilter,
					Chains: []Chain{{
						Name: iptables.Chain("KUBE-SERVICES"),
					}, {
						Name: iptables.Chain("KUBE-EXTERNAL-SERVICES"),
					}, {
						Name: iptables.Chain("KUBE-FORWARD"),
						Rules: []*Rule{
							mustParseRule(`-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP`),
							mustParseRule(`-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT`),
							mustParseRule(`-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT`),
						},
					}, {
						Name: iptables.Chain("KUBE-NODEPORTS"),
						Rules: []*Rule{
							mustParseRule(`-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT`),
						},
					}},
				}, {
					Name: iptables.TableNAT,
					Chains: []Chain{{
						Name: iptables.Chain("KUBE-SERVICES"),
						Rules: []*Rule{
							mustParseRule(`-A KUBE-SERVICES -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 10.20.30.41 --dport 80 -j KUBE-SVC-XPGD46QRK7WJZT7O`),
							mustParseRule(`-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS`),
						},
					}, {
						Name: iptables.Chain("KUBE-NODEPORTS"),
					}, {
						Name: iptables.Chain("KUBE-POSTROUTING"),
						Rules: []*Rule{
							mustParseRule(`-A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN`),
							mustParseRule(`-A KUBE-POSTROUTING -j MARK --xor-mark 0x4000`),
							mustParseRule(`-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -j MASQUERADE`),
						},
					}, {
						Name: iptables.Chain("KUBE-MARK-MASQ"),
						Rules: []*Rule{
							mustParseRule(`-A KUBE-MARK-MASQ -j MARK --or-mark 0x4000`),
						},
					}, {
						Name: iptables.Chain("KUBE-SVC-XPGD46QRK7WJZT7O"),
						Rules: []*Rule{
							mustParseRule(`-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment "ns1/svc1:p80 cluster IP" -m tcp -p tcp -d 10.20.30.41 --dport 80 ! -s 10.0.0.0/24 -j KUBE-MARK-MASQ`),
							mustParseRule(`-A KUBE-SVC-XPGD46QRK7WJZT7O -m comment --comment ns1/svc1:p80 -j KUBE-SEP-SXIVWICOYRO3J4NJ`),
						},
					}, {
						Name: iptables.Chain("KUBE-SEP-SXIVWICOYRO3J4NJ"),
						Rules: []*Rule{
							mustParseRule(`-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -s 10.180.0.1 -j KUBE-MARK-MASQ`),
							mustParseRule(`-A KUBE-SEP-SXIVWICOYRO3J4NJ -m comment --comment ns1/svc1:p80 -m tcp -p tcp -j DNAT --to-destination 10.180.0.1:80`),
						},
					}},
				}},
			},
		},
		{
			name: "deletion",
			input: dedent.Dedent(`
				*nat
				:KUBE-SERVICES - [0:0]
				:KUBE-SVC-XPGD46QRK7WJZT7O - [0:0]
				:KUBE-SEP-SXIVWICOYRO3J4NJ - [0:0]
				-X KUBE-SVC-XPGD46QRK7WJZT7O
				-X KUBE-SEP-SXIVWICOYRO3J4NJ
				-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
				COMMIT
				`),
			output: &IPTablesDump{
				Tables: []Table{{
					Name: iptables.TableNAT,
					Chains: []Chain{{
						Name: iptables.Chain("KUBE-SERVICES"),
						Rules: []*Rule{
							mustParseRule(`-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS`),
						},
					}, {
						Name:    iptables.Chain("KUBE-SVC-XPGD46QRK7WJZT7O"),
						Deleted: true,
					}, {
						Name:    iptables.Chain("KUBE-SEP-SXIVWICOYRO3J4NJ"),
						Deleted: true,
					}},
				}},
			},
		},
		{
			name: "whitespace and comments",
			input: dedent.Dedent(`
				# Generated by iptables-save v1.8.7 on Mon May  9 11:22:21 2022
				# (not really...)
				*filter
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]

				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				  -A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				# This rule does a thing
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				# Completed on Mon May  9 11:22:21 2022
				`),
			output: &IPTablesDump{
				Tables: []Table{{
					Name: iptables.TableFilter,
					Chains: []Chain{{
						Name: iptables.Chain("KUBE-SERVICES"),
					}, {
						Name: iptables.Chain("KUBE-EXTERNAL-SERVICES"),
					}, {
						Name: iptables.Chain("KUBE-FORWARD"),
						Rules: []*Rule{
							mustParseRule(`-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP`),
							mustParseRule(`-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT`),
							mustParseRule(`-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT`),
						},
					}, {
						Name: iptables.Chain("KUBE-NODEPORTS"),
						Rules: []*Rule{
							mustParseRule(`-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT`),
						},
					}},
				}},
			},
		},
		{
			name: "no COMMIT line",
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
				`),
			error: "no COMMIT line?",
		},
		{
			name: "two tables, no second COMMIT line",
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
				`),
			error: "no COMMIT line?",
		},
		{
			name: "two tables, no second header line",
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
			error: "not a table name",
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
			error: `table 3 starts with "junk"`,
		},
		{
			name: "add to missing chain",
			input: dedent.Dedent(`
				*filter
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				`),
			error: `no such chain "KUBE-FORWARD"`,
		},
		{
			name: "add to deleted chain",
			input: dedent.Dedent(`
				*filter
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-X KUBE-FORWARD
				-A KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				`),
			error: `cannot add rules to deleted chain`,
		},
		{
			name: "deleted non-empty chain",
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
				-X KUBE-FORWARD
				COMMIT
				`),
			error: `cannot delete chain "KUBE-FORWARD" after adding rules`,
		},
		{
			name: "junk rule",
			input: dedent.Dedent(`
				*filter
				:KUBE-SERVICES - [0:0]
				:KUBE-EXTERNAL-SERVICES - [0:0]
				:KUBE-FORWARD - [0:0]
				:KUBE-NODEPORTS - [0:0]
				-A KUBE-NODEPORTS -m comment --comment "ns2/svc2:p80 health check node port" -m tcp -p tcp --dport 30000 -j ACCEPT
				-Q KUBE-FORWARD -m conntrack --ctstate INVALID -j DROP
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT
				-A KUBE-FORWARD -m comment --comment "kubernetes forwarding conntrack rule" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
				COMMIT
				`),
			error: `"-Q KUBE-FORWARD`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dump, err := ParseIPTablesDump(tc.input)
			if err == nil {
				if tc.error != "" {
					t.Errorf("unexpectedly did not get error")
				} else if !reflect.DeepEqual(tc.output, dump) {
					t.Errorf("bad output: expected %#v got %#v", tc.output, dump)
				}
			} else {
				if tc.error == "" {
					t.Errorf("got unexpected error: %v", err)
				} else if !strings.Contains(err.Error(), tc.error) {
					t.Errorf("got wrong error: %v (expected %q)", err, tc.error)
				}
			}
		})
	}
}
