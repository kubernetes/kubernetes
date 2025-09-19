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
	"testing"

	"github.com/lithammer/dedent"

	klogtesting "k8s.io/klog/v2/ktesting"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	iptablestest "k8s.io/kubernetes/pkg/util/iptables/testing"
)

func TestCleanupLeftovers(t *testing.T) {
	_, ctx := klogtesting.NewTestContext(t)
	ipt := iptablestest.NewFake()

	initial := dedent.Dedent(`
		*filter
		:INPUT - [0:0]
		:FORWARD - [0:0]
		:OUTPUT - [0:0]
		:KUBE-EXTERNAL-SERVICES - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-FORWARD - [0:0]
		:KUBE-KUBELET-CANARY - [0:0]
		:KUBE-PROXY-CANARY - [0:0]
		:KUBE-NODEPORTS - [0:0]
		:KUBE-SERVICES - [0:0]
		:KUBE-PROXY-FIREWALL - [0:0]
		-A INPUT -j KUBE-FIREWALL
		-A INPUT -m comment --comment kubernetes health check service ports -j KUBE-NODEPORTS
		-A INPUT -m comment --comment "someone else's input rule" -s 1.2.3.4 -j DROP
		-A INPUT -m conntrack --ctstate NEW -m comment --comment kubernetes externally-visible service portals -j KUBE-EXTERNAL-SERVICES
		-A FORWARD -m comment --comment "someone else's forward rule" -s 1.2.3.4 -j DROP
		-A FORWARD -m comment --comment kubernetes forwarding rules -j KUBE-FORWARD
		-A FORWARD -m conntrack --ctstate NEW -m comment --comment kubernetes service portals -j KUBE-SERVICES
		-A FORWARD -m conntrack --ctstate NEW -m comment --comment kubernetes externally-visible service portals -j KUBE-EXTERNAL-SERVICES
		-A OUTPUT -j KUBE-FIREWALL
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
		-A OUTPUT -m comment --comment "someone else's output rule" -s 1.2.3.4 -j DROP
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
		:KUBE-KUBELET-CANARY - [0:0]
		:KUBE-MARK-MASQ - [0:0]
		:KUBE-NODEPORTS - [0:0]
		:KUBE-POSTROUTING - [0:0]
		:KUBE-PROXY-CANARY - [0:0]
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
		*mangle
		:KUBE-IPTABLES-HINT - [0:0]
		:KUBE-KUBELET-CANARY - [0:0]
		:KUBE-PROXY-CANARY - [0:0]
		COMMIT
		`)

	err := ipt.RestoreAll([]byte(initial), utiliptables.NoFlushTables, utiliptables.NoRestoreCounters)
	if err != nil {
		t.Fatalf("Unexpected error setting up iptables state: %v", err)
	}

	encounteredError := cleanupLeftoversForFamily(ctx, ipt)
	if encounteredError {
		t.Fatal("Unexpected error from cleanupLeftoversForFamily")
	}

	var buf bytes.Buffer
	err = ipt.SaveInto("", &buf)
	if err != nil {
		t.Fatalf("Unexpected error reading filter table: %v", err)
	}

	expected := dedent.Dedent(`
		*filter
		:INPUT - [0:0]
		:FORWARD - [0:0]
		:OUTPUT - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-KUBELET-CANARY - [0:0]

		# These are created by both kubelet and kube-proxy and intentionally not
		# cleaned up by kube-proxy.
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		-A INPUT -j KUBE-FIREWALL
		-A OUTPUT -j KUBE-FIREWALL

		# Non-Kubernetes rules are preserved
		-A INPUT -m comment --comment "someone else's input rule" -s 1.2.3.4 -j DROP
		-A FORWARD -m comment --comment "someone else's forward rule" -s 1.2.3.4 -j DROP
		-A OUTPUT -m comment --comment "someone else's output rule" -s 1.2.3.4 -j DROP

		COMMIT

		*nat
		:PREROUTING - [0:0]
		:INPUT - [0:0]
		:OUTPUT - [0:0]
		:POSTROUTING - [0:0]
		:KUBE-KUBELET-CANARY - [0:0]
		COMMIT

		*mangle
		:KUBE-IPTABLES-HINT - [0:0]
		:KUBE-KUBELET-CANARY - [0:0]
		COMMIT
		`)
	expected, err = sortIPTablesRules(expected)
	if err != nil {
		t.Fatalf("Unexpected error sorting expected output: %v", err)
	}

	assertIPTablesRulesEqual(t, getLine(), false, expected, buf.String())
}
