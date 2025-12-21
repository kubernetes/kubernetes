//go:build linux
// +build linux

/*
Copyright 2017 The Kubernetes Authors.

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

package ipvs

import (
	"bytes"
	"strings"
	"testing"

	"github.com/lithammer/dedent"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/kubernetes/pkg/proxy"
	ipsettest "k8s.io/kubernetes/pkg/proxy/ipvs/ipset/testing"
	ipvstest "k8s.io/kubernetes/pkg/proxy/ipvs/util/testing"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	iptablestest "k8s.io/kubernetes/pkg/util/iptables/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestCleanupLeftovers(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ipt := iptablestest.NewFake()
	ipts := map[v1.IPFamily]utiliptables.Interface{
		v1.IPv4Protocol: ipt,
	}
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)

	// Preload some iptables rules
	initial := dedent.Dedent(`
		*filter
		:INPUT - [0:0]
		:FORWARD - [0:0]
		:OUTPUT - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-KUBELET-CANARY - [0:0]
		-A INPUT -j KUBE-FIREWALL
		-A INPUT -m comment --comment "someone else's input rule" -s 1.2.3.4 -j DROP
		-A FORWARD -m comment --comment "someone else's forward rule" -s 1.2.3.4 -j DROP
		-A OUTPUT -j KUBE-FIREWALL
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
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
	err := ipt.RestoreAll([]byte(initial), utiliptables.NoFlushTables, utiliptables.NoRestoreCounters)
	if err != nil {
		t.Fatalf("Unexpected error setting up iptables state: %v", err)
	}

	fp := NewFakeProxier(ctx, ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
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

	_ = fp.syncProxyRules()

	// test cleanup left over
	encounteredError := cleanupLeftovers(ctx, ipvs, ipts, ipset)
	if encounteredError {
		t.Errorf("Cleanup leftovers failed")
	}

	var buf bytes.Buffer
	err = ipt.SaveInto("", &buf)
	if err != nil {
		t.Fatalf("Unexpected error reading filter table: %v", err)
	}
	got := buf.String()

	expected := strings.TrimLeft(dedent.Dedent(`
		*nat
		:PREROUTING - [0:0]
		:INPUT - [0:0]
		:OUTPUT - [0:0]
		:POSTROUTING - [0:0]
		:KUBE-KUBELET-CANARY - [0:0]
		COMMIT
		*filter
		:INPUT - [0:0]
		:FORWARD - [0:0]
		:OUTPUT - [0:0]
		:KUBE-FIREWALL - [0:0]
		:KUBE-KUBELET-CANARY - [0:0]
		-A INPUT -j KUBE-FIREWALL
		-A INPUT -m comment --comment "someone else's input rule" -s 1.2.3.4 -j DROP
		-A FORWARD -m comment --comment "someone else's forward rule" -s 1.2.3.4 -j DROP
		-A OUTPUT -j KUBE-FIREWALL
		-A OUTPUT -m comment --comment "someone else's output rule" -s 1.2.3.4 -j DROP
		-A KUBE-FIREWALL -m comment --comment "block incoming localnet connections" -d 127.0.0.0/8 ! -s 127.0.0.0/8 -m conntrack ! --ctstate RELATED,ESTABLISHED,DNAT -j DROP
		COMMIT
		*mangle
		:KUBE-IPTABLES-HINT - [0:0]
		:KUBE-KUBELET-CANARY - [0:0]
		COMMIT
		`), "\n")

	if got != expected {
		t.Fatalf("expected post-cleanup rules to be:\n%s\n\ngot:\n%s\n", expected, got)
	}

	// FIXME: check ipvs and ipset state
}
