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

package metrics

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-base/metrics"
	"sigs.k8s.io/knftables"
)

func TestNFTCounterMetricCollector(t *testing.T) {
	fake := knftables.NewFake(knftables.IPv4Family, "kube-proxy")
	tx := fake.NewTransaction()
	tx.Add(&knftables.Table{})
	tx.Add(&knftables.Counter{Name: LocalhostNodePortRejectedCounter})
	if err := fake.Run(context.Background(), tx); err != nil {
		t.Fatalf("failed to set up fake nftables: %v", err)
	}
	fake.Table.Counters[LocalhostNodePortRejectedCounter].Packets = new(uint64(7))

	c := &nftCounterMetricCollector{
		clients:     map[v1.IPFamily]knftables.Interface{v1.IPv4Protocol: fake},
		counter:     LocalhostNodePortRejectedCounter,
		description: localhostNodePortRejectedPacketsDescription,
	}

	// Registration initializes the lazy Desc, which NewConstMetric requires.
	reg := metrics.NewKubeRegistry()
	reg.CustomMustRegister(c)

	mfs, err := reg.Gather()
	if err != nil {
		t.Fatalf("failed to gather metrics: %v", err)
	}

	const want = "kubeproxy_nftables_localhost_nodeport_rejected_packets_total"
	var values []float64
	for _, mf := range mfs {
		if mf.GetName() != want {
			continue
		}
		for _, m := range mf.GetMetric() {
			values = append(values, m.GetCounter().GetValue())
		}
	}

	if len(values) != 1 {
		t.Fatalf("expected 1 sample of %s, got %d: %v", want, len(values), values)
	}
	if values[0] != 7 {
		t.Errorf("expected counter value 7, got %v", values[0])
	}
}
