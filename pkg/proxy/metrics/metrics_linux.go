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

package metrics

import (
	"context"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy/util/nfacct"
	"sigs.k8s.io/knftables"
)

var _ metrics.StableCollector = &nfacctMetricCollector{}

// FIXME: The metrics code should not know details about how the iptables and nftables
// backends work. The logic of `nfacctMetricCollector` and `nftCounterMetricCollector`
// needs to move into their respective backends.
func newNFAcctMetricCollector(counter string, description *metrics.Desc) *nfacctMetricCollector {
	client, err := nfacct.New()
	if err != nil {
		klog.ErrorS(err, "failed to initialize nfacct client")
		return nil
	}
	return &nfacctMetricCollector{
		client:      client,
		counter:     counter,
		description: description,
	}
}

type nfacctMetricCollector struct {
	metrics.BaseStableCollector
	client      nfacct.Interface
	counter     string
	description *metrics.Desc
}

// DescribeWithStability implements the metrics.StableCollector interface.
func (n *nfacctMetricCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- n.description
}

// CollectWithStability implements the metrics.StableCollector interface.
func (n *nfacctMetricCollector) CollectWithStability(ch chan<- metrics.Metric) {
	if n.client != nil {
		counter, err := n.client.Get(n.counter)
		if err != nil {
			klog.ErrorS(err, "failed to collect nfacct counter", "counter", n.counter)
		} else {
			metric, err := metrics.NewConstMetric(n.description, metrics.CounterValue, float64(counter.Packets))
			if err != nil {
				klog.ErrorS(err, "failed to create constant metric")
			} else {
				ch <- metric
			}
		}
	}
}

// nftCounterListTimeout bounds a single nft counter listing during a metrics scrape
const nftCounterListTimeout = 2 * time.Second

var _ metrics.StableCollector = &nftCounterMetricCollector{}

// nftCounterMetricCollector reports named nftables counters from the kube-proxy
// table of each IP family as const counter metrics labeled by ip_family and
// protocol. counters maps each nftables counter name to its protocol label.
type nftCounterMetricCollector struct {
	metrics.BaseStableCollector
	clients     map[v1.IPFamily]knftables.Interface
	counters    map[string]string
	description *metrics.Desc
}

func newNFTablesCounterMetricCollector(counters map[string]string, description *metrics.Desc) *nftCounterMetricCollector {
	families := map[v1.IPFamily]knftables.Family{
		v1.IPv4Protocol: knftables.IPv4Family,
		v1.IPv6Protocol: knftables.IPv6Family,
	}
	clients := make(map[v1.IPFamily]knftables.Interface, len(families))
	for family, nftFamily := range families {
		nft, err := knftables.New(nftFamily, "kube-proxy")
		if err != nil {
			klog.ErrorS(err, "failed to initialize nftables client for metrics", "ipFamily", family)
			continue
		}
		clients[family] = nft
	}
	if len(clients) == 0 {
		return nil
	}
	return &nftCounterMetricCollector{
		clients:     clients,
		counters:    counters,
		description: description,
	}
}

// DescribeWithStability implements the metrics.StableCollector interface.
func (n *nftCounterMetricCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- n.description
}

// CollectWithStability implements the metrics.StableCollector interface.
func (n *nftCounterMetricCollector) CollectWithStability(ch chan<- metrics.Metric) {
	for family, client := range n.clients {
		ctx, cancel := context.WithTimeout(context.Background(), nftCounterListTimeout)
		counters, err := client.ListCounters(ctx)
		cancel()
		if err != nil {
			// A not-found error means the table doesn't exist
			if !knftables.IsNotFound(err) {
				klog.ErrorS(err, "failed to list nftables counters", "ipFamily", family)
			}
			continue
		}
		for _, c := range counters {
			if c == nil || c.Packets == nil {
				continue
			}
			protocol, ok := n.counters[c.Name]
			if !ok {
				continue
			}
			metric, err := metrics.NewConstMetric(n.description, metrics.CounterValue, float64(*c.Packets), string(family), protocol)
			if err != nil {
				klog.ErrorS(err, "failed to create constant metric")
				continue
			}
			ch <- metric
		}
	}
}
