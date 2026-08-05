/*
Copyright 2018 The Kubernetes Authors.

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

package leaderelection

import (
	"sync"

	"k8s.io/client-go/tools/leaderelection"
	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	leaderGauge = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Name:           "leader_election_master_status",
		StabilityLevel: k8smetrics.ALPHA,
		Help:           "Gauge of if the reporting system is master of the relevant lease, 0 indicates backup, 1 indicates master. 'name' is the string used to identify the lease. Please make sure to group by name.",
	}, []string{"name"})
	// A cumulative counter should be sufficient to get a rough ratio of slow path
	// exercised given the leader election frequency is specified explicitly. So that
	// to avoid the overhead to report a counter exercising fastpath.
	leaderSlowpathCounter = k8smetrics.NewCounterVec(&k8smetrics.CounterOpts{
		Name:           "leader_election_slowpath_total",
		StabilityLevel: k8smetrics.ALPHA,
		Help:           "Total number of slow path exercised in renewing leader leases. 'name' is the string used to identify the lease. Please make sure to group by name.",
	}, []string{"name"})
)

var registerOnce sync.Once

// init only installs the provider with leaderelection. The actual metric
// registration with legacyregistry is deferred until the first factory
// method is called (which happens at runtime when the first leader election
// manager is constructed).
func init() {
	leaderelection.SetProvider(prometheusMetricsProvider{})
}

type prometheusMetricsProvider struct{}

// register registers leaderelection metrics with the legacy registry. It is
// invoked from the provider factory method via registerOnce so that the
// registration (and thus Create()) happens at runtime.
func register() {
	registerOnce.Do(func() {
		legacyregistry.MustRegister(leaderGauge)
		legacyregistry.MustRegister(leaderSlowpathCounter)
	})
}

func (prometheusMetricsProvider) NewLeaderMetric() leaderelection.LeaderMetric {
	register()
	return &leaderAdapter{gauge: leaderGauge, counter: leaderSlowpathCounter}
}

type leaderAdapter struct {
	gauge   *k8smetrics.GaugeVec
	counter *k8smetrics.CounterVec
}

func (s *leaderAdapter) On(name string) {
	s.gauge.WithLabelValues(name).Set(1.0)
}

func (s *leaderAdapter) Off(name string) {
	s.gauge.WithLabelValues(name).Set(0.0)
}

func (s *leaderAdapter) SlowpathExercised(name string) {
	s.counter.WithLabelValues(name).Inc()
}
