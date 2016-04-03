// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package prometheus_test

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
)

// ClusterManager is an example for a system that might have been built without
// Prometheus in mind. It models a central manager of jobs running in a
// cluster. To turn it into something that collects Prometheus metrics, we
// simply add the two methods required for the Collector interface.
//
// An additional challenge is that multiple instances of the ClusterManager are
// run within the same binary, each in charge of a different zone. We need to
// make use of ConstLabels to be able to register each ClusterManager instance
// with Prometheus.
type ClusterManager struct {
	Zone     string
	OOMCount *prometheus.CounterVec
	RAMUsage *prometheus.GaugeVec
	mtx      sync.Mutex // Protects OOMCount and RAMUsage.
	// ... many more fields
}

// ReallyExpensiveAssessmentOfTheSystemState is a mock for the data gathering a
// real cluster manager would have to do. Since it may actually be really
// expensive, it must only be called once per collection. This implementation,
// obviously, only returns some made-up data.
func (c *ClusterManager) ReallyExpensiveAssessmentOfTheSystemState() (
	oomCountByHost map[string]int, ramUsageByHost map[string]float64,
) {
	// Just example fake data.
	oomCountByHost = map[string]int{
		"foo.example.org": 42,
		"bar.example.org": 2001,
	}
	ramUsageByHost = map[string]float64{
		"foo.example.org": 6.023e23,
		"bar.example.org": 3.14,
	}
	return
}

// Describe faces the interesting challenge that the two metric vectors that are
// used in this example are already Collectors themselves. However, thanks to
// the use of channels, it is really easy to "chain" Collectors. Here we simply
// call the Describe methods of the two metric vectors.
func (c *ClusterManager) Describe(ch chan<- *prometheus.Desc) {
	c.OOMCount.Describe(ch)
	c.RAMUsage.Describe(ch)
}

// Collect first triggers the ReallyExpensiveAssessmentOfTheSystemState. Then it
// sets the retrieved values in the two metric vectors and then sends all their
// metrics to the channel (again using a chaining technique as in the Describe
// method). Since Collect could be called multiple times concurrently, that part
// is protected by a mutex.
func (c *ClusterManager) Collect(ch chan<- prometheus.Metric) {
	oomCountByHost, ramUsageByHost := c.ReallyExpensiveAssessmentOfTheSystemState()
	c.mtx.Lock()
	defer c.mtx.Unlock()
	for host, oomCount := range oomCountByHost {
		c.OOMCount.WithLabelValues(host).Set(float64(oomCount))
	}
	for host, ramUsage := range ramUsageByHost {
		c.RAMUsage.WithLabelValues(host).Set(ramUsage)
	}
	c.OOMCount.Collect(ch)
	c.RAMUsage.Collect(ch)
	// All metrics in OOMCount and RAMUsage are sent to the channel now. We
	// can safely reset the two metric vectors now, so that we can start
	// fresh in the next Collect cycle. (Imagine a host disappears from the
	// cluster. If we did not reset here, its Metric would stay in the
	// metric vectors forever.)
	c.OOMCount.Reset()
	c.RAMUsage.Reset()
}

// NewClusterManager creates the two metric vectors OOMCount and RAMUsage. Note
// that the zone is set as a ConstLabel. (It's different in each instance of the
// ClusterManager, but constant over the lifetime of an instance.) The reported
// values are partitioned by host, which is therefore a variable label.
func NewClusterManager(zone string) *ClusterManager {
	return &ClusterManager{
		Zone: zone,
		OOMCount: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Subsystem:   "clustermanager",
				Name:        "oom_count",
				Help:        "number of OOM crashes",
				ConstLabels: prometheus.Labels{"zone": zone},
			},
			[]string{"host"},
		),
		RAMUsage: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Subsystem:   "clustermanager",
				Name:        "ram_usage_bytes",
				Help:        "RAM usage as reported to the cluster manager",
				ConstLabels: prometheus.Labels{"zone": zone},
			},
			[]string{"host"},
		),
	}
}

func ExampleCollector_clustermanager() {
	workerDB := NewClusterManager("db")
	workerCA := NewClusterManager("ca")
	prometheus.MustRegister(workerDB)
	prometheus.MustRegister(workerCA)

	// Since we are dealing with custom Collector implementations, it might
	// be a good idea to enable the collect checks in the registry.
	prometheus.EnableCollectChecks(true)
}
