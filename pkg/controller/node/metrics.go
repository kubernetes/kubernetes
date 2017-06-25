/*
Copyright 2016 The Kubernetes Authors.

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

package node

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
)

const (
	NodeControllerSubsystem = "node_collector"
	ZoneHealthStatisticKey  = "zone_health"
	ZoneSizeKey             = "zone_size"
	ZoneNoUnhealthyNodesKey = "unhealthy_nodes_in_zone"
	EvictionsNumberKey      = "evictions_number"
)

var (
	ZoneHealth = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: NodeControllerSubsystem,
			Name:      ZoneHealthStatisticKey,
			Help:      "Gauge measuring percentage of healthy nodes per zone.",
		},
		[]string{"zone"},
	)
	ZoneSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: NodeControllerSubsystem,
			Name:      ZoneSizeKey,
			Help:      "Gauge measuring number of registered Nodes per zones.",
		},
		[]string{"zone"},
	)
	UnhealthyNodes = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: NodeControllerSubsystem,
			Name:      ZoneNoUnhealthyNodesKey,
			Help:      "Gauge measuring number of not Ready Nodes per zones.",
		},
		[]string{"zone"},
	)
	EvictionsNumber = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: NodeControllerSubsystem,
			Name:      EvictionsNumberKey,
			Help:      "Number of Node evictions that happened since current instance of NodeController started.",
		},
		[]string{"zone"},
	)
)

var registerMetrics sync.Once

func Register() {
	registerMetrics.Do(func() {
		prometheus.MustRegister(ZoneHealth)
		prometheus.MustRegister(ZoneSize)
		prometheus.MustRegister(UnhealthyNodes)
		prometheus.MustRegister(EvictionsNumber)
	})
}
