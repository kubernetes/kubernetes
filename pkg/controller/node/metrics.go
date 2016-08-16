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
	"time"

	"k8s.io/kubernetes/pkg/api/unversioned"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
)

const (
	NodeControllerSubsystem = "node_collector"
	ZoneHealthStatisticKey  = "zone_health"
	ZoneSizeKey             = "zone_size"
	ZoneNoUnhealthyNodesKey = "unhealty_nodes_in_zone"
	EvictionsIn10MinutesKey = "10_minute_evictions"
	EvictionsIn1HourKey     = "1_hour_evictions"
)

var (
	ZoneHealth = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: NodeControllerSubsystem,
			Name:      ZoneHealthStatisticKey,
			Help:      "Gauge measuring percentage of healty nodes per zone.",
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
	Evictions10Minutes = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: NodeControllerSubsystem,
			Name:      EvictionsIn10MinutesKey,
			Help:      "Gauge measuring number of Node evictions that happened in previous 10 minutes per zone.",
		},
		[]string{"zone"},
	)
	Evictions1Hour = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: NodeControllerSubsystem,
			Name:      EvictionsIn1HourKey,
			Help:      "Gauge measuring number of Node evictions that happened in previous hour per zone.",
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
		prometheus.MustRegister(Evictions10Minutes)
		prometheus.MustRegister(Evictions1Hour)
	})
}

type eviction struct {
	node string
	time unversioned.Time
}

type evictionData struct {
	sync.Mutex
	nodeEvictionCount map[string]map[string]int
	nodeEvictionList  []eviction
	now               func() unversioned.Time
	windowSize        time.Duration
}

func newEvictionData(windowSize time.Duration) *evictionData {
	return &evictionData{
		nodeEvictionCount: make(map[string]map[string]int),
		nodeEvictionList:  make([]eviction, 0),
		now:               unversioned.Now,
		windowSize:        windowSize,
	}
}

func (e *evictionData) slideWindow() {
	e.Lock()
	defer e.Unlock()
	now := e.now()
	firstInside := 0
	for _, v := range e.nodeEvictionList {
		if v.time.Add(e.windowSize).Before(now.Time) {
			firstInside++
			zone := ""
			for z := range e.nodeEvictionCount {
				if _, ok := e.nodeEvictionCount[z][v.node]; ok {
					zone = z
					break
				}
			}
			if zone == "" {
				glog.Warningf("EvictionData corruption - unknown zone for node %v", v.node)
				continue
			}
			if e.nodeEvictionCount[zone][v.node] > 1 {
				e.nodeEvictionCount[zone][v.node] = e.nodeEvictionCount[zone][v.node] - 1
			} else {
				delete(e.nodeEvictionCount[zone], v.node)
			}
		} else {
			break
		}
	}
	e.nodeEvictionList = e.nodeEvictionList[firstInside:]
}

func (e *evictionData) registerEviction(node, zone string) {
	e.Lock()
	defer e.Unlock()

	e.nodeEvictionList = append(e.nodeEvictionList, eviction{node: node, time: e.now()})
	if _, ok := e.nodeEvictionCount[zone]; !ok {
		e.nodeEvictionCount[zone] = make(map[string]int)
	}
	if _, ok := e.nodeEvictionCount[zone][node]; !ok {
		e.nodeEvictionCount[zone][node] = 1
	} else {
		e.nodeEvictionCount[zone][node] = e.nodeEvictionCount[zone][node] + 1
	}
}

func (e *evictionData) removeEviction(node, zone string) {
	e.Lock()
	defer e.Unlock()

	// TODO: This may be inefficient, but hopefully will be rarely called. Verify that this is true.
	for i := len(e.nodeEvictionList) - 1; i >= 0; i-- {
		if e.nodeEvictionList[i].node == node {
			e.nodeEvictionList = append(e.nodeEvictionList[:i], e.nodeEvictionList[i+1:]...)
			break
		}
	}
	if e.nodeEvictionCount[zone][node] > 1 {
		e.nodeEvictionCount[zone][node] = e.nodeEvictionCount[zone][node] - 1
	} else {
		delete(e.nodeEvictionCount[zone], node)
	}
}

func (e *evictionData) countEvictions(zone string) int {
	e.Lock()
	defer e.Unlock()
	return len(e.nodeEvictionCount[zone])
}

func (e *evictionData) getZones() []string {
	e.Lock()
	defer e.Unlock()

	zones := make([]string, 0, len(e.nodeEvictionCount))
	for k := range e.nodeEvictionCount {
		zones = append(zones, k)
	}
	return zones
}

func (e *evictionData) initZone(zone string) {
	e.Lock()
	defer e.Unlock()

	e.nodeEvictionCount[zone] = make(map[string]int)
}
