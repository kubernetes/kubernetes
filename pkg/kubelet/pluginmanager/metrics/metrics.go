/*
Copyright 2019 The Kubernetes Authors.

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
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
)

const (
	pluginNameNotAvailable = "N/A"
	// Metric keys for Plugin Manager.
	pluginManagerTotalPlugins = "plugin_manager_total_plugins"
)

var (
	registerMetrics sync.Once

	totalPluginsDesc = prometheus.NewDesc(
		pluginManagerTotalPlugins,
		"Number of plugins in Plugin Manager",
		[]string{"socket_path", "state"},
		nil,
	)
)

// pluginCount is a map of maps used as a counter.
type pluginCount map[string]map[string]int64

func (pc pluginCount) add(state, pluginName string) {
	count, ok := pc[state]
	if !ok {
		count = map[string]int64{}
	}
	count[pluginName]++
	pc[state] = count
}

// Register registers Plugin Manager metrics.
func Register(asw cache.ActualStateOfWorld, dsw cache.DesiredStateOfWorld) {
	registerMetrics.Do(func() {
		prometheus.MustRegister(&totalPluginsCollector{asw, dsw})
	})
}

type totalPluginsCollector struct {
	asw cache.ActualStateOfWorld
	dsw cache.DesiredStateOfWorld
}

var _ prometheus.Collector = &totalPluginsCollector{}

// Describe implements the prometheus.Collector interface.
func (c *totalPluginsCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- totalPluginsDesc
}

// Collect implements the prometheus.Collector interface.
func (c *totalPluginsCollector) Collect(ch chan<- prometheus.Metric) {
	for stateName, pluginCount := range c.getPluginCount() {
		for socketPath, count := range pluginCount {
			metric, err := prometheus.NewConstMetric(totalPluginsDesc,
				prometheus.GaugeValue,
				float64(count),
				socketPath,
				stateName)
			if err != nil {
				klog.Warningf("Failed to create metric : %v", err)
			}
			ch <- metric
		}
	}
}

func (c *totalPluginsCollector) getPluginCount() pluginCount {
	counter := make(pluginCount)
	for _, registeredPlugin := range c.asw.GetRegisteredPlugins() {
		socketPath := registeredPlugin.SocketPath
		counter.add("actual_state_of_world", socketPath)
	}

	for _, pluginToRegister := range c.dsw.GetPluginsToRegister() {
		socketPath := pluginToRegister.SocketPath
		counter.add("desired_state_of_world", socketPath)
	}
	return counter
}
