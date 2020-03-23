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

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
)

const (
	// Metric keys for Plugin Manager.
	pluginManagerTotalPlugins = "plugin_manager_total_plugins"
)

var (
	registerMetrics sync.Once

	totalPluginsDesc = metrics.NewDesc(
		pluginManagerTotalPlugins,
		"Number of plugins in Plugin Manager",
		[]string{"socket_path", "state"},
		nil,
		metrics.ALPHA,
		"",
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
		legacyregistry.CustomMustRegister(&totalPluginsCollector{asw: asw, dsw: dsw})
	})
}

type totalPluginsCollector struct {
	metrics.BaseStableCollector

	asw cache.ActualStateOfWorld
	dsw cache.DesiredStateOfWorld
}

var _ metrics.StableCollector = &totalPluginsCollector{}

// DescribeWithStability implements the metrics.StableCollector interface.
func (c *totalPluginsCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- totalPluginsDesc
}

// CollectWithStability implements the metrics.StableCollector interface.
func (c *totalPluginsCollector) CollectWithStability(ch chan<- metrics.Metric) {
	for stateName, pluginCount := range c.getPluginCount() {
		for socketPath, count := range pluginCount {
			ch <- metrics.NewLazyConstMetric(totalPluginsDesc,
				metrics.GaugeValue,
				float64(count),
				socketPath,
				stateName)
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
