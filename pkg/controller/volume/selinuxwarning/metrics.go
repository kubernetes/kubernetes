/*
Copyright 2024 The Kubernetes Authors.

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

package selinuxwarning

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/selinuxwarning/cache"
)

var (
	seLinuxConflictDesc = metrics.NewDesc(
		metrics.BuildFQName("", "selinux_warning_controller", "selinux_volume_conflict"),
		"Conflict between two Pods using the same volume",
		[]string{"property", "pod1_namespace", "pod1_name", "pod1_value", "pod2_namespace", "pod2_name", "pod2_value"}, nil,
		metrics.ALPHA, "")
)
var registerMetrics sync.Once

func RegisterMetrics(logger klog.Logger, cache cache.VolumeCache) {
	registerMetrics.Do(func() {
		legacyregistry.CustomMustRegister(newCollector(logger, cache))
	})
}
func newCollector(logger klog.Logger, cache cache.VolumeCache) *collector {
	return &collector{
		logger: logger,
		cache:  cache,
	}
}

var _ metrics.StableCollector = &collector{}

type collector struct {
	metrics.BaseStableCollector
	cache  cache.VolumeCache
	logger klog.Logger
}

func (c *collector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- seLinuxConflictDesc
}

func (c *collector) CollectWithStability(ch chan<- metrics.Metric) {
	conflictCh := make(chan cache.Conflict)
	go func() {
		c.cache.SendConflicts(c.logger, conflictCh)
		close(conflictCh)
	}()

	for conflict := range conflictCh {
		ch <- metrics.NewLazyConstMetric(seLinuxConflictDesc,
			metrics.GaugeValue,
			1.0,
			conflict.PropertyName,
			conflict.Pod.Namespace,
			conflict.Pod.Name,
			conflict.PropertyValue,
			conflict.OtherPod.Namespace,
			conflict.OtherPod.Name,
			conflict.OtherPropertyValue,
		)
	}
}
