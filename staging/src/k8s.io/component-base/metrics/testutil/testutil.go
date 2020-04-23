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

package testutil

import (
	"fmt"
	"io"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/prometheus/common/model"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/component-base/metrics"
)

// CollectAndCompare registers the provided Collector with a newly created
// pedantic Registry. It then does the same as GatherAndCompare, gathering the
// metrics from the pedantic Registry.
func CollectAndCompare(c metrics.Collector, expected io.Reader, metricNames ...string) error {
	return testutil.CollectAndCompare(c, expected, metricNames...)
}

// GatherAndCompare gathers all metrics from the provided Gatherer and compares
// it to an expected output read from the provided Reader in the Prometheus text
// exposition format. If any metricNames are provided, only metrics with those
// names are compared.
func GatherAndCompare(g metrics.Gatherer, expected io.Reader, metricNames ...string) error {
	return testutil.GatherAndCompare(g, expected, metricNames...)
}

// CustomCollectAndCompare registers the provided StableCollector with a newly created
// registry. It then does the same as GatherAndCompare, gathering the
// metrics from the pedantic Registry.
func CustomCollectAndCompare(c metrics.StableCollector, expected io.Reader, metricNames ...string) error {
	registry := metrics.NewKubeRegistry()
	registry.CustomMustRegister(c)

	return GatherAndCompare(registry, expected, metricNames...)
}

// NewFakeKubeRegistry creates a fake `KubeRegistry` that takes the input version as `build in version`.
// It should only be used in testing scenario especially for the deprecated metrics.
// The input version format should be `major.minor.patch`, e.g. '1.18.0'.
func NewFakeKubeRegistry(ver string) metrics.KubeRegistry {
	backup := metrics.BuildVersion
	defer func() {
		metrics.BuildVersion = backup
	}()

	metrics.BuildVersion = func() apimachineryversion.Info {
		return apimachineryversion.Info{
			GitVersion: fmt.Sprintf("v%s-alpha+1.12345", ver),
		}
	}

	return metrics.NewKubeRegistry()
}

// GetMetricLabelValue returns string that consists of the value of labelName in Metric
func GetMetricLabelValue(element interface{}, labelName ...string) string {
	el := element.(*model.Sample)
	var value string
	var i int
	if len(labelName) > 1 {
		for i = 0; i < len(labelName)-1; i++ {
			value += fmt.Sprintf("%s::", el.Metric[model.LabelName(labelName[i])])
		}
	}
	value += fmt.Sprintf("%s", el.Metric[model.LabelName(labelName[i])])
	return value
}

// GetModelTimeFun returns fun of model.Time
func GetModelTimeFun(sec int64) interface{} {
	return func(t model.Time) time.Time {
		// model.Time is in Milliseconds since epoch
		return time.Unix(sec, int64(t)*int64(time.Millisecond))
	}
}
