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

package conversion

import (
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	latencyBuckets = prometheus.ExponentialBuckets(0.001, 2, 15)
)

// converterMetricFactory holds metrics for all CRD converters
type converterMetricFactory struct {
	// A map from a converter name to it's metric. Allows the converterMetric to be created
	// again with the same metric for a specific converter (e.g. 'webhook').
	durations   map[string]*prometheus.HistogramVec
	factoryLock sync.Mutex
}

func newConverterMertricFactory() *converterMetricFactory {
	return &converterMetricFactory{durations: map[string]*prometheus.HistogramVec{}, factoryLock: sync.Mutex{}}
}

var _ crConverterInterface = &converterMetric{}

type converterMetric struct {
	delegate  crConverterInterface
	latencies *prometheus.HistogramVec
	crdName   string
}

func (c *converterMetricFactory) addMetrics(converterName string, crdName string, converter crConverterInterface) (crConverterInterface, error) {
	c.factoryLock.Lock()
	defer c.factoryLock.Unlock()
	metric, exists := c.durations[converterName]
	if !exists {
		metric = prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    fmt.Sprintf("apiserver_crd_%s_conversion_duration_seconds", converterName),
				Help:    fmt.Sprintf("CRD %s conversion duration in seconds", converterName),
				Buckets: latencyBuckets,
			},
			[]string{"crd_name", "from_version", "to_version", "succeeded"})
		err := prometheus.Register(metric)
		if err != nil {
			return nil, err
		}
		c.durations[converterName] = metric
	}
	return &converterMetric{latencies: metric, delegate: converter, crdName: crdName}, nil
}

func (m *converterMetric) Convert(in runtime.Object, targetGV schema.GroupVersion) (runtime.Object, error) {
	start := time.Now()
	obj, err := m.delegate.Convert(in, targetGV)
	fromVersion := in.GetObjectKind().GroupVersionKind().Version
	toVersion := targetGV.Version

	// only record this observation if the version is different
	if fromVersion != toVersion {
		m.latencies.WithLabelValues(
			m.crdName, fromVersion, toVersion, strconv.FormatBool(err == nil)).Observe(time.Since(start).Seconds())
	}
	return obj, err
}
