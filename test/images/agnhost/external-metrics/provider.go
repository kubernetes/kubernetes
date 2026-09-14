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

package externalmetrics

import (
	"fmt"
	"strings"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type externalMetric struct {
	metricName string
	labels     map[string]string
	value      int
	shouldFail bool        // If true, this metric returns errors
	timestamp  metav1.Time // for mocking we are returning only the current time
}

type metricProvider struct {
	externalMetricsLock sync.RWMutex
	externalMetrics     map[string]*externalMetric
}

func NewConfigurableProvider() *metricProvider {
	p := &metricProvider{
		externalMetrics:     make(map[string]*externalMetric),
		externalMetricsLock: sync.RWMutex{},
	}
	return p
}

// buildMetricKey creates a unique key combining metric name and labels
// This allows multiple instances of the same metric with different labels
func buildMetricKey(metricName string, metricLabels map[string]string) string {
	key := metricName
	if len(metricLabels) > 0 {
		var labelPairs []string
		for k, v := range metricLabels {
			labelPairs = append(labelPairs, k+"="+v)
		}
		key = metricName + ":" + strings.Join(labelPairs, ",")
	}
	return key
}

// parseLabels parses label selector from query string format
// Example: "env=prod,region=us-west"
func parseLabels(labelSelector string) map[string]string {
	labels := make(map[string]string)
	if labelSelector == "" {
		return labels
	}

	pairs := strings.Split(labelSelector, ",")
	for _, pair := range pairs {
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) == 2 {
			labels[strings.TrimSpace(kv[0])] = strings.TrimSpace(kv[1])
		}
	}
	return labels
}

// setMetricFailure controls whether a specific metric should fail
func (p *metricProvider) setMetricFailure(metricKey string, failStr string) error {
	shouldFail := failStr == "true"

	p.externalMetricsLock.Lock()
	defer p.externalMetricsLock.Unlock()

	metric, exists := p.externalMetrics[metricKey]
	if !exists {
		return fmt.Errorf("metric %s not found", metricKey)
	}
	metric.shouldFail = shouldFail
	return nil
}

// setMetricFailure controls whether a specific metric should fail
func (p *metricProvider) setMetricValue(metricKey string, metricValue int) error {
	p.externalMetricsLock.Lock()
	defer p.externalMetricsLock.Unlock()
	metric, exists := p.externalMetrics[metricKey]
	if !exists {
		return fmt.Errorf("metric %s not found", metricKey)
	}
	metric.value = metricValue
	return nil
}

func (p *metricProvider) createMetric(metricName string, metricLabels map[string]string, metricValue int, shouldFail bool) {
	p.externalMetricsLock.Lock()
	defer p.externalMetricsLock.Unlock()
	key := buildMetricKey(metricName, metricLabels)
	p.externalMetrics[key] = &externalMetric{
		metricName: metricName,
		labels:     metricLabels,
		value:      metricValue,
		shouldFail: shouldFail,
		timestamp:  metav1.Now(),
	}
}

// getMetrics returns all metrics matching the given name and label selector
func (p *metricProvider) getMetrics(metricName string, labelSelector map[string]string) []*externalMetric {
	p.externalMetricsLock.RLock()
	defer p.externalMetricsLock.RUnlock()

	var results []*externalMetric

	// If label selector is provided, try exact match first
	if len(labelSelector) > 0 {
		key := buildMetricKey(metricName, labelSelector)
		if metric, exists := p.externalMetrics[key]; exists {
			results = append(results, metric)
			return results
		}
	}

	// Otherwise, find all metrics with the given name
	for _, metric := range p.externalMetrics {
		if metric.metricName != metricName {
			continue
		}

		// If no label selector, include all metrics with this name
		if len(labelSelector) == 0 {
			results = append(results, metric)
			continue
		}

		// Check if metric labels match the selector
		matches := true
		for selectorKey, selectorValue := range labelSelector {
			if metricValue, exists := metric.labels[selectorKey]; !exists || metricValue != selectorValue {
				matches = false
				break
			}
		}
		if matches {
			results = append(results, metric)
		}
	}

	// For backward compatibility, also check if metricName exists as a key without labels
	if len(results) == 0 && len(labelSelector) == 0 {
		if metric, exists := p.externalMetrics[metricName]; exists {
			results = append(results, metric)
		}
	}

	return results
}
