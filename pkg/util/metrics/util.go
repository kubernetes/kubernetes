/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/flowcontrol"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
)

const (
	updatePeriod = 5 * time.Second
)

var (
	metricsLock        sync.Mutex
	rateLimiterMetrics = make(map[string]prometheus.Gauge)
)

func registerRateLimiterMetric(ownerName string) error {
	metricsLock.Lock()
	defer metricsLock.Unlock()

	if _, ok := rateLimiterMetrics[ownerName]; ok {
		glog.Errorf("Metric for %v already registered", ownerName)
		return fmt.Errorf("Metric for %v already registered", ownerName)
	}
	metric := prometheus.NewGauge(prometheus.GaugeOpts{
		Name:      "rate_limiter_use",
		Subsystem: ownerName,
		Help:      fmt.Sprintf("A metric measuring the saturation of the rate limiter for %v", ownerName),
	})
	rateLimiterMetrics[ownerName] = metric
	prometheus.MustRegister(metric)
	return nil
}

// RegisterMetricAndTrackRateLimiterUsage registers a metric ownerName_rate_limiter_use in prometheus to track
// how much used rateLimiter is and starts a goroutine that updates this metric every updatePeriod
func RegisterMetricAndTrackRateLimiterUsage(ownerName string, rateLimiter flowcontrol.RateLimiter) error {
	err := registerRateLimiterMetric(ownerName)
	if err != nil {
		return err
	}
	go wait.Forever(func() {
		metricsLock.Lock()
		defer metricsLock.Unlock()
		rateLimiterMetrics[ownerName].Set(rateLimiter.Saturation())
	}, updatePeriod)
	return nil
}
