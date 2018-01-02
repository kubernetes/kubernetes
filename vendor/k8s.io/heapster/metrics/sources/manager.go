// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package sources

import (
	"math/rand"
	"time"

	. "k8s.io/heapster/metrics/core"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
)

const (
	DefaultMetricsScrapeTimeout = 20 * time.Second
	MaxDelayMs                  = 4 * 1000
	DelayPerSourceMs            = 8
)

var (
	// Last time Heapster performed a scrape since unix epoch in seconds.
	lastScrapeTimestamp = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "heapster",
			Subsystem: "scraper",
			Name:      "last_time_seconds",
			Help:      "Last time Heapster performed a scrape since unix epoch in seconds.",
		},
		[]string{"source"},
	)

	// Time spent exporting scraping sources in microseconds..
	scraperDuration = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Namespace: "heapster",
			Subsystem: "scraper",
			Name:      "duration_microseconds",
			Help:      "Time spent scraping sources in microseconds.",
		},
		[]string{"source"},
	)
)

func init() {
	prometheus.MustRegister(lastScrapeTimestamp)
	prometheus.MustRegister(scraperDuration)
}

func NewSourceManager(metricsSourceProvider MetricsSourceProvider, metricsScrapeTimeout time.Duration) (MetricsSource, error) {
	return &sourceManager{
		metricsSourceProvider: metricsSourceProvider,
		metricsScrapeTimeout:  metricsScrapeTimeout,
	}, nil
}

type sourceManager struct {
	metricsSourceProvider MetricsSourceProvider
	metricsScrapeTimeout  time.Duration
}

func (this *sourceManager) Name() string {
	return "source_manager"
}

func (this *sourceManager) ScrapeMetrics(start, end time.Time) *DataBatch {
	glog.V(1).Infof("Scraping metrics start: %s, end: %s", start, end)
	sources := this.metricsSourceProvider.GetMetricsSources()

	responseChannel := make(chan *DataBatch)
	startTime := time.Now()
	timeoutTime := startTime.Add(this.metricsScrapeTimeout)

	delayMs := DelayPerSourceMs * len(sources)
	if delayMs > MaxDelayMs {
		delayMs = MaxDelayMs
	}

	for _, source := range sources {

		go func(source MetricsSource, channel chan *DataBatch, start, end, timeoutTime time.Time, delayInMs int) {

			// Prevents network congestion.
			time.Sleep(time.Duration(rand.Intn(delayMs)) * time.Millisecond)

			glog.V(2).Infof("Querying source: %s", source)
			metrics := scrape(source, start, end)
			now := time.Now()
			if !now.Before(timeoutTime) {
				glog.Warningf("Failed to get %s response in time", source)
				return
			}
			timeForResponse := timeoutTime.Sub(now)

			select {
			case channel <- metrics:
				// passed the response correctly.
				return
			case <-time.After(timeForResponse):
				glog.Warningf("Failed to send the response back %s", source)
				return
			}
		}(source, responseChannel, start, end, timeoutTime, delayMs)
	}
	response := DataBatch{
		Timestamp:  end,
		MetricSets: map[string]*MetricSet{},
	}

	latencies := make([]int, 11)

responseloop:
	for i := range sources {
		now := time.Now()
		if !now.Before(timeoutTime) {
			glog.Warningf("Failed to get all responses in time (got %d/%d)", i, len(sources))
			break
		}

		select {
		case dataBatch := <-responseChannel:
			if dataBatch != nil {
				for key, value := range dataBatch.MetricSets {
					response.MetricSets[key] = value
				}
			}
			latency := now.Sub(startTime)
			bucket := int(latency.Seconds())
			if bucket >= len(latencies) {
				bucket = len(latencies) - 1
			}
			latencies[bucket]++

		case <-time.After(timeoutTime.Sub(now)):
			glog.Warningf("Failed to get all responses in time (got %d/%d)", i, len(sources))
			break responseloop
		}
	}

	glog.V(1).Infof("ScrapeMetrics: time: %s size: %d", time.Since(startTime), len(response.MetricSets))
	for i, value := range latencies {
		glog.V(1).Infof("   scrape  bucket %d: %d", i, value)
	}
	return &response
}

func scrape(s MetricsSource, start, end time.Time) *DataBatch {
	sourceName := s.Name()
	startTime := time.Now()
	defer lastScrapeTimestamp.
		WithLabelValues(sourceName).
		Set(float64(time.Now().Unix()))
	defer scraperDuration.
		WithLabelValues(sourceName).
		Observe(float64(time.Since(startTime)) / float64(time.Microsecond))

	return s.ScrapeMetrics(start, end)
}
