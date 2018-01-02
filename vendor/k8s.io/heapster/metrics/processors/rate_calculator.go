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

package processors

import (
	"k8s.io/heapster/metrics/core"

	"github.com/golang/glog"
)

type RateCalculator struct {
	rateMetricsMapping map[string]core.Metric
	previousBatch      *core.DataBatch
}

func (this *RateCalculator) Name() string {
	return "rate calculator"
}

func (this *RateCalculator) Process(batch *core.DataBatch) (*core.DataBatch, error) {
	if this.previousBatch == nil {
		this.previousBatch = batch
		return batch, nil
	}
	if !batch.Timestamp.After(this.previousBatch.Timestamp) {
		// something got out of sync, do nothing.
		glog.Errorf("New data batch has timestamp before the previous one: new:%v old:%v", batch.Timestamp, this.previousBatch.Timestamp)
		return batch, nil
	}

	for key, newMs := range batch.MetricSets {

		if oldMs, found := this.previousBatch.MetricSets[key]; found {
			if !newMs.ScrapeTime.After(oldMs.ScrapeTime) {
				// New must be strictly after old.
				continue
			}
			if !newMs.CreateTime.Equal(oldMs.CreateTime) {
				glog.V(4).Infof("Skipping rates for %s - different create time new:%v  old:%v", key, newMs.CreateTime, oldMs.CreateTime)
				// Create time for container must be the same.
				continue
			}

			for metricName, targetMetric := range this.rateMetricsMapping {
				metricValNew, foundNew := newMs.MetricValues[metricName]
				metricValOld, foundOld := oldMs.MetricValues[metricName]
				if foundNew && foundOld {
					if metricName == core.MetricCpuUsage.MetricDescriptor.Name {
						// cpu/usage values are in nanoseconds; we want to have it in millicores (that's why constant 1000 is here).
						newVal := 1000 * (metricValNew.IntValue - metricValOld.IntValue) /
							(newMs.ScrapeTime.UnixNano() - oldMs.ScrapeTime.UnixNano())

						newMs.MetricValues[targetMetric.MetricDescriptor.Name] = core.MetricValue{
							ValueType:  core.ValueInt64,
							MetricType: core.MetricGauge,
							IntValue:   newVal,
						}

					} else if targetMetric.MetricDescriptor.ValueType == core.ValueFloat {
						newVal := 1e9 * float32(metricValNew.IntValue-metricValOld.IntValue) /
							float32(newMs.ScrapeTime.UnixNano()-oldMs.ScrapeTime.UnixNano())

						newMs.MetricValues[targetMetric.MetricDescriptor.Name] = core.MetricValue{
							ValueType:  core.ValueFloat,
							MetricType: core.MetricGauge,
							FloatValue: newVal,
						}
					}
				}
			}
		}
	}
	this.previousBatch = batch
	return batch, nil
}

func NewRateCalculator(metrics map[string]core.Metric) *RateCalculator {
	return &RateCalculator{
		rateMetricsMapping: metrics,
	}
}
