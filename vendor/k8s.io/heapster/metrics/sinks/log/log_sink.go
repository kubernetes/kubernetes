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

package logsink

import (
	"bytes"
	"fmt"
	"sort"

	"github.com/golang/glog"
	"k8s.io/heapster/metrics/core"
)

type LogSink struct {
}

func (this *LogSink) Name() string {
	return "Log Sink"
}

func (this *LogSink) Stop() {
	// Do nothing.
}

func batchToString(batch *core.DataBatch) string {
	var buffer bytes.Buffer
	buffer.WriteString(fmt.Sprintf("DataBatch     Timestamp: %s\n\n", batch.Timestamp))
	for _, key := range sortedMetricSetKeys(batch.MetricSets) {
		ms := batch.MetricSets[key]
		buffer.WriteString(fmt.Sprintf("MetricSet: %s\n", key))
		padding := "   "
		buffer.WriteString(fmt.Sprintf("%sScrape time: %v %v\n", padding, ms.ScrapeTime, ms.ScrapeTime.UnixNano()))
		buffer.WriteString(fmt.Sprintf("%sCreate time: %v %v\n", padding, ms.CreateTime, ms.CreateTime.UnixNano()))
		buffer.WriteString(fmt.Sprintf("%sLabels:\n", padding))
		for _, labelName := range sortedLabelKeys(ms.Labels) {
			labelValue := ms.Labels[labelName]
			buffer.WriteString(fmt.Sprintf("%s%s%s = %s\n", padding, padding, labelName, labelValue))
		}
		buffer.WriteString(fmt.Sprintf("%sMetrics:\n", padding))
		for _, metricName := range sortedMetricValueKeys(ms.MetricValues) {
			metricValue := ms.MetricValues[metricName]
			if core.ValueInt64 == metricValue.ValueType {
				buffer.WriteString(fmt.Sprintf("%s%s%s = %d\n", padding, padding, metricName, metricValue.IntValue))
			} else if core.ValueFloat == metricValue.ValueType {
				buffer.WriteString(fmt.Sprintf("%s%s%s = %f\n", padding, padding, metricName, metricValue.FloatValue))
			} else {
				buffer.WriteString(fmt.Sprintf("%s%s%s = ?\n", padding, padding, metricName))
			}
		}
		buffer.WriteString(fmt.Sprintf("%sLabeled Metrics:\n", padding))
		for _, metric := range ms.LabeledMetrics {
			if core.ValueInt64 == metric.ValueType {
				buffer.WriteString(fmt.Sprintf("%s%s%s = %d\n", padding, padding, metric.Name, metric.IntValue))
			} else if core.ValueFloat == metric.ValueType {
				buffer.WriteString(fmt.Sprintf("%s%s%s = %f\n", padding, padding, metric.Name, metric.FloatValue))
			} else {
				buffer.WriteString(fmt.Sprintf("%s%s%s = ?\n", padding, padding, metric.Name))
			}
			for labelName, labelValue := range metric.Labels {
				buffer.WriteString(fmt.Sprintf("%s%s%s%s = %s\n", padding, padding, padding, labelName, labelValue))
			}
		}
		buffer.WriteString("\n")
	}
	return buffer.String()
}

func (this *LogSink) ExportData(batch *core.DataBatch) {
	glog.Info(batchToString(batch))
}

func NewLogSink() *LogSink {
	return &LogSink{}
}

func sortedMetricSetKeys(m map[string]*core.MetricSet) []string {
	keys := make([]string, len(m))
	i := 0
	for k := range m {
		keys[i] = k
		i++
	}
	sort.Strings(keys)
	return keys
}

func sortedLabelKeys(m map[string]string) []string {
	keys := make([]string, len(m))
	i := 0
	for k := range m {
		keys[i] = k
		i++
	}
	sort.Strings(keys)
	return keys
}

func sortedMetricValueKeys(m map[string]core.MetricValue) []string {
	keys := make([]string, len(m))
	i := 0
	for k := range m {
		keys[i] = k
		i++
	}
	sort.Strings(keys)
	return keys
}
