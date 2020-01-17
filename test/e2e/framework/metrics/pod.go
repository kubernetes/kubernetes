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
	"time"

	e2eperftype "k8s.io/kubernetes/test/e2e/perftype"
)

// LatencyMetric is a struct for dashboard metrics.
type LatencyMetric struct {
	Perc50  time.Duration `json:"Perc50"`
	Perc90  time.Duration `json:"Perc90"`
	Perc99  time.Duration `json:"Perc99"`
	Perc100 time.Duration `json:"Perc100"`
}

// PodStartupLatency is a struct for managing latency of pod startup.
type PodStartupLatency struct {
	CreateToScheduleLatency LatencyMetric `json:"createToScheduleLatency"`
	ScheduleToRunLatency    LatencyMetric `json:"scheduleToRunLatency"`
	RunToWatchLatency       LatencyMetric `json:"runToWatchLatency"`
	ScheduleToWatchLatency  LatencyMetric `json:"scheduleToWatchLatency"`
	E2ELatency              LatencyMetric `json:"e2eLatency"`
}

// SummaryKind returns the summary of pod startup latency.
func (l *PodStartupLatency) SummaryKind() string {
	return "PodStartupLatency"
}

// PrintHumanReadable returns pod startup letency with JSON format.
func (l *PodStartupLatency) PrintHumanReadable() string {
	return PrettyPrintJSON(l)
}

// PrintJSON returns pod startup letency with JSON format.
func (l *PodStartupLatency) PrintJSON() string {
	return PrettyPrintJSON(PodStartupLatencyToPerfData(l))
}

func latencyToPerfData(l LatencyMetric, name string) e2eperftype.DataItem {
	return e2eperftype.DataItem{
		Data: map[string]float64{
			"Perc50":  float64(l.Perc50) / 1000000, // us -> ms
			"Perc90":  float64(l.Perc90) / 1000000,
			"Perc99":  float64(l.Perc99) / 1000000,
			"Perc100": float64(l.Perc100) / 1000000,
		},
		Unit: "ms",
		Labels: map[string]string{
			"Metric": name,
		},
	}
}

// PodStartupLatencyToPerfData transforms PodStartupLatency to PerfData.
func PodStartupLatencyToPerfData(latency *PodStartupLatency) *e2eperftype.PerfData {
	perfData := &e2eperftype.PerfData{Version: currentAPICallMetricsVersion}
	perfData.DataItems = append(perfData.DataItems, latencyToPerfData(latency.CreateToScheduleLatency, "create_to_schedule"))
	perfData.DataItems = append(perfData.DataItems, latencyToPerfData(latency.ScheduleToRunLatency, "schedule_to_run"))
	perfData.DataItems = append(perfData.DataItems, latencyToPerfData(latency.RunToWatchLatency, "run_to_watch"))
	perfData.DataItems = append(perfData.DataItems, latencyToPerfData(latency.ScheduleToWatchLatency, "schedule_to_watch"))
	perfData.DataItems = append(perfData.DataItems, latencyToPerfData(latency.E2ELatency, "pod_startup"))
	return perfData
}
