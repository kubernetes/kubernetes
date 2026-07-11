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
	"fmt"

	e2eperftype "k8s.io/kubernetes/test/e2e/perftype"
)

// APICall is a struct for managing API call.
type APICall struct {
	Resource    string        `json:"resource"`
	Subresource string        `json:"subresource"`
	Verb        string        `json:"verb"`
	Scope       string        `json:"scope"`
	Latency     LatencyMetric `json:"latency"`
	Count       int           `json:"count"`
}

// APIResponsiveness is a struct for managing multiple API calls.
type APIResponsiveness struct {
	APICalls []APICall `json:"apicalls"`
}

// SummaryKind returns the summary of API responsiveness.
func (a *APIResponsiveness) SummaryKind() string {
	return "APIResponsiveness"
}

// PrintHumanReadable returns metrics with JSON format.
func (a *APIResponsiveness) PrintHumanReadable() string {
	return PrettyPrintJSON(a)
}

// PrintJSON returns metrics of PerfData(50, 90 and 99th percentiles) with JSON format.
func (a *APIResponsiveness) PrintJSON() string {
	return PrettyPrintJSON(APICallToPerfData(a))
}

func (a *APIResponsiveness) Len() int { return len(a.APICalls) }
func (a *APIResponsiveness) Swap(i, j int) {
	a.APICalls[i], a.APICalls[j] = a.APICalls[j], a.APICalls[i]
}
func (a *APIResponsiveness) Less(i, j int) bool {
	return a.APICalls[i].Latency.Perc99 < a.APICalls[j].Latency.Perc99
}

// currentAPICallMetricsVersion is the current apicall performance metrics version. We should
// bump up the version each time we make incompatible change to the metrics.
const currentAPICallMetricsVersion = "v1"

// APICallToPerfData transforms APIResponsiveness to PerfData.
func APICallToPerfData(apicalls *APIResponsiveness) *e2eperftype.PerfData {
	perfData := &e2eperftype.PerfData{Version: currentAPICallMetricsVersion}
	for _, apicall := range apicalls.APICalls {
		item := e2eperftype.DataItem{
			Data: map[string]float64{
				"Perc50": float64(apicall.Latency.Perc50) / 1000000, // us -> ms
				"Perc90": float64(apicall.Latency.Perc90) / 1000000,
				"Perc99": float64(apicall.Latency.Perc99) / 1000000,
			},
			Unit: "ms",
			Labels: map[string]string{
				"Verb":        apicall.Verb,
				"Resource":    apicall.Resource,
				"Subresource": apicall.Subresource,
				"Scope":       apicall.Scope,
				"Count":       fmt.Sprintf("%v", apicall.Count),
			},
		}
		perfData.DataItems = append(perfData.DataItems, item)
	}
	return perfData
}
