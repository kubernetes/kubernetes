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
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/prometheus/common/model"

	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

const (
	// Cluster Autoscaler metrics names
	caFunctionMetric      = "cluster_autoscaler_function_duration_seconds_bucket"
	caFunctionMetricLabel = "function"
)

// ComponentCollection is metrics collection of components.
type ComponentCollection Collection

func (m *ComponentCollection) filterMetrics() {
	apiServerMetrics := make(APIServerMetrics)
	for _, metric := range interestingAPIServerMetrics {
		apiServerMetrics[metric] = (*m).APIServerMetrics[metric]
	}
	controllerManagerMetrics := make(ControllerManagerMetrics)
	for _, metric := range interestingControllerManagerMetrics {
		controllerManagerMetrics[metric] = (*m).ControllerManagerMetrics[metric]
	}
	kubeletMetrics := make(map[string]KubeletMetrics)
	for kubelet, grabbed := range (*m).KubeletMetrics {
		kubeletMetrics[kubelet] = make(KubeletMetrics)
		for _, metric := range interestingKubeletMetrics {
			kubeletMetrics[kubelet][metric] = grabbed[metric]
		}
	}
	(*m).APIServerMetrics = apiServerMetrics
	(*m).ControllerManagerMetrics = controllerManagerMetrics
	(*m).KubeletMetrics = kubeletMetrics
}

func printSample(sample *model.Sample) string {
	buf := make([]string, 0)
	// Id is a VERY special label. For 'normal' container it's useless, but it's necessary
	// for 'system' containers (e.g. /docker-daemon, /kubelet, etc.). We know if that's the
	// case by checking if there's a label "kubernetes_container_name" present. It's hacky
	// but it works...
	_, normalContainer := sample.Metric["kubernetes_container_name"]
	for k, v := range sample.Metric {
		if strings.HasPrefix(string(k), "__") {
			continue
		}

		if string(k) == "id" && normalContainer {
			continue
		}
		buf = append(buf, fmt.Sprintf("%v=%v", string(k), v))
	}
	return fmt.Sprintf("[%v] = %v", strings.Join(buf, ","), sample.Value)
}

// PrintHumanReadable returns e2e metrics with JSON format.
func (m *ComponentCollection) PrintHumanReadable() string {
	buf := bytes.Buffer{}
	for _, interestingMetric := range interestingAPIServerMetrics {
		buf.WriteString(fmt.Sprintf("For %v:\n", interestingMetric))
		for _, sample := range (*m).APIServerMetrics[interestingMetric] {
			buf.WriteString(fmt.Sprintf("\t%v\n", printSample(sample)))
		}
	}
	for _, interestingMetric := range interestingControllerManagerMetrics {
		buf.WriteString(fmt.Sprintf("For %v:\n", interestingMetric))
		for _, sample := range (*m).ControllerManagerMetrics[interestingMetric] {
			buf.WriteString(fmt.Sprintf("\t%v\n", printSample(sample)))
		}
	}
	for _, interestingMetric := range interestingClusterAutoscalerMetrics {
		buf.WriteString(fmt.Sprintf("For %v:\n", interestingMetric))
		for _, sample := range (*m).ClusterAutoscalerMetrics[interestingMetric] {
			buf.WriteString(fmt.Sprintf("\t%v\n", printSample(sample)))
		}
	}
	for kubelet, grabbed := range (*m).KubeletMetrics {
		buf.WriteString(fmt.Sprintf("For %v:\n", kubelet))
		for _, interestingMetric := range interestingKubeletMetrics {
			buf.WriteString(fmt.Sprintf("\tFor %v:\n", interestingMetric))
			for _, sample := range grabbed[interestingMetric] {
				buf.WriteString(fmt.Sprintf("\t\t%v\n", printSample(sample)))
			}
		}
	}
	return buf.String()
}

// PrettyPrintJSON converts metrics to JSON format.
func PrettyPrintJSON(metrics interface{}) string {
	output := &bytes.Buffer{}
	if err := json.NewEncoder(output).Encode(metrics); err != nil {
		e2elog.Logf("Error building encoder: %v", err)
		return ""
	}
	formatted := &bytes.Buffer{}
	if err := json.Indent(formatted, output.Bytes(), "", "  "); err != nil {
		e2elog.Logf("Error indenting: %v", err)
		return ""
	}
	return string(formatted.Bytes())
}

// PrintJSON returns e2e metrics with JSON format.
func (m *ComponentCollection) PrintJSON() string {
	m.filterMetrics()
	return PrettyPrintJSON(m)
}

// SummaryKind returns the summary of e2e metrics.
func (m *ComponentCollection) SummaryKind() string {
	return "ComponentCollection"
}

func makeKey(a, b model.LabelValue) string {
	return string(a) + "___" + string(b)
}

// ComputeClusterAutoscalerMetricsDelta computes the change in cluster
// autoscaler metrics.
func (m *ComponentCollection) ComputeClusterAutoscalerMetricsDelta(before Collection) {
	if beforeSamples, found := before.ClusterAutoscalerMetrics[caFunctionMetric]; found {
		if afterSamples, found := m.ClusterAutoscalerMetrics[caFunctionMetric]; found {
			beforeSamplesMap := make(map[string]*model.Sample)
			for _, bSample := range beforeSamples {
				beforeSamplesMap[makeKey(bSample.Metric[caFunctionMetricLabel], bSample.Metric["le"])] = bSample
			}
			for _, aSample := range afterSamples {
				if bSample, found := beforeSamplesMap[makeKey(aSample.Metric[caFunctionMetricLabel], aSample.Metric["le"])]; found {
					aSample.Value = aSample.Value - bSample.Value
				}

			}
		}
	}
}
