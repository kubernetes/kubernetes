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
	"reflect"
	"strings"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
)

var (
	// MetricNameLabel is label under which model.Sample stores metric name
	MetricNameLabel model.LabelName = model.MetricNameLabel
	// QuantileLabel is label under which model.Sample stores latency quantile value
	QuantileLabel model.LabelName = model.QuantileLabel
)

// Metrics is generic metrics for other specific metrics
type Metrics map[string]model.Samples

// Equal returns true if all metrics are the same as the arguments.
func (m *Metrics) Equal(o Metrics) bool {
	leftKeySet := []string{}
	rightKeySet := []string{}
	for k := range *m {
		leftKeySet = append(leftKeySet, k)
	}
	for k := range o {
		rightKeySet = append(rightKeySet, k)
	}
	if !reflect.DeepEqual(leftKeySet, rightKeySet) {
		return false
	}
	for _, k := range leftKeySet {
		if !(*m)[k].Equal(o[k]) {
			return false
		}
	}
	return true
}

// NewMetrics returns new metrics which are initialized.
func NewMetrics() Metrics {
	result := make(Metrics)
	return result
}

// ParseMetrics parses Metrics from data returned from prometheus endpoint
func ParseMetrics(data string, output *Metrics) error {
	dec := expfmt.NewDecoder(strings.NewReader(data), expfmt.FmtText)
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	for {
		var v model.Vector
		if err := decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return nil
			}
			continue
		}
		for _, metric := range v {
			name := string(metric.Metric[model.MetricNameLabel])
			(*output)[name] = append((*output)[name], metric)
		}
	}
}

// ExtractMetricSamples parses the prometheus metric samples from the input string.
func ExtractMetricSamples(metricsBlob string) ([]*model.Sample, error) {
	dec := expfmt.NewDecoder(strings.NewReader(metricsBlob), expfmt.FmtText)
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	var samples []*model.Sample
	for {
		var v model.Vector
		if err := decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return samples, nil
			}
			return nil, err
		}
		samples = append(samples, v...)
	}
}

// PrintSample returns formated representation of metric Sample
func PrintSample(sample *model.Sample) string {
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

// ComputeHistogramDelta computes the change in histogram metric for a selected label.
// Results are stored in after samples
func ComputeHistogramDelta(before, after model.Samples, label model.LabelName) {
	beforeSamplesMap := make(map[string]*model.Sample)
	for _, bSample := range before {
		beforeSamplesMap[makeKey(bSample.Metric[label], bSample.Metric["le"])] = bSample
	}
	for _, aSample := range after {
		if bSample, found := beforeSamplesMap[makeKey(aSample.Metric[label], aSample.Metric["le"])]; found {
			aSample.Value = aSample.Value - bSample.Value
		}
	}
}

func makeKey(a, b model.LabelValue) string {
	return string(a) + "___" + string(b)
}
