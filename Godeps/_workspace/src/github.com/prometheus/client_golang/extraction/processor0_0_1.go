// Copyright 2013 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package extraction

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"

	"github.com/prometheus/client_golang/model"
)

const (
	baseLabels001 = "baseLabels"
	counter001    = "counter"
	docstring001  = "docstring"
	gauge001      = "gauge"
	histogram001  = "histogram"
	labels001     = "labels"
	metric001     = "metric"
	type001       = "type"
	value001      = "value"
	percentile001 = "percentile"
)

// Processor001 is responsible for decoding payloads from protocol version
// 0.0.1.
var Processor001 = &processor001{}

// processor001 is responsible for handling API version 0.0.1.
type processor001 struct{}

// entity001 represents a the JSON structure that 0.0.1 uses.
type entity001 []struct {
	BaseLabels map[string]string `json:"baseLabels"`
	Docstring  string            `json:"docstring"`
	Metric     struct {
		MetricType string `json:"type"`
		Value      []struct {
			Labels map[string]string `json:"labels"`
			Value  interface{}       `json:"value"`
		} `json:"value"`
	} `json:"metric"`
}

func (p *processor001) ProcessSingle(in io.Reader, out Ingester, o *ProcessOptions) error {
	// TODO(matt): Replace with plain-jane JSON unmarshalling.
	buffer, err := ioutil.ReadAll(in)
	if err != nil {
		return err
	}

	entities := entity001{}
	if err = json.Unmarshal(buffer, &entities); err != nil {
		return err
	}

	// TODO(matt): This outer loop is a great basis for parallelization.
	pendingSamples := model.Samples{}
	for _, entity := range entities {
		for _, value := range entity.Metric.Value {
			labels := labelSet(entity.BaseLabels).Merge(labelSet(value.Labels))

			switch entity.Metric.MetricType {
			case gauge001, counter001:
				sampleValue, ok := value.Value.(float64)
				if !ok {
					return fmt.Errorf("could not convert value from %s %s to float64", entity, value)
				}

				pendingSamples = append(pendingSamples, &model.Sample{
					Metric:    model.Metric(labels),
					Timestamp: o.Timestamp,
					Value:     model.SampleValue(sampleValue),
				})

				break

			case histogram001:
				sampleValue, ok := value.Value.(map[string]interface{})
				if !ok {
					return fmt.Errorf("could not convert value from %q to a map[string]interface{}", value.Value)
				}

				for percentile, percentileValue := range sampleValue {
					individualValue, ok := percentileValue.(float64)
					if !ok {
						return fmt.Errorf("could not convert value from %q to a float64", percentileValue)
					}

					childMetric := make(map[model.LabelName]model.LabelValue, len(labels)+1)

					for k, v := range labels {
						childMetric[k] = v
					}

					childMetric[model.LabelName(percentile001)] = model.LabelValue(percentile)

					pendingSamples = append(pendingSamples, &model.Sample{
						Metric:    model.Metric(childMetric),
						Timestamp: o.Timestamp,
						Value:     model.SampleValue(individualValue),
					})
				}

				break
			}
		}
	}
	if len(pendingSamples) > 0 {
		return out.Ingest(pendingSamples)
	}

	return nil
}
