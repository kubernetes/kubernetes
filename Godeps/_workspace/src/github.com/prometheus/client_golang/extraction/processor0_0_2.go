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

	"github.com/prometheus/client_golang/model"
)

// Processor002 is responsible for decoding payloads from protocol version
// 0.0.2.
var Processor002 = &processor002{}

type histogram002 struct {
	Labels map[string]string            `json:"labels"`
	Values map[string]model.SampleValue `json:"value"`
}

type counter002 struct {
	Labels map[string]string `json:"labels"`
	Value  model.SampleValue `json:"value"`
}

type processor002 struct{}

func (p *processor002) ProcessSingle(in io.Reader, out Ingester, o *ProcessOptions) error {
	// Processor for telemetry schema version 0.0.2.
	// container for telemetry data
	var entities []struct {
		BaseLabels map[string]string `json:"baseLabels"`
		Docstring  string            `json:"docstring"`
		Metric     struct {
			Type   string          `json:"type"`
			Values json.RawMessage `json:"value"`
		} `json:"metric"`
	}

	if err := json.NewDecoder(in).Decode(&entities); err != nil {
		return err
	}

	pendingSamples := model.Samples{}
	for _, entity := range entities {
		switch entity.Metric.Type {
		case "counter", "gauge":
			var values []counter002

			if err := json.Unmarshal(entity.Metric.Values, &values); err != nil {
				return fmt.Errorf("could not extract %s value: %s", entity.Metric.Type, err)
			}

			for _, counter := range values {
				labels := labelSet(entity.BaseLabels).Merge(labelSet(counter.Labels))

				pendingSamples = append(pendingSamples, &model.Sample{
					Metric:    model.Metric(labels),
					Timestamp: o.Timestamp,
					Value:     counter.Value,
				})
			}

		case "histogram":
			var values []histogram002

			if err := json.Unmarshal(entity.Metric.Values, &values); err != nil {
				return fmt.Errorf("could not extract %s value: %s", entity.Metric.Type, err)
			}

			for _, histogram := range values {
				for percentile, value := range histogram.Values {
					labels := labelSet(entity.BaseLabels).Merge(labelSet(histogram.Labels))
					labels[model.LabelName("percentile")] = model.LabelValue(percentile)

					pendingSamples = append(pendingSamples, &model.Sample{
						Metric:    model.Metric(labels),
						Timestamp: o.Timestamp,
						Value:     value,
					})
				}
			}

		default:
			return fmt.Errorf("unknown metric type %q", entity.Metric.Type)
		}
	}

	if len(pendingSamples) > 0 {
		return out.Ingest(pendingSamples)
	}

	return nil
}
