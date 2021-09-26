// Copyright 2019 The Prometheus Authors
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
package v1

import (
	"encoding/json"
	"strconv"
	"testing"
	"time"

	jsoniter "github.com/json-iterator/go"

	"github.com/prometheus/common/model"
)

func generateData(timeseries, datapoints int) model.Matrix {
	m := make(model.Matrix, 0)

	for i := 0; i < timeseries; i++ {
		lset := map[model.LabelName]model.LabelValue{
			model.MetricNameLabel: model.LabelValue("timeseries_" + strconv.Itoa(i)),
		}
		now := model.Now()
		values := make([]model.SamplePair, datapoints)

		for x := datapoints; x > 0; x-- {
			values[x-1] = model.SamplePair{
				// Set the time back assuming a 15s interval. Since this is used for
				// Marshal/Unmarshal testing the actual interval doesn't matter.
				Timestamp: now.Add(time.Second * -15 * time.Duration(x)),
				Value:     model.SampleValue(float64(x)),
			}
		}

		ss := &model.SampleStream{
			Metric: model.Metric(lset),
			Values: values,
		}

		m = append(m, ss)
	}
	return m
}

func BenchmarkSamplesJsonSerialization(b *testing.B) {
	for _, timeseriesCount := range []int{10, 100, 1000} {
		b.Run(strconv.Itoa(timeseriesCount), func(b *testing.B) {
			for _, datapointCount := range []int{10, 100, 1000} {
				b.Run(strconv.Itoa(datapointCount), func(b *testing.B) {
					data := generateData(timeseriesCount, datapointCount)

					dataBytes, err := json.Marshal(data)
					if err != nil {
						b.Fatalf("Error marshaling: %v", err)
					}

					b.Run("marshal", func(b *testing.B) {
						b.Run("encoding/json", func(b *testing.B) {
							b.ReportAllocs()
							for i := 0; i < b.N; i++ {
								if _, err := json.Marshal(data); err != nil {
									b.Fatal(err)
								}
							}
						})

						b.Run("jsoniter", func(b *testing.B) {
							b.ReportAllocs()
							for i := 0; i < b.N; i++ {
								if _, err := jsoniter.Marshal(data); err != nil {
									b.Fatal(err)
								}
							}
						})
					})

					b.Run("unmarshal", func(b *testing.B) {
						b.Run("encoding/json", func(b *testing.B) {
							b.ReportAllocs()
							var m model.Matrix
							for i := 0; i < b.N; i++ {
								if err := json.Unmarshal(dataBytes, &m); err != nil {
									b.Fatal(err)
								}
							}
						})

						b.Run("jsoniter", func(b *testing.B) {
							b.ReportAllocs()
							var m model.Matrix
							for i := 0; i < b.N; i++ {
								if err := jsoniter.Unmarshal(dataBytes, &m); err != nil {
									b.Fatal(err)
								}
							}
						})
					})
				})
			}
		})
	}
}
