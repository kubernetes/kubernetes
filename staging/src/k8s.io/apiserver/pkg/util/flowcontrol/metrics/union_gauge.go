/*
Copyright 2022 The Kubernetes Authors.

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

type unionGauge []Gauge

var _ Gauge = unionGauge(nil)

// NewUnionGauge constructs a Gauge that delegates to all of the given Gauges
func NewUnionGauge(elts ...Gauge) Gauge {
	return unionGauge(elts)
}

func (ug unionGauge) Set(x float64) {
	for _, gauge := range ug {
		gauge.Set(x)
	}
}

func (ug unionGauge) Add(x float64) {
	for _, gauge := range ug {
		gauge.Add(x)
	}
}

func (ug unionGauge) Inc() {
	for _, gauge := range ug {
		gauge.Inc()
	}
}

func (ug unionGauge) Dec() {
	for _, gauge := range ug {
		gauge.Dec()
	}
}

func (ug unionGauge) SetToCurrentTime() {
	for _, gauge := range ug {
		gauge.SetToCurrentTime()
	}
}
