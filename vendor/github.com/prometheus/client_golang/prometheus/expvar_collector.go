// Copyright 2014 The Prometheus Authors
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

package prometheus

import (
	"encoding/json"
	"expvar"
)

type expvarCollector struct {
	exports map[string]*Desc
}

// NewExpvarCollector is the obsolete version of collectors.NewExpvarCollector.
// See there for documentation.
//
// Deprecated: Use collectors.NewExpvarCollector instead.
func NewExpvarCollector(exports map[string]*Desc) Collector {
	return &expvarCollector{
		exports: exports,
	}
}

// Describe implements Collector.
func (e *expvarCollector) Describe(ch chan<- *Desc) {
	for _, desc := range e.exports {
		ch <- desc
	}
}

// Collect implements Collector.
func (e *expvarCollector) Collect(ch chan<- Metric) {
	for name, desc := range e.exports {
		var m Metric
		expVar := expvar.Get(name)
		if expVar == nil {
			continue
		}
		var v interface{}
		labels := make([]string, len(desc.variableLabels))
		if err := json.Unmarshal([]byte(expVar.String()), &v); err != nil {
			ch <- NewInvalidMetric(desc, err)
			continue
		}
		var processValue func(v interface{}, i int)
		processValue = func(v interface{}, i int) {
			if i >= len(labels) {
				copiedLabels := append(make([]string, 0, len(labels)), labels...)
				switch v := v.(type) {
				case float64:
					m = MustNewConstMetric(desc, UntypedValue, v, copiedLabels...)
				case bool:
					if v {
						m = MustNewConstMetric(desc, UntypedValue, 1, copiedLabels...)
					} else {
						m = MustNewConstMetric(desc, UntypedValue, 0, copiedLabels...)
					}
				default:
					return
				}
				ch <- m
				return
			}
			vm, ok := v.(map[string]interface{})
			if !ok {
				return
			}
			for lv, val := range vm {
				labels[i] = lv
				processValue(val, i+1)
			}
		}
		processValue(v, 0)
	}
}
