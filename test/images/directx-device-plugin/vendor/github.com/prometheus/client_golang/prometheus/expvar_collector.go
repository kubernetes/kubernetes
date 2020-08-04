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

// NewExpvarCollector returns a newly allocated expvar Collector that still has
// to be registered with a Prometheus registry.
//
// An expvar Collector collects metrics from the expvar interface. It provides a
// quick way to expose numeric values that are already exported via expvar as
// Prometheus metrics. Note that the data models of expvar and Prometheus are
// fundamentally different, and that the expvar Collector is inherently slower
// than native Prometheus metrics. Thus, the expvar Collector is probably great
// for experiments and prototying, but you should seriously consider a more
// direct implementation of Prometheus metrics for monitoring production
// systems.
//
// The exports map has the following meaning:
//
// The keys in the map correspond to expvar keys, i.e. for every expvar key you
// want to export as Prometheus metric, you need an entry in the exports
// map. The descriptor mapped to each key describes how to export the expvar
// value. It defines the name and the help string of the Prometheus metric
// proxying the expvar value. The type will always be Untyped.
//
// For descriptors without variable labels, the expvar value must be a number or
// a bool. The number is then directly exported as the Prometheus sample
// value. (For a bool, 'false' translates to 0 and 'true' to 1). Expvar values
// that are not numbers or bools are silently ignored.
//
// If the descriptor has one variable label, the expvar value must be an expvar
// map. The keys in the expvar map become the various values of the one
// Prometheus label. The values in the expvar map must be numbers or bools again
// as above.
//
// For descriptors with more than one variable label, the expvar must be a
// nested expvar map, i.e. where the values of the topmost map are maps again
// etc. until a depth is reached that corresponds to the number of labels. The
// leaves of that structure must be numbers or bools as above to serve as the
// sample values.
//
// Anything that does not fit into the scheme above is silently ignored.
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
