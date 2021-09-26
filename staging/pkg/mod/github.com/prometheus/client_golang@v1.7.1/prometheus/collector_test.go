// Copyright 2018 The Prometheus Authors
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

import "testing"

type collectorDescribedByCollect struct {
	cnt Counter
	gge Gauge
}

func (c collectorDescribedByCollect) Collect(ch chan<- Metric) {
	ch <- c.cnt
	ch <- c.gge
}

func (c collectorDescribedByCollect) Describe(ch chan<- *Desc) {
	DescribeByCollect(c, ch)
}

func TestDescribeByCollect(t *testing.T) {

	goodCollector := collectorDescribedByCollect{
		cnt: NewCounter(CounterOpts{Name: "c1", Help: "help c1"}),
		gge: NewGauge(GaugeOpts{Name: "g1", Help: "help g1"}),
	}
	collidingCollector := collectorDescribedByCollect{
		cnt: NewCounter(CounterOpts{Name: "c2", Help: "help c2"}),
		gge: NewGauge(GaugeOpts{Name: "g1", Help: "help g1"}),
	}
	inconsistentCollector := collectorDescribedByCollect{
		cnt: NewCounter(CounterOpts{Name: "c3", Help: "help c3"}),
		gge: NewGauge(GaugeOpts{Name: "c3", Help: "help inconsistent"}),
	}

	reg := NewPedanticRegistry()

	if err := reg.Register(goodCollector); err != nil {
		t.Error("registration failed:", err)
	}
	if err := reg.Register(collidingCollector); err == nil {
		t.Error("registration unexpectedly succeeded")
	}
	if err := reg.Register(inconsistentCollector); err == nil {
		t.Error("registration unexpectedly succeeded")
	}

	if _, err := reg.Gather(); err != nil {
		t.Error("gathering failed:", err)
	}
}
