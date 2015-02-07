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
	"math"
	"testing"

	dto "github.com/prometheus/client_model/go"
)

func TestCounterAdd(t *testing.T) {
	counter := NewCounter(CounterOpts{
		Name:        "test",
		Help:        "test help",
		ConstLabels: Labels{"a": "1", "b": "2"},
	}).(*counter)
	counter.Inc()
	if expected, got := 1., math.Float64frombits(counter.valBits); expected != got {
		t.Errorf("Expected %f, got %f.", expected, got)
	}
	counter.Add(42)
	if expected, got := 43., math.Float64frombits(counter.valBits); expected != got {
		t.Errorf("Expected %f, got %f.", expected, got)
	}

	if expected, got := "counter cannot decrease in value", decreaseCounter(counter).Error(); expected != got {
		t.Errorf("Expected error %q, got %q.", expected, got)
	}

	m := &dto.Metric{}
	counter.Write(m)

	if expected, got := `label:<name:"a" value:"1" > label:<name:"b" value:"2" > counter:<value:43 > `, m.String(); expected != got {
		t.Errorf("expected %q, got %q", expected, got)
	}
}

func decreaseCounter(c *counter) (err error) {
	defer func() {
		if e := recover(); e != nil {
			err = e.(error)
		}
	}()
	c.Add(-1)
	return nil
}
