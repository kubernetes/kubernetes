/*
Copyright 2020 The Kubernetes Authors.

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

package fairqueuing

import (
	"math"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
)

func TestIntegrator(t *testing.T) {
	now := time.Now()
	clk := clock.NewFakeClock(now)
	igr := NewIntegrator(clk)
	igr.Add(3)
	clk.Step(time.Second)
	results := igr.GetResults()
	rToo := igr.Reset()
	if e := (IntegratorResults{Duration: time.Second.Seconds(), Average: 3, Deviation: 0, Min: 0, Max: 3}); !e.Equal(&results) {
		t.Errorf("expected %#+v, got %#+v", e, results)
	}
	if !results.Equal(&rToo) {
		t.Errorf("expected %#+v, got %#+v", results, rToo)
	}
	igr.Set(2)
	results = igr.GetResults()
	if e := (IntegratorResults{Duration: 0, Average: math.NaN(), Deviation: math.NaN(), Min: 2, Max: 3}); !e.Equal(&results) {
		t.Errorf("expected %#+v, got %#+v", e, results)
	}
	clk.Step(time.Millisecond)
	igr.Add(-1)
	clk.Step(time.Millisecond)
	results = igr.GetResults()
	if e := (IntegratorResults{Duration: 2 * time.Millisecond.Seconds(), Average: 1.5, Deviation: 0.5, Min: 1, Max: 3}); !e.Equal(&results) {
		t.Errorf("expected %#+v, got %#+v", e, results)
	}
}
