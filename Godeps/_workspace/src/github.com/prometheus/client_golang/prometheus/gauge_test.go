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
	"math/rand"
	"sync"
	"testing"
	"testing/quick"

	dto "github.com/prometheus/client_model/go"
)

func listenGaugeStream(vals, result chan float64, done chan struct{}) {
	var sum float64
outer:
	for {
		select {
		case <-done:
			close(vals)
			for v := range vals {
				sum += v
			}
			break outer
		case v := <-vals:
			sum += v
		}
	}
	result <- sum
	close(result)
}

func TestGaugeConcurrency(t *testing.T) {
	it := func(n uint32) bool {
		mutations := int(n % 10000)
		concLevel := int(n%15 + 1)

		var start, end sync.WaitGroup
		start.Add(1)
		end.Add(concLevel)

		sStream := make(chan float64, mutations*concLevel)
		result := make(chan float64)
		done := make(chan struct{})

		go listenGaugeStream(sStream, result, done)
		go func() {
			end.Wait()
			close(done)
		}()

		gge := NewGauge(GaugeOpts{
			Name: "test_gauge",
			Help: "no help can be found here",
		})
		for i := 0; i < concLevel; i++ {
			vals := make([]float64, mutations)
			for j := 0; j < mutations; j++ {
				vals[j] = rand.Float64() - 0.5
			}

			go func(vals []float64) {
				start.Wait()
				for _, v := range vals {
					sStream <- v
					gge.Add(v)
				}
				end.Done()
			}(vals)
		}
		start.Done()

		if expected, got := <-result, math.Float64frombits(gge.(*value).valBits); math.Abs(expected-got) > 0.000001 {
			t.Fatalf("expected approx. %f, got %f", expected, got)
			return false
		}
		return true
	}

	if err := quick.Check(it, nil); err != nil {
		t.Fatal(err)
	}
}

func TestGaugeVecConcurrency(t *testing.T) {
	it := func(n uint32) bool {
		mutations := int(n % 10000)
		concLevel := int(n%15 + 1)
		vecLength := int(n%5 + 1)

		var start, end sync.WaitGroup
		start.Add(1)
		end.Add(concLevel)

		sStreams := make([]chan float64, vecLength)
		results := make([]chan float64, vecLength)
		done := make(chan struct{})

		for i := 0; i < vecLength; i++ {
			sStreams[i] = make(chan float64, mutations*concLevel)
			results[i] = make(chan float64)
			go listenGaugeStream(sStreams[i], results[i], done)
		}

		go func() {
			end.Wait()
			close(done)
		}()

		gge := NewGaugeVec(
			GaugeOpts{
				Name: "test_gauge",
				Help: "no help can be found here",
			},
			[]string{"label"},
		)
		for i := 0; i < concLevel; i++ {
			vals := make([]float64, mutations)
			pick := make([]int, mutations)
			for j := 0; j < mutations; j++ {
				vals[j] = rand.Float64() - 0.5
				pick[j] = rand.Intn(vecLength)
			}

			go func(vals []float64) {
				start.Wait()
				for i, v := range vals {
					sStreams[pick[i]] <- v
					gge.WithLabelValues(string('A' + pick[i])).Add(v)
				}
				end.Done()
			}(vals)
		}
		start.Done()

		for i := range sStreams {
			if expected, got := <-results[i], math.Float64frombits(gge.WithLabelValues(string('A'+i)).(*value).valBits); math.Abs(expected-got) > 0.000001 {
				t.Fatalf("expected approx. %f, got %f", expected, got)
				return false
			}
		}
		return true
	}

	if err := quick.Check(it, nil); err != nil {
		t.Fatal(err)
	}
}

func TestGaugeFunc(t *testing.T) {
	gf := NewGaugeFunc(
		GaugeOpts{
			Name:        "test_name",
			Help:        "test help",
			ConstLabels: Labels{"a": "1", "b": "2"},
		},
		func() float64 { return 3.1415 },
	)

	if expected, got := `Desc{fqName: "test_name", help: "test help", constLabels: {a="1",b="2"}, variableLabels: []}`, gf.Desc().String(); expected != got {
		t.Errorf("expected %q, got %q", expected, got)
	}

	m := &dto.Metric{}
	gf.Write(m)

	if expected, got := `label:<name:"a" value:"1" > label:<name:"b" value:"2" > gauge:<value:3.1415 > `, m.String(); expected != got {
		t.Errorf("expected %q, got %q", expected, got)
	}
}
