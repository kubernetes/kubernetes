// Copyright 2020 The Prometheus Authors
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

package testutil

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
)

func TestCollectAndLintGood(t *testing.T) {
	cnt := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "some_total",
			Help: "A value that represents a counter.",
			ConstLabels: prometheus.Labels{
				"label1": "value1",
			},
		},
		[]string{"foo"},
	)
	cnt.WithLabelValues("bar")
	cnt.WithLabelValues("baz")

	problems, err := CollectAndLint(cnt)
	if err != nil {
		t.Error("Unexpected error:", err)
	}
	if len(problems) > 0 {
		t.Error("Unexpected lint problems:", problems)
	}
}

func TestCollectAndLintBad(t *testing.T) {
	cnt := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "someThing_ms",
			Help: "A value that represents a counter.",
			ConstLabels: prometheus.Labels{
				"label1": "value1",
			},
		},
		[]string{"fooBar"},
	)
	cnt.WithLabelValues("bar")
	cnt.WithLabelValues("baz")

	problems, err := CollectAndLint(cnt)
	if err != nil {
		t.Error("Unexpected error:", err)
	}
	if len(problems) < 5 {
		// The exact nature of the lint problems found is tested within
		// the promlint package itself. Here we only want to make sure
		// that the collector successfully hits the linter and that at
		// least the five problems that the linter could recognize at
		// the time of writing this test are flagged.
		t.Error("Not enough lint problems found.")
	}
}
