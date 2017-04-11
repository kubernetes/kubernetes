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
	"hash/fnv"
	"testing"
)

func TestDelete(t *testing.T) {
	desc := NewDesc("test", "helpless", []string{"l1", "l2"}, nil)
	vec := MetricVec{
		children: map[uint64]Metric{},
		desc:     desc,
		hash:     fnv.New64a(),
		newMetric: func(lvs ...string) Metric {
			return newValue(desc, UntypedValue, 0, lvs...)
		},
	}

	if got, want := vec.Delete(Labels{"l1": "v1", "l2": "v2"}), false; got != want {
		t.Errorf("got %v, want %v", got, want)
	}

	vec.With(Labels{"l1": "v1", "l2": "v2"}).(Untyped).Set(42)
	if got, want := vec.Delete(Labels{"l1": "v1", "l2": "v2"}), true; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
	if got, want := vec.Delete(Labels{"l1": "v1", "l2": "v2"}), false; got != want {
		t.Errorf("got %v, want %v", got, want)
	}

	vec.With(Labels{"l1": "v1", "l2": "v2"}).(Untyped).Set(42)
	if got, want := vec.Delete(Labels{"l2": "v2", "l1": "v1"}), true; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
	if got, want := vec.Delete(Labels{"l2": "v2", "l1": "v1"}), false; got != want {
		t.Errorf("got %v, want %v", got, want)
	}

	vec.With(Labels{"l1": "v1", "l2": "v2"}).(Untyped).Set(42)
	if got, want := vec.Delete(Labels{"l2": "v1", "l1": "v2"}), false; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
	if got, want := vec.Delete(Labels{"l1": "v1"}), false; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestDeleteLabelValues(t *testing.T) {
	desc := NewDesc("test", "helpless", []string{"l1", "l2"}, nil)
	vec := MetricVec{
		children: map[uint64]Metric{},
		desc:     desc,
		hash:     fnv.New64a(),
		newMetric: func(lvs ...string) Metric {
			return newValue(desc, UntypedValue, 0, lvs...)
		},
	}

	if got, want := vec.DeleteLabelValues("v1", "v2"), false; got != want {
		t.Errorf("got %v, want %v", got, want)
	}

	vec.With(Labels{"l1": "v1", "l2": "v2"}).(Untyped).Set(42)
	if got, want := vec.DeleteLabelValues("v1", "v2"), true; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
	if got, want := vec.DeleteLabelValues("v1", "v2"), false; got != want {
		t.Errorf("got %v, want %v", got, want)
	}

	vec.With(Labels{"l1": "v1", "l2": "v2"}).(Untyped).Set(42)
	if got, want := vec.DeleteLabelValues("v2", "v1"), false; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
	if got, want := vec.DeleteLabelValues("v1"), false; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
}
