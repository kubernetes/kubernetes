// Copyright 2017, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package stats

import (
	"sync"
	"sync/atomic"
)

// Measure represents a single numeric value to be tracked and recorded.
// For example, latency, request bytes, and response bytes could be measures
// to collect from a server.
//
// Measures by themselves have no outside effects. In order to be exported,
// the measure needs to be used in a View. If no Views are defined over a
// measure, there is very little cost in recording it.
type Measure interface {
	// Name returns the name of this measure.
	//
	// Measure names are globally unique (among all libraries linked into your program).
	// We recommend prefixing the measure name with a domain name relevant to your
	// project or application.
	//
	// Measure names are never sent over the wire or exported to backends.
	// They are only used to create Views.
	Name() string

	// Description returns the human-readable description of this measure.
	Description() string

	// Unit returns the units for the values this measure takes on.
	//
	// Units are encoded according to the case-sensitive abbreviations from the
	// Unified Code for Units of Measure: http://unitsofmeasure.org/ucum.html
	Unit() string
}

// measureDescriptor is the untyped descriptor associated with each measure.
// Int64Measure and Float64Measure wrap measureDescriptor to provide typed
// recording APIs.
// Two Measures with the same name will have the same measureDescriptor.
type measureDescriptor struct {
	subs int32 // access atomically

	name        string
	description string
	unit        string
}

func (m *measureDescriptor) subscribe() {
	atomic.StoreInt32(&m.subs, 1)
}

func (m *measureDescriptor) subscribed() bool {
	return atomic.LoadInt32(&m.subs) == 1
}

// Name returns the name of the measure.
func (m *measureDescriptor) Name() string {
	return m.name
}

// Description returns the description of the measure.
func (m *measureDescriptor) Description() string {
	return m.description
}

// Unit returns the unit of the measure.
func (m *measureDescriptor) Unit() string {
	return m.unit
}

var (
	mu       sync.RWMutex
	measures = make(map[string]*measureDescriptor)
)

func registerMeasureHandle(name, desc, unit string) *measureDescriptor {
	mu.Lock()
	defer mu.Unlock()

	if stored, ok := measures[name]; ok {
		return stored
	}
	m := &measureDescriptor{
		name:        name,
		description: desc,
		unit:        unit,
	}
	measures[name] = m
	return m
}

// Measurement is the numeric value measured when recording stats. Each measure
// provides methods to create measurements of their kind. For example, Int64Measure
// provides M to convert an int64 into a measurement.
type Measurement struct {
	v float64
	m *measureDescriptor
}

// Value returns the value of the Measurement as a float64.
func (m Measurement) Value() float64 {
	return m.v
}

// Measure returns the Measure from which this Measurement was created.
func (m Measurement) Measure() Measure {
	return m.m
}
