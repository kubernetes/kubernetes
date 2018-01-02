// Copyright 2015 Google Inc. All Rights Reserved.
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

package core

import (
	"time"
)

type MetricType int8

const (
	MetricCumulative MetricType = iota
	MetricGauge
	MetricDelta
)

func (self *MetricType) String() string {
	switch *self {
	case MetricCumulative:
		return "cumulative"
	case MetricGauge:
		return "gauge"
	case MetricDelta:
		return "delta"
	}
	return ""
}

type ValueType int8

const (
	ValueInt64 ValueType = iota
	ValueFloat
)

func (self *ValueType) String() string {
	switch *self {
	case ValueInt64:
		return "int64"
	case ValueFloat:
		return "double"
	}
	return ""
}

type UnitsType int8

const (
	// A counter metric.
	UnitsCount UnitsType = iota
	// A metric in bytes.
	UnitsBytes
	// A metric in milliseconds.
	UnitsMilliseconds
	// A metric in nanoseconds.
	UnitsNanoseconds
	// A metric in millicores.
	UnitsMillicores
)

func (self *UnitsType) String() string {
	switch *self {
	case UnitsBytes:
		return "bytes"
	case UnitsMilliseconds:
		return "ms"
	case UnitsNanoseconds:
		return "ns"
	case UnitsMillicores:
		return "millicores"
	}
	return ""
}

type MetricValue struct {
	IntValue   int64
	FloatValue float32
	MetricType MetricType
	ValueType  ValueType
}

func (this *MetricValue) GetValue() interface{} {
	if ValueInt64 == this.ValueType {
		return this.IntValue
	} else if ValueFloat == this.ValueType {
		return this.FloatValue
	} else {
		return nil
	}
}

type LabeledMetric struct {
	Name   string
	Labels map[string]string
	MetricValue
}

func (this *LabeledMetric) GetValue() interface{} {
	if ValueInt64 == this.ValueType {
		return this.IntValue
	} else if ValueFloat == this.ValueType {
		return this.FloatValue
	} else {
		return nil
	}
}

type MetricSet struct {
	CreateTime     time.Time
	ScrapeTime     time.Time
	MetricValues   map[string]MetricValue
	Labels         map[string]string
	LabeledMetrics []LabeledMetric
}

type DataBatch struct {
	Timestamp time.Time
	// Should use key functions from ms_keys.go
	MetricSets map[string]*MetricSet
}

// A place from where the metrics should be scraped.
type MetricsSource interface {
	Name() string
	ScrapeMetrics(start, end time.Time) *DataBatch
}

// Provider of list of sources to be scaped.
type MetricsSourceProvider interface {
	GetMetricsSources() []MetricsSource
}

type DataSink interface {
	Name() string

	// Exports data to the external storge. The funciton should be synchronous/blocking and finish only
	// after the given DataBatch was written. This will allow sink manager to push data only to these
	// sinks that finished writing the previous data.
	ExportData(*DataBatch)
	Stop()
}

type DataProcessor interface {
	Name() string
	Process(*DataBatch) (*DataBatch, error)
}
