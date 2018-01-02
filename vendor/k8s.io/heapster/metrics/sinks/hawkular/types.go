// Copyright 2016 Google Inc. All Rights Reserved.
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

package hawkular

import (
	"net/url"
	"sync"

	"github.com/hawkular/hawkular-client-go/metrics"
	"k8s.io/heapster/metrics/core"
)

type Filter func(ms *core.MetricSet, metricName string) bool
type FilterType int

const (
	// Filter by label's value
	Label FilterType = iota
	// Filter by metric name
	Name
	// Unknown filter type
	Unknown
)

func (f FilterType) From(s string) FilterType {
	switch s {
	case "label":
		return Label
	case "name":
		return Name
	default:
		return Unknown
	}
}

type hawkularSink struct {
	client  *metrics.Client
	models  map[string]*metrics.MetricDefinition // Model definitions
	regLock sync.Mutex
	reg     map[string]*metrics.MetricDefinition // Real definitions

	uri *url.URL

	labelTenant string
	labelNodeId string
	modifiers   []metrics.Modifier
	filters     []Filter

	batchSize int
}

func heapsterTypeToHawkularType(t core.MetricType) metrics.MetricType {
	switch t {
	case core.MetricCumulative:
		return metrics.Counter
	case core.MetricGauge:
		return metrics.Gauge
	default:
		return metrics.Gauge
	}
}
