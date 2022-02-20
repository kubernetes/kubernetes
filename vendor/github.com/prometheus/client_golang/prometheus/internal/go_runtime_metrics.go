// Copyright 2021 The Prometheus Authors
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

//go:build go1.17
// +build go1.17

package internal

import (
	"path"
	"runtime/metrics"
	"strings"

	"github.com/prometheus/common/model"
)

// RuntimeMetricsToProm produces a Prometheus metric name from a runtime/metrics
// metric description and validates whether the metric is suitable for integration
// with Prometheus.
//
// Returns false if a name could not be produced, or if Prometheus does not understand
// the runtime/metrics Kind.
//
// Note that the main reason a name couldn't be produced is if the runtime/metrics
// package exports a name with characters outside the valid Prometheus metric name
// character set. This is theoretically possible, but should never happen in practice.
// Still, don't rely on it.
func RuntimeMetricsToProm(d *metrics.Description) (string, string, string, bool) {
	namespace := "go"

	comp := strings.SplitN(d.Name, ":", 2)
	key := comp[0]
	unit := comp[1]

	// The last path element in the key is the name,
	// the rest is the subsystem.
	subsystem := path.Dir(key[1:] /* remove leading / */)
	name := path.Base(key)

	// subsystem is translated by replacing all / and - with _.
	subsystem = strings.ReplaceAll(subsystem, "/", "_")
	subsystem = strings.ReplaceAll(subsystem, "-", "_")

	// unit is translated assuming that the unit contains no
	// non-ASCII characters.
	unit = strings.ReplaceAll(unit, "-", "_")
	unit = strings.ReplaceAll(unit, "*", "_")
	unit = strings.ReplaceAll(unit, "/", "_per_")

	// name has - replaced with _ and is concatenated with the unit and
	// other data.
	name = strings.ReplaceAll(name, "-", "_")
	name = name + "_" + unit
	if d.Cumulative {
		name = name + "_total"
	}

	valid := model.IsValidMetricName(model.LabelValue(namespace + "_" + subsystem + "_" + name))
	switch d.Kind {
	case metrics.KindUint64:
	case metrics.KindFloat64:
	case metrics.KindFloat64Histogram:
	default:
		valid = false
	}
	return namespace, subsystem, name, valid
}
