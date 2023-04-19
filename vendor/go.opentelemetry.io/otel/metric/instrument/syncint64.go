// Copyright The OpenTelemetry Authors
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

package instrument // import "go.opentelemetry.io/otel/metric/instrument"

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
)

// Int64Counter is an instrument that records increasing int64 values.
//
// Warning: methods may be added to this interface in minor releases.
type Int64Counter interface {
	// Add records a change to the counter.
	Add(ctx context.Context, incr int64, attrs ...attribute.KeyValue)

	Synchronous
}

// Int64UpDownCounter is an instrument that records increasing or decreasing
// int64 values.
//
// Warning: methods may be added to this interface in minor releases.
type Int64UpDownCounter interface {
	// Add records a change to the counter.
	Add(ctx context.Context, incr int64, attrs ...attribute.KeyValue)

	Synchronous
}

// Int64Histogram is an instrument that records a distribution of int64
// values.
//
// Warning: methods may be added to this interface in minor releases.
type Int64Histogram interface {
	// Record adds an additional value to the distribution.
	Record(ctx context.Context, incr int64, attrs ...attribute.KeyValue)

	Synchronous
}

// Int64Config contains options for Synchronous instruments that record int64
// values.
type Int64Config struct {
	description string
	unit        string
}

// NewInt64Config returns a new Int64Config with all opts
// applied.
func NewInt64Config(opts ...Int64Option) Int64Config {
	var config Int64Config
	for _, o := range opts {
		config = o.applyInt64(config)
	}
	return config
}

// Description returns the Config description.
func (c Int64Config) Description() string {
	return c.description
}

// Unit returns the Config unit.
func (c Int64Config) Unit() string {
	return c.unit
}

// Int64Option applies options to synchronous int64 instruments.
type Int64Option interface {
	applyInt64(Int64Config) Int64Config
}
