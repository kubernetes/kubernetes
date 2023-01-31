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

import "go.opentelemetry.io/otel/metric/unit"

// Config contains options for metric instrument descriptors.
type Config struct {
	description string
	unit        unit.Unit
}

// Description describes the instrument in human-readable terms.
func (cfg Config) Description() string {
	return cfg.description
}

// Unit describes the measurement unit for an instrument.
func (cfg Config) Unit() unit.Unit {
	return cfg.unit
}

// Option is an interface for applying metric instrument options.
type Option interface {
	applyInstrument(Config) Config
}

// NewConfig creates a new Config and applies all the given options.
func NewConfig(opts ...Option) Config {
	var config Config
	for _, o := range opts {
		config = o.applyInstrument(config)
	}
	return config
}

type optionFunc func(Config) Config

func (fn optionFunc) applyInstrument(cfg Config) Config {
	return fn(cfg)
}

// WithDescription applies provided description.
func WithDescription(desc string) Option {
	return optionFunc(func(cfg Config) Config {
		cfg.description = desc
		return cfg
	})
}

// WithUnit applies provided unit.
func WithUnit(u unit.Unit) Option {
	return optionFunc(func(cfg Config) Config {
		cfg.unit = u
		return cfg
	})
}
