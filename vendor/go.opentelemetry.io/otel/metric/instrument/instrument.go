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

// Asynchronous instruments are instruments that are updated within a Callback.
// If an instrument is observed outside of it's callback it should be an error.
//
// This interface is used as a grouping mechanism.
type Asynchronous interface {
	asynchronous()
}

// Synchronous instruments are updated in line with application code.
//
// This interface is used as a grouping mechanism.
type Synchronous interface {
	synchronous()
}

// Option applies options to all instruments.
type Option interface {
	Float64ObserverOption
	Int64ObserverOption
	Float64Option
	Int64Option
}

type descOpt string

func (o descOpt) applyFloat64(c Float64Config) Float64Config {
	c.description = string(o)
	return c
}

func (o descOpt) applyInt64(c Int64Config) Int64Config {
	c.description = string(o)
	return c
}

func (o descOpt) applyFloat64Observer(c Float64ObserverConfig) Float64ObserverConfig {
	c.description = string(o)
	return c
}

func (o descOpt) applyInt64Observer(c Int64ObserverConfig) Int64ObserverConfig {
	c.description = string(o)
	return c
}

// WithDescription sets the instrument description.
func WithDescription(desc string) Option { return descOpt(desc) }

type unitOpt string

func (o unitOpt) applyFloat64(c Float64Config) Float64Config {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyInt64(c Int64Config) Int64Config {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyFloat64Observer(c Float64ObserverConfig) Float64ObserverConfig {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyInt64Observer(c Int64ObserverConfig) Int64ObserverConfig {
	c.unit = string(o)
	return c
}

// WithUnit sets the instrument unit.
func WithUnit(u string) Option { return unitOpt(u) }
