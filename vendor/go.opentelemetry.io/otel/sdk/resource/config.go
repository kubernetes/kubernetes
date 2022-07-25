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

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
)

// config contains configuration for Resource creation.
type config struct {
	// detectors that will be evaluated.
	detectors []Detector

	// telemetrySDK is used to specify non-default
	// `telemetry.sdk.*` attributes.
	telemetrySDK Detector

	// HostResource is used to specify non-default `host.*`
	// attributes.
	host Detector

	// FromEnv is used to specify non-default OTEL_RESOURCE_ATTRIBUTES
	// attributes.
	fromEnv Detector
}

// Option is the interface that applies a configuration option.
type Option interface {
	// Apply sets the Option value of a config.
	Apply(*config)

	// A private method to prevent users implementing the
	// interface and so future additions to it will not
	// violate compatibility.
	private()
}

type option struct{}

func (option) private() {}

// WithAttributes adds attributes to the configured Resource.
func WithAttributes(attributes ...attribute.KeyValue) Option {
	return WithDetectors(detectAttributes{attributes})
}

type detectAttributes struct {
	attributes []attribute.KeyValue
}

func (d detectAttributes) Detect(context.Context) (*Resource, error) {
	return NewWithAttributes(d.attributes...), nil
}

// WithDetectors adds detectors to be evaluated for the configured resource.
func WithDetectors(detectors ...Detector) Option {
	return detectorsOption{detectors: detectors}
}

type detectorsOption struct {
	option
	detectors []Detector
}

// Apply implements Option.
func (o detectorsOption) Apply(cfg *config) {
	cfg.detectors = append(cfg.detectors, o.detectors...)
}

// WithTelemetrySDK overrides the builtin `telemetry.sdk.*`
// attributes.  Use nil to disable these attributes entirely.
func WithTelemetrySDK(d Detector) Option {
	return telemetrySDKOption{Detector: d}
}

type telemetrySDKOption struct {
	option
	Detector
}

// Apply implements Option.
func (o telemetrySDKOption) Apply(cfg *config) {
	cfg.telemetrySDK = o.Detector
}

// WithHost overrides the builtin `host.*` attributes.  Use nil to
// disable these attributes entirely.
func WithHost(d Detector) Option {
	return hostOption{Detector: d}
}

type hostOption struct {
	option
	Detector
}

// Apply implements Option.
func (o hostOption) Apply(cfg *config) {
	cfg.host = o.Detector
}

// WithFromEnv overrides the builtin detector for
// OTEL_RESOURCE_ATTRIBUTES.  Use nil to disable environment checking.
func WithFromEnv(d Detector) Option {
	return fromEnvOption{Detector: d}
}

type fromEnvOption struct {
	option
	Detector
}

// Apply implements Option.
func (o fromEnvOption) Apply(cfg *config) {
	cfg.fromEnv = o.Detector
}

// WithoutBuiltin disables all the builtin detectors, including the
// telemetry.sdk.*, host.*, and the environment detector.
func WithoutBuiltin() Option {
	return noBuiltinOption{}
}

type noBuiltinOption struct {
	option
}

// Apply implements Option.
func (o noBuiltinOption) Apply(cfg *config) {
	cfg.host = nil
	cfg.telemetrySDK = nil
	cfg.fromEnv = nil
}

// New returns a Resource combined from the provided attributes,
// user-provided detectors and builtin detectors.
func New(ctx context.Context, opts ...Option) (*Resource, error) {
	cfg := config{
		telemetrySDK: TelemetrySDK{},
		host:         Host{},
		fromEnv:      FromEnv{},
	}
	for _, opt := range opts {
		opt.Apply(&cfg)
	}
	detectors := append(
		[]Detector{cfg.telemetrySDK, cfg.host, cfg.fromEnv},
		cfg.detectors...,
	)
	return Detect(ctx, detectors...)
}
