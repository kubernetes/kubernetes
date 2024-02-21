// Code created by gotmpl. DO NOT MODIFY.
// source: internal/shared/otlp/otlptrace/otlpconfig/options.go.tmpl

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

package otlpconfig // import "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc/internal/otlpconfig"

import (
	"crypto/tls"
	"fmt"
	"path"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/encoding/gzip"

	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc/internal/retry"
)

const (
	// DefaultTracesPath is a default URL path for endpoint that
	// receives spans.
	DefaultTracesPath string = "/v1/traces"
	// DefaultTimeout is a default max waiting time for the backend to process
	// each span batch.
	DefaultTimeout time.Duration = 10 * time.Second
)

type (
	SignalConfig struct {
		Endpoint    string
		Insecure    bool
		TLSCfg      *tls.Config
		Headers     map[string]string
		Compression Compression
		Timeout     time.Duration
		URLPath     string

		// gRPC configurations
		GRPCCredentials credentials.TransportCredentials
	}

	Config struct {
		// Signal specific configurations
		Traces SignalConfig

		RetryConfig retry.Config

		// gRPC configurations
		ReconnectionPeriod time.Duration
		ServiceConfig      string
		DialOptions        []grpc.DialOption
		GRPCConn           *grpc.ClientConn
	}
)

// NewHTTPConfig returns a new Config with all settings applied from opts and
// any unset setting using the default HTTP config values.
func NewHTTPConfig(opts ...HTTPOption) Config {
	cfg := Config{
		Traces: SignalConfig{
			Endpoint:    fmt.Sprintf("%s:%d", DefaultCollectorHost, DefaultCollectorHTTPPort),
			URLPath:     DefaultTracesPath,
			Compression: NoCompression,
			Timeout:     DefaultTimeout,
		},
		RetryConfig: retry.DefaultConfig,
	}
	cfg = ApplyHTTPEnvConfigs(cfg)
	for _, opt := range opts {
		cfg = opt.ApplyHTTPOption(cfg)
	}
	cfg.Traces.URLPath = cleanPath(cfg.Traces.URLPath, DefaultTracesPath)
	return cfg
}

// cleanPath returns a path with all spaces trimmed and all redundancies
// removed. If urlPath is empty or cleaning it results in an empty string,
// defaultPath is returned instead.
func cleanPath(urlPath string, defaultPath string) string {
	tmp := path.Clean(strings.TrimSpace(urlPath))
	if tmp == "." {
		return defaultPath
	}
	if !path.IsAbs(tmp) {
		tmp = fmt.Sprintf("/%s", tmp)
	}
	return tmp
}

// NewGRPCConfig returns a new Config with all settings applied from opts and
// any unset setting using the default gRPC config values.
func NewGRPCConfig(opts ...GRPCOption) Config {
	userAgent := "OTel OTLP Exporter Go/" + otlptrace.Version()
	cfg := Config{
		Traces: SignalConfig{
			Endpoint:    fmt.Sprintf("%s:%d", DefaultCollectorHost, DefaultCollectorGRPCPort),
			URLPath:     DefaultTracesPath,
			Compression: NoCompression,
			Timeout:     DefaultTimeout,
		},
		RetryConfig: retry.DefaultConfig,
		DialOptions: []grpc.DialOption{grpc.WithUserAgent(userAgent)},
	}
	cfg = ApplyGRPCEnvConfigs(cfg)
	for _, opt := range opts {
		cfg = opt.ApplyGRPCOption(cfg)
	}

	if cfg.ServiceConfig != "" {
		cfg.DialOptions = append(cfg.DialOptions, grpc.WithDefaultServiceConfig(cfg.ServiceConfig))
	}
	// Priroritize GRPCCredentials over Insecure (passing both is an error).
	if cfg.Traces.GRPCCredentials != nil {
		cfg.DialOptions = append(cfg.DialOptions, grpc.WithTransportCredentials(cfg.Traces.GRPCCredentials))
	} else if cfg.Traces.Insecure {
		cfg.DialOptions = append(cfg.DialOptions, grpc.WithTransportCredentials(insecure.NewCredentials()))
	} else {
		// Default to using the host's root CA.
		creds := credentials.NewTLS(nil)
		cfg.Traces.GRPCCredentials = creds
		cfg.DialOptions = append(cfg.DialOptions, grpc.WithTransportCredentials(creds))
	}
	if cfg.Traces.Compression == GzipCompression {
		cfg.DialOptions = append(cfg.DialOptions, grpc.WithDefaultCallOptions(grpc.UseCompressor(gzip.Name)))
	}
	if cfg.ReconnectionPeriod != 0 {
		p := grpc.ConnectParams{
			Backoff:           backoff.DefaultConfig,
			MinConnectTimeout: cfg.ReconnectionPeriod,
		}
		cfg.DialOptions = append(cfg.DialOptions, grpc.WithConnectParams(p))
	}

	return cfg
}

type (
	// GenericOption applies an option to the HTTP or gRPC driver.
	GenericOption interface {
		ApplyHTTPOption(Config) Config
		ApplyGRPCOption(Config) Config

		// A private method to prevent users implementing the
		// interface and so future additions to it will not
		// violate compatibility.
		private()
	}

	// HTTPOption applies an option to the HTTP driver.
	HTTPOption interface {
		ApplyHTTPOption(Config) Config

		// A private method to prevent users implementing the
		// interface and so future additions to it will not
		// violate compatibility.
		private()
	}

	// GRPCOption applies an option to the gRPC driver.
	GRPCOption interface {
		ApplyGRPCOption(Config) Config

		// A private method to prevent users implementing the
		// interface and so future additions to it will not
		// violate compatibility.
		private()
	}
)

// genericOption is an option that applies the same logic
// for both gRPC and HTTP.
type genericOption struct {
	fn func(Config) Config
}

func (g *genericOption) ApplyGRPCOption(cfg Config) Config {
	return g.fn(cfg)
}

func (g *genericOption) ApplyHTTPOption(cfg Config) Config {
	return g.fn(cfg)
}

func (genericOption) private() {}

func newGenericOption(fn func(cfg Config) Config) GenericOption {
	return &genericOption{fn: fn}
}

// splitOption is an option that applies different logics
// for gRPC and HTTP.
type splitOption struct {
	httpFn func(Config) Config
	grpcFn func(Config) Config
}

func (g *splitOption) ApplyGRPCOption(cfg Config) Config {
	return g.grpcFn(cfg)
}

func (g *splitOption) ApplyHTTPOption(cfg Config) Config {
	return g.httpFn(cfg)
}

func (splitOption) private() {}

func newSplitOption(httpFn func(cfg Config) Config, grpcFn func(cfg Config) Config) GenericOption {
	return &splitOption{httpFn: httpFn, grpcFn: grpcFn}
}

// httpOption is an option that is only applied to the HTTP driver.
type httpOption struct {
	fn func(Config) Config
}

func (h *httpOption) ApplyHTTPOption(cfg Config) Config {
	return h.fn(cfg)
}

func (httpOption) private() {}

func NewHTTPOption(fn func(cfg Config) Config) HTTPOption {
	return &httpOption{fn: fn}
}

// grpcOption is an option that is only applied to the gRPC driver.
type grpcOption struct {
	fn func(Config) Config
}

func (h *grpcOption) ApplyGRPCOption(cfg Config) Config {
	return h.fn(cfg)
}

func (grpcOption) private() {}

func NewGRPCOption(fn func(cfg Config) Config) GRPCOption {
	return &grpcOption{fn: fn}
}

// Generic Options

func WithEndpoint(endpoint string) GenericOption {
	return newGenericOption(func(cfg Config) Config {
		cfg.Traces.Endpoint = endpoint
		return cfg
	})
}

func WithCompression(compression Compression) GenericOption {
	return newGenericOption(func(cfg Config) Config {
		cfg.Traces.Compression = compression
		return cfg
	})
}

func WithURLPath(urlPath string) GenericOption {
	return newGenericOption(func(cfg Config) Config {
		cfg.Traces.URLPath = urlPath
		return cfg
	})
}

func WithRetry(rc retry.Config) GenericOption {
	return newGenericOption(func(cfg Config) Config {
		cfg.RetryConfig = rc
		return cfg
	})
}

func WithTLSClientConfig(tlsCfg *tls.Config) GenericOption {
	return newSplitOption(func(cfg Config) Config {
		cfg.Traces.TLSCfg = tlsCfg.Clone()
		return cfg
	}, func(cfg Config) Config {
		cfg.Traces.GRPCCredentials = credentials.NewTLS(tlsCfg)
		return cfg
	})
}

func WithInsecure() GenericOption {
	return newGenericOption(func(cfg Config) Config {
		cfg.Traces.Insecure = true
		return cfg
	})
}

func WithSecure() GenericOption {
	return newGenericOption(func(cfg Config) Config {
		cfg.Traces.Insecure = false
		return cfg
	})
}

func WithHeaders(headers map[string]string) GenericOption {
	return newGenericOption(func(cfg Config) Config {
		cfg.Traces.Headers = headers
		return cfg
	})
}

func WithTimeout(duration time.Duration) GenericOption {
	return newGenericOption(func(cfg Config) Config {
		cfg.Traces.Timeout = duration
		return cfg
	})
}
