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

package otlptracegrpc // import "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"

import (
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/internal/retry"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/internal/otlpconfig"
)

// Option applies an option to the gRPC driver.
type Option interface {
	applyGRPCOption(otlpconfig.Config) otlpconfig.Config
}

func asGRPCOptions(opts []Option) []otlpconfig.GRPCOption {
	converted := make([]otlpconfig.GRPCOption, len(opts))
	for i, o := range opts {
		converted[i] = otlpconfig.NewGRPCOption(o.applyGRPCOption)
	}
	return converted
}

// RetryConfig defines configuration for retrying export of span batches that
// failed to be received by the target endpoint.
//
// This configuration does not define any network retry strategy. That is
// entirely handled by the gRPC ClientConn.
type RetryConfig retry.Config

type wrappedOption struct {
	otlpconfig.GRPCOption
}

func (w wrappedOption) applyGRPCOption(cfg otlpconfig.Config) otlpconfig.Config {
	return w.ApplyGRPCOption(cfg)
}

// WithInsecure disables client transport security for the exporter's gRPC
// connection just like grpc.WithInsecure()
// (https://pkg.go.dev/google.golang.org/grpc#WithInsecure) does. Note, by
// default, client security is required unless WithInsecure is used.
//
// This option has no effect if WithGRPCConn is used.
func WithInsecure() Option {
	return wrappedOption{otlpconfig.WithInsecure()}
}

// WithEndpoint sets the target endpoint the exporter will connect to. If
// unset, localhost:4317 will be used as a default.
//
// This option has no effect if WithGRPCConn is used.
func WithEndpoint(endpoint string) Option {
	return wrappedOption{otlpconfig.WithEndpoint(endpoint)}
}

// WithReconnectionPeriod set the minimum amount of time between connection
// attempts to the target endpoint.
//
// This option has no effect if WithGRPCConn is used.
func WithReconnectionPeriod(rp time.Duration) Option {
	return wrappedOption{otlpconfig.NewGRPCOption(func(cfg otlpconfig.Config) otlpconfig.Config {
		cfg.ReconnectionPeriod = rp
		return cfg
	})}
}

func compressorToCompression(compressor string) otlpconfig.Compression {
	if compressor == "gzip" {
		return otlpconfig.GzipCompression
	}

	otel.Handle(fmt.Errorf("invalid compression type: '%s', using no compression as default", compressor))
	return otlpconfig.NoCompression
}

// WithCompressor sets the compressor for the gRPC client to use when sending
// requests. It is the responsibility of the caller to ensure that the
// compressor set has been registered with google.golang.org/grpc/encoding.
// This can be done by encoding.RegisterCompressor. Some compressors
// auto-register on import, such as gzip, which can be registered by calling
// `import _ "google.golang.org/grpc/encoding/gzip"`.
//
// This option has no effect if WithGRPCConn is used.
func WithCompressor(compressor string) Option {
	return wrappedOption{otlpconfig.WithCompression(compressorToCompression(compressor))}
}

// WithHeaders will send the provided headers with each gRPC requests.
func WithHeaders(headers map[string]string) Option {
	return wrappedOption{otlpconfig.WithHeaders(headers)}
}

// WithTLSCredentials allows the connection to use TLS credentials when
// talking to the server. It takes in grpc.TransportCredentials instead of say
// a Certificate file or a tls.Certificate, because the retrieving of these
// credentials can be done in many ways e.g. plain file, in code tls.Config or
// by certificate rotation, so it is up to the caller to decide what to use.
//
// This option has no effect if WithGRPCConn is used.
func WithTLSCredentials(creds credentials.TransportCredentials) Option {
	return wrappedOption{otlpconfig.NewGRPCOption(func(cfg otlpconfig.Config) otlpconfig.Config {
		cfg.Traces.GRPCCredentials = creds
		return cfg
	})}
}

// WithServiceConfig defines the default gRPC service config used.
//
// This option has no effect if WithGRPCConn is used.
func WithServiceConfig(serviceConfig string) Option {
	return wrappedOption{otlpconfig.NewGRPCOption(func(cfg otlpconfig.Config) otlpconfig.Config {
		cfg.ServiceConfig = serviceConfig
		return cfg
	})}
}

// WithDialOption sets explicit grpc.DialOptions to use when making a
// connection. The options here are appended to the internal grpc.DialOptions
// used so they will take precedence over any other internal grpc.DialOptions
// they might conflict with.
//
// This option has no effect if WithGRPCConn is used.
func WithDialOption(opts ...grpc.DialOption) Option {
	return wrappedOption{otlpconfig.NewGRPCOption(func(cfg otlpconfig.Config) otlpconfig.Config {
		cfg.DialOptions = opts
		return cfg
	})}
}

// WithGRPCConn sets conn as the gRPC ClientConn used for all communication.
//
// This option takes precedence over any other option that relates to
// establishing or persisting a gRPC connection to a target endpoint. Any
// other option of those types passed will be ignored.
//
// It is the callers responsibility to close the passed conn. The client
// Shutdown method will not close this connection.
func WithGRPCConn(conn *grpc.ClientConn) Option {
	return wrappedOption{otlpconfig.NewGRPCOption(func(cfg otlpconfig.Config) otlpconfig.Config {
		cfg.GRPCConn = conn
		return cfg
	})}
}

// WithTimeout sets the max amount of time a client will attempt to export a
// batch of spans. This takes precedence over any retry settings defined with
// WithRetry, once this time limit has been reached the export is abandoned
// and the batch of spans is dropped.
//
// If unset, the default timeout will be set to 10 seconds.
func WithTimeout(duration time.Duration) Option {
	return wrappedOption{otlpconfig.WithTimeout(duration)}
}

// WithRetry sets the retry policy for transient retryable errors that may be
// returned by the target endpoint when exporting a batch of spans.
//
// If the target endpoint responds with not only a retryable error, but
// explicitly returns a backoff time in the response. That time will take
// precedence over these settings.
//
// These settings do not define any network retry strategy. That is entirely
// handled by the gRPC ClientConn.
//
// If unset, the default retry policy will be used. It will retry the export
// 5 seconds after receiving a retryable error and increase exponentially
// after each error for no more than a total time of 1 minute.
func WithRetry(settings RetryConfig) Option {
	return wrappedOption{otlpconfig.WithRetry(retry.Config(settings))}
}
