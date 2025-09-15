// Code created by gotmpl. DO NOT MODIFY.
// source: internal/shared/otlp/otlptrace/otlpconfig/envconfig.go.tmpl

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otlpconfig // import "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc/internal/otlpconfig"

import (
	"crypto/tls"
	"crypto/x509"
	"net/url"
	"os"
	"path"
	"strings"
	"time"

	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc/internal/envconfig"
)

// DefaultEnvOptionsReader is the default environments reader.
var DefaultEnvOptionsReader = envconfig.EnvOptionsReader{
	GetEnv:    os.Getenv,
	ReadFile:  os.ReadFile,
	Namespace: "OTEL_EXPORTER_OTLP",
}

// ApplyGRPCEnvConfigs applies the env configurations for gRPC.
func ApplyGRPCEnvConfigs(cfg Config) Config {
	opts := getOptionsFromEnv()
	for _, opt := range opts {
		cfg = opt.ApplyGRPCOption(cfg)
	}
	return cfg
}

// ApplyHTTPEnvConfigs applies the env configurations for HTTP.
func ApplyHTTPEnvConfigs(cfg Config) Config {
	opts := getOptionsFromEnv()
	for _, opt := range opts {
		cfg = opt.ApplyHTTPOption(cfg)
	}
	return cfg
}

func getOptionsFromEnv() []GenericOption {
	opts := []GenericOption{}

	tlsConf := &tls.Config{}
	DefaultEnvOptionsReader.Apply(
		envconfig.WithURL("ENDPOINT", func(u *url.URL) {
			opts = append(opts, withEndpointScheme(u))
			opts = append(opts, newSplitOption(func(cfg Config) Config {
				cfg.Traces.Endpoint = u.Host
				// For OTLP/HTTP endpoint URLs without a per-signal
				// configuration, the passed endpoint is used as a base URL
				// and the signals are sent to these paths relative to that.
				cfg.Traces.URLPath = path.Join(u.Path, DefaultTracesPath)
				return cfg
			}, withEndpointForGRPC(u)))
		}),
		envconfig.WithURL("TRACES_ENDPOINT", func(u *url.URL) {
			opts = append(opts, withEndpointScheme(u))
			opts = append(opts, newSplitOption(func(cfg Config) Config {
				cfg.Traces.Endpoint = u.Host
				// For endpoint URLs for OTLP/HTTP per-signal variables, the
				// URL MUST be used as-is without any modification. The only
				// exception is that if an URL contains no path part, the root
				// path / MUST be used.
				path := u.Path
				if path == "" {
					path = "/"
				}
				cfg.Traces.URLPath = path
				return cfg
			}, withEndpointForGRPC(u)))
		}),
		envconfig.WithCertPool("CERTIFICATE", func(p *x509.CertPool) { tlsConf.RootCAs = p }),
		envconfig.WithCertPool("TRACES_CERTIFICATE", func(p *x509.CertPool) { tlsConf.RootCAs = p }),
		envconfig.WithClientCert("CLIENT_CERTIFICATE", "CLIENT_KEY", func(c tls.Certificate) { tlsConf.Certificates = []tls.Certificate{c} }),
		envconfig.WithClientCert("TRACES_CLIENT_CERTIFICATE", "TRACES_CLIENT_KEY", func(c tls.Certificate) { tlsConf.Certificates = []tls.Certificate{c} }),
		withTLSConfig(tlsConf, func(c *tls.Config) { opts = append(opts, WithTLSClientConfig(c)) }),
		envconfig.WithBool("INSECURE", func(b bool) { opts = append(opts, withInsecure(b)) }),
		envconfig.WithBool("TRACES_INSECURE", func(b bool) { opts = append(opts, withInsecure(b)) }),
		envconfig.WithHeaders("HEADERS", func(h map[string]string) { opts = append(opts, WithHeaders(h)) }),
		envconfig.WithHeaders("TRACES_HEADERS", func(h map[string]string) { opts = append(opts, WithHeaders(h)) }),
		WithEnvCompression("COMPRESSION", func(c Compression) { opts = append(opts, WithCompression(c)) }),
		WithEnvCompression("TRACES_COMPRESSION", func(c Compression) { opts = append(opts, WithCompression(c)) }),
		envconfig.WithDuration("TIMEOUT", func(d time.Duration) { opts = append(opts, WithTimeout(d)) }),
		envconfig.WithDuration("TRACES_TIMEOUT", func(d time.Duration) { opts = append(opts, WithTimeout(d)) }),
	)

	return opts
}

func withEndpointScheme(u *url.URL) GenericOption {
	switch strings.ToLower(u.Scheme) {
	case "http", "unix":
		return WithInsecure()
	default:
		return WithSecure()
	}
}

func withEndpointForGRPC(u *url.URL) func(cfg Config) Config {
	return func(cfg Config) Config {
		// For OTLP/gRPC endpoints, this is the target to which the
		// exporter is going to send telemetry.
		cfg.Traces.Endpoint = path.Join(u.Host, u.Path)
		return cfg
	}
}

// WithEnvCompression retrieves the specified config and passes it to ConfigFn as a Compression.
func WithEnvCompression(n string, fn func(Compression)) func(e *envconfig.EnvOptionsReader) {
	return func(e *envconfig.EnvOptionsReader) {
		if v, ok := e.GetEnvValue(n); ok {
			cp := NoCompression
			if v == "gzip" {
				cp = GzipCompression
			}

			fn(cp)
		}
	}
}

// revive:disable-next-line:flag-parameter
func withInsecure(b bool) GenericOption {
	if b {
		return WithInsecure()
	}
	return WithSecure()
}

func withTLSConfig(c *tls.Config, fn func(*tls.Config)) func(e *envconfig.EnvOptionsReader) {
	return func(e *envconfig.EnvOptionsReader) {
		if c.RootCAs != nil || len(c.Certificates) > 0 {
			fn(c)
		}
	}
}
