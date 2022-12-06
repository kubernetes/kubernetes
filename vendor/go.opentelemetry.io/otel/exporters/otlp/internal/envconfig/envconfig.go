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

package envconfig // import "go.opentelemetry.io/otel/exporters/otlp/internal/envconfig"

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// ConfigFn is the generic function used to set a config.
type ConfigFn func(*EnvOptionsReader)

// EnvOptionsReader reads the required environment variables.
type EnvOptionsReader struct {
	GetEnv    func(string) string
	ReadFile  func(string) ([]byte, error)
	Namespace string
}

// Apply runs every ConfigFn.
func (e *EnvOptionsReader) Apply(opts ...ConfigFn) {
	for _, o := range opts {
		o(e)
	}
}

// GetEnvValue gets an OTLP environment variable value of the specified key
// using the GetEnv function.
// This function prepends the OTLP specified namespace to all key lookups.
func (e *EnvOptionsReader) GetEnvValue(key string) (string, bool) {
	v := strings.TrimSpace(e.GetEnv(keyWithNamespace(e.Namespace, key)))
	return v, v != ""
}

// WithString retrieves the specified config and passes it to ConfigFn as a string.
func WithString(n string, fn func(string)) func(e *EnvOptionsReader) {
	return func(e *EnvOptionsReader) {
		if v, ok := e.GetEnvValue(n); ok {
			fn(v)
		}
	}
}

// WithDuration retrieves the specified config and passes it to ConfigFn as a duration.
func WithDuration(n string, fn func(time.Duration)) func(e *EnvOptionsReader) {
	return func(e *EnvOptionsReader) {
		if v, ok := e.GetEnvValue(n); ok {
			if d, err := strconv.Atoi(v); err == nil {
				fn(time.Duration(d) * time.Millisecond)
			}
		}
	}
}

// WithHeaders retrieves the specified config and passes it to ConfigFn as a map of HTTP headers.
func WithHeaders(n string, fn func(map[string]string)) func(e *EnvOptionsReader) {
	return func(e *EnvOptionsReader) {
		if v, ok := e.GetEnvValue(n); ok {
			fn(stringToHeader(v))
		}
	}
}

// WithURL retrieves the specified config and passes it to ConfigFn as a net/url.URL.
func WithURL(n string, fn func(*url.URL)) func(e *EnvOptionsReader) {
	return func(e *EnvOptionsReader) {
		if v, ok := e.GetEnvValue(n); ok {
			if u, err := url.Parse(v); err == nil {
				fn(u)
			}
		}
	}
}

// WithTLSConfig retrieves the specified config and passes it to ConfigFn as a crypto/tls.Config.
func WithTLSConfig(n string, fn func(*tls.Config)) func(e *EnvOptionsReader) {
	return func(e *EnvOptionsReader) {
		if v, ok := e.GetEnvValue(n); ok {
			if b, err := e.ReadFile(v); err == nil {
				if c, err := createTLSConfig(b); err == nil {
					fn(c)
				}
			}
		}
	}
}

func keyWithNamespace(ns, key string) string {
	if ns == "" {
		return key
	}
	return fmt.Sprintf("%s_%s", ns, key)
}

func stringToHeader(value string) map[string]string {
	headersPairs := strings.Split(value, ",")
	headers := make(map[string]string)

	for _, header := range headersPairs {
		nameValue := strings.SplitN(header, "=", 2)
		if len(nameValue) < 2 {
			continue
		}
		name, err := url.QueryUnescape(nameValue[0])
		if err != nil {
			continue
		}
		trimmedName := strings.TrimSpace(name)
		value, err := url.QueryUnescape(nameValue[1])
		if err != nil {
			continue
		}
		trimmedValue := strings.TrimSpace(value)

		headers[trimmedName] = trimmedValue
	}

	return headers
}

func createTLSConfig(certBytes []byte) (*tls.Config, error) {
	cp := x509.NewCertPool()
	if ok := cp.AppendCertsFromPEM(certBytes); !ok {
		return nil, errors.New("failed to append certificate to the cert pool")
	}

	return &tls.Config{
		RootCAs: cp,
	}, nil
}
