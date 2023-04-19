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
	"fmt"
	"os"
	"path/filepath"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

type (
	// telemetrySDK is a Detector that provides information about
	// the OpenTelemetry SDK used.  This Detector is included as a
	// builtin. If these resource attributes are not wanted, use
	// the WithTelemetrySDK(nil) or WithoutBuiltin() options to
	// explicitly disable them.
	telemetrySDK struct{}

	// host is a Detector that provides information about the host
	// being run on. This Detector is included as a builtin. If
	// these resource attributes are not wanted, use the
	// WithHost(nil) or WithoutBuiltin() options to explicitly
	// disable them.
	host struct{}

	stringDetector struct {
		schemaURL string
		K         attribute.Key
		F         func() (string, error)
	}

	defaultServiceNameDetector struct{}
)

var (
	_ Detector = telemetrySDK{}
	_ Detector = host{}
	_ Detector = stringDetector{}
	_ Detector = defaultServiceNameDetector{}
)

// Detect returns a *Resource that describes the OpenTelemetry SDK used.
func (telemetrySDK) Detect(context.Context) (*Resource, error) {
	return NewWithAttributes(
		semconv.SchemaURL,
		semconv.TelemetrySDKName("opentelemetry"),
		semconv.TelemetrySDKLanguageGo,
		semconv.TelemetrySDKVersion(otel.Version()),
	), nil
}

// Detect returns a *Resource that describes the host being run on.
func (host) Detect(ctx context.Context) (*Resource, error) {
	return StringDetector(semconv.SchemaURL, semconv.HostNameKey, os.Hostname).Detect(ctx)
}

// StringDetector returns a Detector that will produce a *Resource
// containing the string as a value corresponding to k. The resulting Resource
// will have the specified schemaURL.
func StringDetector(schemaURL string, k attribute.Key, f func() (string, error)) Detector {
	return stringDetector{schemaURL: schemaURL, K: k, F: f}
}

// Detect returns a *Resource that describes the string as a value
// corresponding to attribute.Key as well as the specific schemaURL.
func (sd stringDetector) Detect(ctx context.Context) (*Resource, error) {
	value, err := sd.F()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", string(sd.K), err)
	}
	a := sd.K.String(value)
	if !a.Valid() {
		return nil, fmt.Errorf("invalid attribute: %q -> %q", a.Key, a.Value.Emit())
	}
	return NewWithAttributes(sd.schemaURL, sd.K.String(value)), nil
}

// Detect implements Detector.
func (defaultServiceNameDetector) Detect(ctx context.Context) (*Resource, error) {
	return StringDetector(
		semconv.SchemaURL,
		semconv.ServiceNameKey,
		func() (string, error) {
			executable, err := os.Executable()
			if err != nil {
				return "unknown_service:go", nil
			}
			return "unknown_service:" + filepath.Base(executable), nil
		},
	).Detect(ctx)
}
