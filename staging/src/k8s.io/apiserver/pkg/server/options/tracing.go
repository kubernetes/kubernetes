/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package options

import (
	"context"
	"fmt"
	"io/ioutil"
	"net"

	"github.com/spf13/pflag"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/sdk/resource"
	"go.opentelemetry.io/otel/semconv/v1.12.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"github.com/go-logr/logr"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/apis/apiserver/install"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/egressselector"
	"k8s.io/apiserver/pkg/util/feature"
	tracing "k8s.io/component-base/tracing"
	tracingapi "k8s.io/component-base/tracing/api/v1"
	"k8s.io/utils/path"
	"k8s.io/klog/v2"
)

const apiserverService = "apiserver"

var (
	cfgScheme = runtime.NewScheme()
	codecs    = serializer.NewCodecFactory(cfgScheme)
)

func init() {
	install.Install(cfgScheme)
}

// TracingOptions contain configuration options for tracing
// exporters
type TracingOptions struct {
	// ConfigFile is the file path with api-server tracing configuration.
	ConfigFile string
}

// NewTracingOptions creates a new instance of TracingOptions
func NewTracingOptions() *TracingOptions {
	return &TracingOptions{}
}

// AddFlags adds flags related to tracing to the specified FlagSet
func (o *TracingOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.ConfigFile, "tracing-config-file", o.ConfigFile,
		"File with apiserver tracing configuration.")
}

// ApplyTo fills up Tracing config with options.
func (o *TracingOptions) ApplyTo(es *egressselector.EgressSelector, c *server.Config) error {
	if o == nil || o.ConfigFile == "" {
		return nil
	}
	if !feature.DefaultFeatureGate.Enabled(features.APIServerTracing) {
		return fmt.Errorf("APIServerTracing feature is not enabled, but tracing config file was provided")
	}

	traceConfig, err := ReadTracingConfiguration(o.ConfigFile)
	if err != nil {
		return fmt.Errorf("failed to read tracing config: %v", err)
	}

	errs := tracingapi.ValidateTracingConfiguration(traceConfig, feature.DefaultFeatureGate, nil)
	if len(errs) > 0 {
		return fmt.Errorf("failed to validate tracing configuration: %v", errs.ToAggregate())
	}

	opts := []otlptracegrpc.Option{}
	if es != nil {
		// Only use the egressselector dialer if egressselector is enabled.
		// Endpoint is on the "ControlPlane" network
		egressDialer, err := es.Lookup(egressselector.ControlPlane.AsNetworkContext())
		if err != nil {
			return err
		}
		if egressDialer != nil {
			otelDialer := func(ctx context.Context, addr string) (net.Conn, error) {
				return egressDialer(ctx, "tcp", addr)
			}
			opts = append(opts, otlptracegrpc.WithDialOption(grpc.WithContextDialer(otelDialer)))
		}
	}

	resourceOpts := []resource.Option{
		resource.WithAttributes(
			semconv.ServiceNameKey.String(apiserverService),
			semconv.ServiceInstanceIDKey.String(c.APIServerID),
		),
	}
	tp, err := tracing.NewProvider(context.Background(), traceConfig, opts, resourceOpts)
	if err != nil {
		return err
	}
	c.TracerProvider = tp
	if c.LoopbackClientConfig != nil {
		c.LoopbackClientConfig.Wrap(tracing.WrapperFor(c.TracerProvider))
	}

	// Also enable logging of span and trace ID.
	//
	// This has to run after applying the logging configuration, something
	// that intentionally is done as early as possible before applying
	// other configuration options.
	logger := klog.Background()
	logger = logger.WithContextHandlers(logr.ContextHandler{ValuesFromContext: traceIDsFromContext})
	klog.SetLoggerWithOptions(logger, klog.ContextualLogger(true))

	return nil
}

func traceIDsFromContext(ctx context.Context) logr.KeysAndValues {
	spanContext := trace.SpanContextFromContext(ctx)
	if !spanContext.IsValid() {
		return nil
	}

	// A valid span context has both IDs set. These key names are
	// determined by
	// https://github.com/open-telemetry/opentelemetry-specification/tree/a87fb8c5a60f942d08c67c1582034eb8b4322009/specification/logs#trace-context-in-legacy-formats
	return logr.KeysAndValues{
		{ Key: "trace_id", Value: spanContext.TraceID().String() },
		{ Key: "span_id", Value: spanContext.SpanID().String() },
	}
}

// Validate verifies flags passed to TracingOptions.
func (o *TracingOptions) Validate() (errs []error) {
	if o == nil || o.ConfigFile == "" {
		return
	}

	if exists, err := path.Exists(path.CheckFollowSymlink, o.ConfigFile); !exists {
		errs = append(errs, fmt.Errorf("tracing-config-file %s does not exist", o.ConfigFile))
	} else if err != nil {
		errs = append(errs, fmt.Errorf("error checking if tracing-config-file %s exists: %v", o.ConfigFile, err))
	}
	return
}

// ReadTracingConfiguration reads the tracing configuration from a file
func ReadTracingConfiguration(configFilePath string) (*tracingapi.TracingConfiguration, error) {
	if configFilePath == "" {
		return nil, fmt.Errorf("tracing config file was empty")
	}
	data, err := ioutil.ReadFile(configFilePath)
	if err != nil {
		return nil, fmt.Errorf("unable to read tracing configuration from %q: %v", configFilePath, err)
	}
	internalConfig := &apiserver.TracingConfiguration{}
	// this handles json/yaml/whatever, and decodes all registered version to the internal version
	if err := runtime.DecodeInto(codecs.UniversalDecoder(), data, internalConfig); err != nil {
		return nil, fmt.Errorf("unable to decode tracing configuration data: %v", err)
	}
	tc := &tracingapi.TracingConfiguration{
		Endpoint:               internalConfig.Endpoint,
		SamplingRatePerMillion: internalConfig.SamplingRatePerMillion,
	}
	return tc, nil
}
