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
	"net"

	"github.com/spf13/pflag"
	"go.opentelemetry.io/otel/exporters/otlp/otlpgrpc"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/semconv"
	"google.golang.org/grpc"

	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/egressselector"
	"k8s.io/apiserver/pkg/tracing"
	"k8s.io/component-base/traces"
	"k8s.io/utils/path"
)

const apiserverService = "apiserver"

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

	npConfig, err := tracing.ReadTracingConfiguration(o.ConfigFile)
	if err != nil {
		return fmt.Errorf("failed to read tracing config: %v", err)
	}

	errs := tracing.ValidateTracingConfiguration(npConfig)
	if len(errs) > 0 {
		return fmt.Errorf("failed to validate tracing configuration: %v", errs.ToAggregate())
	}

	opts := []otlpgrpc.Option{}
	if npConfig.Endpoint != nil {
		opts = append(opts, otlpgrpc.WithEndpoint(*npConfig.Endpoint))
	}
	if es != nil {
		// Only use the egressselector dialer if egressselector is enabled.
		// Endpoint is on the "ControlPlane" network
		egressDialer, err := es.Lookup(egressselector.ControlPlane.AsNetworkContext())
		if err != nil {
			return err
		}

		otelDialer := func(ctx context.Context, addr string) (net.Conn, error) {
			return egressDialer(ctx, "tcp", addr)
		}
		opts = append(opts, otlpgrpc.WithDialOption(grpc.WithContextDialer(otelDialer)))
	}

	sampler := sdktrace.NeverSample()
	if npConfig.SamplingRatePerMillion != nil && *npConfig.SamplingRatePerMillion > 0 {
		sampler = sdktrace.TraceIDRatioBased(float64(*npConfig.SamplingRatePerMillion) / float64(1000000))
	}

	resourceOpts := []resource.Option{
		resource.WithAttributes(
			semconv.ServiceNameKey.String(apiserverService),
			semconv.ServiceInstanceIDKey.String(c.APIServerID),
		),
	}
	tp := traces.NewProvider(context.Background(), sampler, resourceOpts, opts...)
	c.TracerProvider = &tp
	if c.LoopbackClientConfig != nil {
		c.LoopbackClientConfig.Wrap(traces.WrapperFor(c.TracerProvider))
	}
	return nil
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
