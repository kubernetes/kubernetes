// Copyright (c) 2016 Uber Technologies, Inc.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package config

import (
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/opentracing/opentracing-go"

	"github.com/uber/jaeger-client-go"
	"github.com/uber/jaeger-client-go/rpcmetrics"
	"github.com/uber/jaeger-client-go/transport"
	"github.com/uber/jaeger-client-go/transport/udp"
)

const defaultSamplingProbability = 0.001

// Configuration configures and creates Jaeger Tracer
type Configuration struct {
	Disabled   bool            `yaml:"disabled"`
	Sampler    *SamplerConfig  `yaml:"sampler"`
	Reporter   *ReporterConfig `yaml:"reporter"`
	RPCMetrics bool            `yaml:"rpc_metrics"`
}

// SamplerConfig allows initializing a non-default sampler.  All fields are optional.
type SamplerConfig struct {
	// Type specifies the type of the sampler: const, probabilistic, rateLimiting, or remote
	Type string `yaml:"type"`

	// Param is a value passed to the sampler.
	// Valid values for Param field are:
	// - for "const" sampler, 0 or 1 for always false/true respectively
	// - for "probabilistic" sampler, a probability between 0 and 1
	// - for "rateLimiting" sampler, the number of spans per second
	// - for "remote" sampler, param is the same as for "probabilistic"
	//   and indicates the initial sampling rate before the actual one
	//   is received from the mothership
	Param float64 `yaml:"param"`

	// SamplingServerURL is the address of jaeger-agent's HTTP sampling server
	SamplingServerURL string `yaml:"samplingServerURL"`

	// MaxOperations is the maximum number of operations that the sampler
	// will keep track of. If an operation is not tracked, a default probabilistic
	// sampler will be used rather than the per operation specific sampler.
	MaxOperations int `yaml:"maxOperations"`

	// SamplingRefreshInterval controls how often the remotely controlled sampler will poll
	// jaeger-agent for the appropriate sampling strategy.
	SamplingRefreshInterval time.Duration `yaml:"samplingRefreshInterval"`
}

// ReporterConfig configures the reporter. All fields are optional.
type ReporterConfig struct {
	// QueueSize controls how many spans the reporter can keep in memory before it starts dropping
	// new spans. The queue is continuously drained by a background go-routine, as fast as spans
	// can be sent out of process.
	QueueSize int `yaml:"queueSize"`

	// BufferFlushInterval controls how often the buffer is force-flushed, even if it's not full.
	// It is generally not useful, as it only matters for very low traffic services.
	BufferFlushInterval time.Duration

	// LogSpans, when true, enables LoggingReporter that runs in parallel with the main reporter
	// and logs all submitted spans. Main Configuration.Logger must be initialized in the code
	// for this option to have any effect.
	LogSpans bool `yaml:"logSpans"`

	// LocalAgentHostPort instructs reporter to send spans to jaeger-agent at this address
	LocalAgentHostPort string `yaml:"localAgentHostPort"`
}

type nullCloser struct{}

func (*nullCloser) Close() error { return nil }

// New creates a new Jaeger Tracer, and a closer func that can be used to flush buffers
// before shutdown.
func (c Configuration) New(
	serviceName string,
	options ...Option,
) (opentracing.Tracer, io.Closer, error) {
	if serviceName == "" {
		return nil, nil, errors.New("no service name provided")
	}
	if c.Disabled {
		return &opentracing.NoopTracer{}, &nullCloser{}, nil
	}
	opts := applyOptions(options...)
	tracerMetrics := jaeger.NewMetrics(opts.metrics, nil)
	if c.RPCMetrics {
		Observer(
			rpcmetrics.NewObserver(
				opts.metrics.Namespace("jaeger", map[string]string{"component": "jaeger"}),
				rpcmetrics.DefaultNameNormalizer,
			),
		)(&opts) // adds to c.observers
	}
	if c.Sampler == nil {
		c.Sampler = &SamplerConfig{
			Type:  jaeger.SamplerTypeRemote,
			Param: defaultSamplingProbability,
		}
	}
	if c.Reporter == nil {
		c.Reporter = &ReporterConfig{}
	}

	sampler, err := c.Sampler.NewSampler(serviceName, tracerMetrics)
	if err != nil {
		return nil, nil, err
	}

	reporter := opts.reporter
	if reporter == nil {
		r, err := c.Reporter.NewReporter(serviceName, tracerMetrics, opts.logger)
		if err != nil {
			return nil, nil, err
		}
		reporter = r
	}

	tracerOptions := []jaeger.TracerOption{
		jaeger.TracerOptions.Metrics(tracerMetrics),
		jaeger.TracerOptions.Logger(opts.logger),
	}

	for _, obs := range opts.observers {
		tracerOptions = append(tracerOptions, jaeger.TracerOptions.Observer(obs))
	}

	tracer, closer := jaeger.NewTracer(
		serviceName,
		sampler,
		reporter,
		tracerOptions...)

	return tracer, closer, nil
}

// InitGlobalTracer creates a new Jaeger Tracer, and sets is as global OpenTracing Tracer.
// It returns a closer func that can be used to flush buffers before shutdown.
func (c Configuration) InitGlobalTracer(
	serviceName string,
	options ...Option,
) (io.Closer, error) {
	if c.Disabled {
		return &nullCloser{}, nil
	}
	tracer, closer, err := c.New(serviceName, options...)
	if err != nil {
		return nil, err
	}
	opentracing.InitGlobalTracer(tracer)
	return closer, nil
}

// NewSampler creates a new sampler based on the configuration
func (sc *SamplerConfig) NewSampler(
	serviceName string,
	metrics *jaeger.Metrics,
) (jaeger.Sampler, error) {
	samplerType := strings.ToLower(sc.Type)
	if samplerType == jaeger.SamplerTypeConst {
		return jaeger.NewConstSampler(sc.Param != 0), nil
	}
	if samplerType == jaeger.SamplerTypeProbabilistic {
		if sc.Param >= 0 && sc.Param <= 1.0 {
			return jaeger.NewProbabilisticSampler(sc.Param)
		}
		return nil, fmt.Errorf(
			"Invalid Param for probabilistic sampler: %v. Expecting value between 0 and 1",
			sc.Param,
		)
	}
	if samplerType == jaeger.SamplerTypeRateLimiting {
		return jaeger.NewRateLimitingSampler(sc.Param), nil
	}
	if samplerType == jaeger.SamplerTypeRemote || sc.Type == "" {
		sc2 := *sc
		sc2.Type = jaeger.SamplerTypeProbabilistic
		initSampler, err := sc2.NewSampler(serviceName, nil)
		if err != nil {
			return nil, err
		}
		options := []jaeger.SamplerOption{
			jaeger.SamplerOptions.Metrics(metrics),
			jaeger.SamplerOptions.InitialSampler(initSampler),
			jaeger.SamplerOptions.SamplingServerURL(sc.SamplingServerURL),
		}
		if sc.MaxOperations != 0 {
			options = append(options, jaeger.SamplerOptions.MaxOperations(sc.MaxOperations))
		}
		if sc.SamplingRefreshInterval != 0 {
			options = append(options, jaeger.SamplerOptions.SamplingRefreshInterval(sc.SamplingRefreshInterval))
		}
		return jaeger.NewRemotelyControlledSampler(serviceName, options...), nil
	}
	return nil, fmt.Errorf("Unknown sampler type %v", sc.Type)
}

// NewReporter instantiates a new reporter that submits spans to tcollector
func (rc *ReporterConfig) NewReporter(
	serviceName string,
	metrics *jaeger.Metrics,
	logger jaeger.Logger,
) (jaeger.Reporter, error) {
	sender, err := rc.newTransport()
	if err != nil {
		return nil, err
	}
	reporter := jaeger.NewRemoteReporter(
		sender,
		jaeger.ReporterOptions.QueueSize(rc.QueueSize),
		jaeger.ReporterOptions.BufferFlushInterval(rc.BufferFlushInterval),
		jaeger.ReporterOptions.Logger(logger),
		jaeger.ReporterOptions.Metrics(metrics))
	if rc.LogSpans && logger != nil {
		logger.Infof("Initializing logging reporter\n")
		reporter = jaeger.NewCompositeReporter(jaeger.NewLoggingReporter(logger), reporter)
	}
	return reporter, err
}

func (rc *ReporterConfig) newTransport() (transport.Transport, error) {
	return udp.NewUDPTransport(rc.LocalAgentHostPort, 0)
}
