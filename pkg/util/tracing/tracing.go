/*
Copyright 2017 The Kubernetes Authors.

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

package tracing

import (
	"io"

	"github.com/golang/glog"
	jaegerclientconfig "github.com/uber/jaeger-client-go/config"
)

var SupportedTracers = []string{"jaeger"}

func InitGlobalTracer(tracer string, serviceName string) io.Closer {
	switch tracer {
	case "jaeger":
		glog.Info("Initializing Jaeger tarcer")
		closer := initGlobalJaegerTracer(serviceName)
		return closer
	case "":
	default:
		glog.Errorf("Unsupported tracer %v won't be started.", tracer)
	}
	return nil
}

// Jaeger tracer

type glogLogger struct{}

func (g *glogLogger) Error(msg string) {
	glog.Error(msg)
}

func (g *glogLogger) Infof(fmt string, args ...interface{}) {
	glog.Infof(fmt, args)
}

func initGlobalJaegerTracer(serviceName string) io.Closer {
	// initialize tracer using Configuration struct
	cfg := jaegerclientconfig.Configuration{
		Sampler: &jaegerclientconfig.SamplerConfig{
			Type:  "const",
			Param: 1,
		},
		Reporter: &jaegerclientconfig.ReporterConfig{
			QueueSize: 1000,
			LogSpans:  false,
		},
	}
	logger := &glogLogger{}
	closer, err := cfg.InitGlobalTracer(serviceName,
		jaegerclientconfig.Logger(logger),
	)
	if err != nil {
		glog.Fatalf("cannot initialize Jaeger Tracer: %v", err)
	}
	return closer
}
