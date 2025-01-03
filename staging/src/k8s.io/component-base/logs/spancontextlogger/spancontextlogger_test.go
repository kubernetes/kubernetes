/*
Copyright 2024 The Kubernetes Authors.

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

package spancontextlogger

import (
	"regexp"
	"testing"

	"go.opentelemetry.io/otel/trace"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
)

// https://github.com/kubernetes/klog/blob/7f0268803762d7acc0231ca71e0b49c6756275ba/ktesting/testinglogger_test.go#L26
var headerRe = regexp.MustCompile(`([IE])[[:digit:]]{4} [[:digit:]]{2}:[[:digit:]]{2}:[[:digit:]]{2}\.[[:digit:]]{6}\] `)

func TestLoggerWithSpanContext(t *testing.T) {
	var buffer ktesting.BufferTL
	logger, ctx := ktesting.NewTestContext(&buffer)

	// Log once without a span context
	logger = klog.FromContext(ctx)
	logger = LoggerWithSpanContext(ctx, logger)
	logger.Info("hello world")

	// Log again with a sampled span context
	traceID, err := trace.TraceIDFromHex("fffefdfcfbfaf9f8f7f6f5f4f3f2f1f0")
	if err != nil {
		t.Fatalf("Error parsing trace ID: %v", err)
	}
	spanID, err := trace.SpanIDFromHex("fefdfcfbfaf9f8f7")
	if err != nil {
		t.Fatalf("Error parsing span ID: %v", err)
	}
	ctx = trace.ContextWithSpanContext(ctx, trace.NewSpanContext(trace.SpanContextConfig{
		SpanID:     spanID,
		TraceID:    traceID,
		TraceFlags: trace.FlagsSampled,
	}))
	logger = klog.FromContext(ctx)
	logger = LoggerWithSpanContext(ctx, logger)
	logger.Info("hello world")

	state := klog.CaptureState()
	defer state.Restore()

	testingLogger, ok := logger.GetSink().(ktesting.Underlier)
	if !ok {
		t.Fatal("Should have had a ktesting LogSink!?")
	}

	actual := testingLogger.GetBuffer().String()
	if actual != "" {
		t.Errorf("testinglogger should not have buffered, got:\n%s", actual)
	}

	actual = buffer.String()
	actual = headerRe.ReplaceAllString(actual, "${1}xxx] ")
	expected := `Ixxx] hello world
Ixxx] hello world trace_id="fffefdfcfbfaf9f8f7f6f5f4f3f2f1f0" span_id="fefdfcfbfaf9f8f7"
`
	if actual != expected {
		t.Errorf("mismatch in captured output, expected:\n%s\ngot:\n%s\n", expected, actual)
	}
}
