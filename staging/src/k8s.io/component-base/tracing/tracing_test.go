/*
Copyright 2022 The Kubernetes Authors.

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
	"bytes"
	"context"
	"flag"
	"fmt"
	"strings"
	"testing"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"k8s.io/klog/v2"
	utiltrace "k8s.io/utils/trace"
)

func init() {
	klog.InitFlags(flag.CommandLine)
	flag.CommandLine.Lookup("logtostderr").Value.Set("false")
}

func TestOpenTelemetryTracing(t *testing.T) {
	// Setup OpenTelemetry for testing
	fakeRecorder := tracetest.NewSpanRecorder()
	otelTracer := trace.NewTracerProvider(trace.WithSpanProcessor(fakeRecorder)).Tracer(instrumentationScope)

	func() {
		ctx := context.Background()
		// Create a parent OpenTelemetry span
		ctx, span := otelTracer.Start(ctx, "parent otel span")
		defer span.End()

		// Creates a child span
		_, tr := Start(ctx, "frobber", attribute.String("foo", "bar"))
		defer tr.End(10 * time.Second)

		time.Sleep(5 * time.Millisecond)
		// Add one event to the frobber span
		tr.AddEvent("reticulated splines", attribute.Bool("should I do it?", false)) // took 5ms
		time.Sleep(10 * time.Millisecond)

		// Add error event to the frobber span
		tr.RecordError(fmt.Errorf("something went wrong"))

		// Ensure setting context with span makes the next span a child
		ctx = ContextWithSpan(context.Background(), tr)

		// Add another event to the frobber span after getting the span from context
		SpanFromContext(ctx).AddEvent("sequenced particles", attribute.Int("inches in foot", 12)) // took 10ms

		// Creates a nested child span
		_, tr = Start(ctx, "nested child span")
		defer tr.End(10 * time.Second)
	}()

	output := fakeRecorder.Ended()
	if len(output) != 3 {
		t.Fatalf("got %d; expected len(output) == 3", len(output))
	}
	// Nested child span is ended first
	nestedChild := output[0]
	if nestedChild.Name() != "nested child span" {
		t.Fatalf("got %s; expected nestedChild.Name() == nested child span", nestedChild.Name())
	}
	// Child span is ended second
	child := output[1]
	if !nestedChild.Parent().Equal(child.SpanContext()) {
		t.Errorf("got child: %v, parent: %v; expected child.Parent() == parent.SpanContext()", nestedChild.Parent(), child.SpanContext())
	}
	if child.Name() != "frobber" {
		t.Errorf("got %s; expected child.Name() == frobber", child.Name())
	}
	if len(child.Attributes()) != 1 {
		t.Errorf("got attributes %v; expected one attribute in child.Attributes()", child.Attributes())
	}
	if len(child.Events()) != 3 {
		t.Errorf("got events %v; expected 2 events in child.Events()", child.Events())
	}
	if child.Events()[0].Name != "reticulated splines" {
		t.Errorf("got event %v; expected child.Events()[0].Name == reticulated splines", child.Events()[0])
	}
	if len(child.Events()[0].Attributes) != 1 {
		t.Errorf("got event %v; expected 1 attribute in child.Events()[0].Attributes", child.Events()[0])
	}
	if child.Events()[1].Name != "exception" {
		t.Errorf("got event %v; expected child.Events()[1].Name == something went wrong", child.Events()[1])
	}
	if len(child.Events()[1].Attributes) != 2 {
		t.Errorf("got event %#v; expected 2 attribute in child.Events()[1].Attributes", child.Events()[1])
	}
	if child.Events()[2].Name != "sequenced particles" {
		t.Errorf("got event %v; expected child.Events()[2].Name == sequenced particles", child.Events()[2])
	}
	if len(child.Events()[2].Attributes) != 1 {
		t.Errorf("got event %v; expected 1 attribute in child.Events()[2].Attributes", child.Events()[2])
	}
	// Parent span is ended last
	parent := output[2]
	if !child.Parent().Equal(parent.SpanContext()) {
		t.Fatalf("got child: %v, parent: %v; expected child.Parent() == parent.SpanContext()", child.Parent(), parent.SpanContext())
	}
	if parent.Name() != "parent otel span" {
		t.Fatalf("got %s; expected parent.Name() == parent otel span", parent.Name())
	}
	if len(parent.Attributes()) != 0 {
		t.Fatalf("got attributes %v; expected empty parent.Attributes()", parent.Attributes())
	}
}

func TestUtilTracing(t *testing.T) {
	var buf bytes.Buffer
	klog.SetOutput(&buf)

	ctx := context.Background()
	// Create a utiltracing span
	tr0 := utiltrace.New("parent utiltrace span")
	ctx = utiltrace.ContextWithTrace(ctx, tr0)

	// Creates a child span
	_, tr1 := Start(ctx, "frobber", attribute.String("foo", "bar"))

	time.Sleep(5 * time.Millisecond)
	// Add one event to the frobber span
	tr1.AddEvent("reticulated splines", attribute.Bool("should I do it?", false)) // took 5ms

	time.Sleep(10 * time.Millisecond)

	// Ensure setting context with span makes the next span a child
	ctx = ContextWithSpan(context.Background(), tr1)

	// Add another event to the frobber span after getting the span from context
	SpanFromContext(ctx).AddEvent("sequenced particles", attribute.Int("inches in foot", 12)) // took 10ms

	// Creates a nested child span
	_, tr2 := Start(ctx, "nested child span")
	// always log
	tr2.End(0 * time.Second)
	tr1.End(0 * time.Second)

	// Since all traces are nested, no logging should have occurred yet
	if buf.String() != "" {
		t.Errorf("child traces were printed out before the parent span completed: %v", buf.String())
	}

	// Now, end the parent span to cause logging to occur
	tr0.Log()

	expected := []string{
		`"frobber" foo:bar`,
		`---"reticulated splines" should I do it?:false`,
		`---"sequenced particles" inches in foot:12`,
		`"nested child span"`,
		`"parent utiltrace span"`,
	}
	for _, msg := range expected {
		if !strings.Contains(buf.String(), msg) {
			t.Errorf("\nMsg %q not found in log: \n%v\n", msg, buf.String())
		}
	}
}

func TestContextNoPanic(t *testing.T) {
	ctx := context.Background()
	// Make sure calling functions on spans from context doesn't panic
	SpanFromContext(ctx).AddEvent("foo")
	SpanFromContext(ctx).End(time.Minute)
}
