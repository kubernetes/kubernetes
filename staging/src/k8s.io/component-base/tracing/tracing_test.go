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
	encodingjson "encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	oteltrace "go.opentelemetry.io/otel/trace"
	"go.uber.org/zap/zapcore"
	logsjson "k8s.io/component-base/logs/json"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/klog/v2/textlogger"
	utiltrace "k8s.io/utils/trace"
)

func init() {
	klog.InitFlags(flag.CommandLine)
	_ = flag.CommandLine.Lookup("logtostderr").Value.Set("false")
	_ = flag.CommandLine.Lookup("v").Value.Set("2")
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
	_ = ContextWithSpan(ctx, nil)
	_ = ContextWithSpanLogger(ctx, nil)
}

var logHeaderRe = regexp.MustCompile(`([IE])[[:digit:]]{4} [[:digit:]]{2}:[[:digit:]]{2}:[[:digit:]]{2}\.[[:digit:]]{6}\] `)

func TestContextualLoggingWithTracing(t *testing.T) {
	var buffer ktesting.BufferTL
	baseLogger, ctx := ktesting.NewTestContext(&buffer)

	// 1. Log without a span in context
	klog.FromContext(ctx).Info("no span")

	// 2. Log with an unsampled span
	unsampledTP := trace.NewTracerProvider(trace.WithSampler(trace.NeverSample()))
	unsampledCtx, unsampledSpan := unsampledTP.Tracer(instrumentationScope).Start(ctx, "unsampled parent")
	unsampledCtx = ContextWithSpanLogger(unsampledCtx, unsampledSpan)
	unsampledChildCtx, unsampledChild := Start(unsampledCtx, "unsampled child")
	klog.FromContext(unsampledChildCtx).Info("unsampled span")
	unsampledChild.End(10 * time.Second)
	unsampledSpan.End()

	// 3. Log with sampled HTTP handler and nested spans
	fakeRecorder := tracetest.NewSpanRecorder()
	sampledTP := trace.NewTracerProvider(
		trace.WithSampler(trace.AlwaysSample()),
		trace.WithSpanProcessor(fakeRecorder),
	)

	handler := WithTracing(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqCtx := r.Context()
		klog.FromContext(reqCtx).Info("in http handler")

		childCtx, childSpan := Start(reqCtx, "child operation")
		defer childSpan.End(10 * time.Second)
		klog.FromContext(childCtx).Info("in child span")

		// Verify ContextWithSpan also attaches the span logger to a fresh context with baseLogger
		transferredCtx := ContextWithSpan(klog.NewContext(context.Background(), baseLogger), childSpan)
		klog.FromContext(transferredCtx).Info("in transferred context")
	}), sampledTP, "http-request")

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://example.com/api/v1/pods", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	handler.ServeHTTP(httptest.NewRecorder(), req)

	ended := fakeRecorder.Ended()
	if len(ended) != 2 {
		t.Fatalf("expected 2 ended spans, got %d", len(ended))
	}
	childRecorded := ended[0]
	httpRecorded := ended[1]
	traceID := httpRecorded.SpanContext().TraceID().String()
	httpSpanID := httpRecorded.SpanContext().SpanID().String()
	childSpanID := childRecorded.SpanContext().SpanID().String()

	actual := logHeaderRe.ReplaceAllString(buffer.String(), "${1}xxx] ")
	expected := fmt.Sprintf(`Ixxx] no span
Ixxx] unsampled span
Ixxx] in http handler trace_id=%q span_id=%q
Ixxx] in child span trace_id=%q span_id=%q
Ixxx] in transferred context trace_id=%q span_id=%q
`, traceID, httpSpanID, traceID, childSpanID, traceID, childSpanID)

	if actual != expected {
		t.Errorf("unexpected log output:\ngot:\n%s\nwant:\n%s", actual, expected)
	}
}

func TestContextualLoggingNoopWithSampledTraceparent(t *testing.T) {
	var buffer ktesting.BufferTL
	_, ctx := ktesting.NewTestContext(&buffer)

	handler := WithTracing(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		klog.FromContext(r.Context()).Info("in noop handler")
		childCtx, childSpan := Start(r.Context(), "child")
		defer childSpan.End(10 * time.Second)
		klog.FromContext(childCtx).Info("in noop child")
	}), NewNoopTracerProvider(), "http-request")

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://example.com/api/v1/pods", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	req.Header.Set("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
	handler.ServeHTTP(httptest.NewRecorder(), req)

	output := buffer.String()
	if strings.Contains(output, "trace_id=") || strings.Contains(output, "span_id=") {
		t.Errorf("expected noop provider with sampled traceparent not to log trace_id or span_id, got:\n%s", output)
	}
	if !strings.Contains(output, "in noop handler") || !strings.Contains(output, "in noop child") {
		t.Errorf("expected handler and child log messages, got:\n%s", output)
	}
}

func TestContextualLoggingNestedAndTransition(t *testing.T) {
	sampledRecorder := tracetest.NewSpanRecorder()
	sampledTP := trace.NewTracerProvider(
		trace.WithSampler(trace.AlwaysSample()),
		trace.WithSpanProcessor(sampledRecorder),
	)
	unsampledTP := trace.NewTracerProvider(
		trace.WithSampler(trace.NeverSample()),
	)

	for _, format := range []string{"text", "json"} {
		t.Run(format, func(t *testing.T) {
			var buf bytes.Buffer
			var baseLogger klog.Logger
			if format == "text" {
				baseLogger = textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(&buf)))
			} else {
				baseLogger, _ = logsjson.NewJSONLogger(0, zapcore.AddSync(&buf), nil, nil)
			}
			baseCtx := klog.NewContext(context.Background(), baseLogger)

			// Outer sampled span
			outerCtx, outerSpan := sampledTP.Tracer(instrumentationScope).Start(baseCtx, "outer")
			outerCtx = ContextWithSpanLogger(outerCtx, outerSpan)

			// Add intermediate contextual logger values and name after outer span
			enrichedCtx := klog.NewContext(outerCtx, klog.LoggerWithName(klog.LoggerWithValues(klog.FromContext(outerCtx), "req", "r1"), "sub"))

			// Inner sampled span (nested child)
			innerCtx, innerSpan := Start(enrichedCtx, "inner")
			_, _, infoLine, _ := runtime.Caller(0)
			klog.FromContext(innerCtx).Info("in inner span", "extra", "val")
			infoLine++

			_, _, errLine, _ := runtime.Caller(0)
			klog.FromContext(innerCtx).Error(errors.New("boom"), "error in inner span", "extra", "errval")
			errLine++

			// Transition from sampled -> unsampled (e.g. WithNewRoot unsampled span)
			unsampledCtx, unsampledSpan := unsampledTP.Tracer(instrumentationScope).Start(innerCtx, "unsampled-root", oteltrace.WithNewRoot())
			unsampledCtx = ContextWithSpanLogger(unsampledCtx, unsampledSpan)
			_, _, unsampledLine, _ := runtime.Caller(0)
			klog.FromContext(unsampledCtx).Info("in unsampled root")
			unsampledLine++
			klog.FromContext(unsampledCtx).Error(errors.New("unsampled-boom"), "error in unsampled root")

			// Transition back from unsampled -> sampled
			resumedCtx, resumedSpan := sampledTP.Tracer(instrumentationScope).Start(unsampledCtx, "resumed-root", oteltrace.WithNewRoot())
			resumedCtx = ContextWithSpanLogger(resumedCtx, resumedSpan)
			klog.FromContext(resumedCtx).Info("in resumed root")

			// Verify outerCtx was not mutated by inner/unsampled transitions
			klog.FromContext(outerCtx).Info("back in outer span")

			resumedSpan.End()
			unsampledSpan.End()
			innerSpan.End(10 * time.Second)
			outerSpan.End()

			lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
			if len(lines) != 6 {
				t.Fatalf("expected 6 log lines, got %d:\n%s", len(lines), buf.String())
			}

			outerSC := outerSpan.SpanContext()
			innerSC := innerSpan.otelSpan.SpanContext()
			resumedSC := resumedSpan.SpanContext()

			if format == "json" {
				checkJSONLine := func(idx int, wantMsg string, wantCaller string, wantTraceID, wantSpanID string, wantEnriched bool) {
					t.Helper()
					line := lines[idx]
					wantCount := 0
					if wantTraceID != "" {
						wantCount = 1
					}
					if got := strings.Count(line, `"trace_id"`); got != wantCount {
						t.Errorf("line %d: expected %d trace_id occurrences, got %d: %s", idx, wantCount, got, line)
					}
					if got := strings.Count(line, `"span_id"`); got != wantCount {
						t.Errorf("line %d: expected %d span_id occurrences, got %d: %s", idx, wantCount, got, line)
					}
					var entry map[string]any
					if err := encodingjson.Unmarshal([]byte(line), &entry); err != nil {
						t.Fatalf("line %d: invalid JSON: %v\n%s", idx, err, line)
					}
					if entry["msg"] != wantMsg {
						t.Errorf("line %d: msg=%v, want %q", idx, entry["msg"], wantMsg)
					}
					if wantCaller != "" && !strings.HasSuffix(fmt.Sprint(entry["caller"]), wantCaller) {
						t.Errorf("line %d: caller=%v, want suffix %q", idx, entry["caller"], wantCaller)
					}
					if wantTraceID != "" {
						if entry["trace_id"] != wantTraceID || entry["span_id"] != wantSpanID {
							t.Errorf("line %d: got trace_id=%v span_id=%v, want %q / %q", idx, entry["trace_id"], entry["span_id"], wantTraceID, wantSpanID)
						}
					}
					if wantEnriched {
						if entry["req"] != "r1" || entry["logger"] != "sub" {
							t.Errorf("line %d: expected req=r1 and logger=sub to be preserved, got %s", idx, line)
						}
					}
				}
				checkJSONLine(0, "in inner span", fmt.Sprintf("tracing_test.go:%d", infoLine), innerSC.TraceID().String(), innerSC.SpanID().String(), true)
				checkJSONLine(1, "error in inner span", fmt.Sprintf("tracing_test.go:%d", errLine), innerSC.TraceID().String(), innerSC.SpanID().String(), true)
				checkJSONLine(2, "in unsampled root", fmt.Sprintf("tracing_test.go:%d", unsampledLine), "", "", true)
				checkJSONLine(3, "error in unsampled root", "", "", "", true)
				checkJSONLine(4, "in resumed root", "", resumedSC.TraceID().String(), resumedSC.SpanID().String(), true)
				checkJSONLine(5, "back in outer span", "", outerSC.TraceID().String(), outerSC.SpanID().String(), false)
			} else {
				checkTextLine := func(idx int, wantCaller string, wantTraceID, wantSpanID string, wantEnriched bool) {
					t.Helper()
					line := lines[idx]
					if wantCaller != "" && !strings.Contains(line, wantCaller) {
						t.Errorf("line %d: expected caller %q, got %s", idx, wantCaller, line)
					}
					if wantTraceID == "" {
						if strings.Contains(line, "trace_id=") || strings.Contains(line, "span_id=") {
							t.Errorf("line %d: expected no trace_id/span_id, got %s", idx, line)
						}
					} else {
						if strings.Count(line, "trace_id=") != 1 || strings.Count(line, "span_id=") != 1 {
							t.Errorf("line %d: expected 1 trace_id and 1 span_id, got %s", idx, line)
						}
						if !strings.Contains(line, fmt.Sprintf("trace_id=%q", wantTraceID)) || !strings.Contains(line, fmt.Sprintf("span_id=%q", wantSpanID)) {
							t.Errorf("line %d: expected trace_id=%q span_id=%q, got %s", idx, wantTraceID, wantSpanID, line)
						}
					}
					if wantEnriched {
						if !strings.Contains(line, `req="r1"`) || !strings.Contains(line, `logger="sub"`) {
							t.Errorf("line %d: expected req=\"r1\" and logger=\"sub\" to be preserved, got %s", idx, line)
						}
					}
				}
				checkTextLine(0, fmt.Sprintf("tracing_test.go:%d", infoLine), innerSC.TraceID().String(), innerSC.SpanID().String(), true)
				checkTextLine(1, fmt.Sprintf("tracing_test.go:%d", errLine), innerSC.TraceID().String(), innerSC.SpanID().String(), true)
				checkTextLine(2, fmt.Sprintf("tracing_test.go:%d", unsampledLine), "", "", true)
				checkTextLine(3, "", "", "", true)
				checkTextLine(4, "", resumedSC.TraceID().String(), resumedSC.SpanID().String(), true)
				checkTextLine(5, "", outerSC.TraceID().String(), outerSC.SpanID().String(), false)
			}
		})
	}
}

type recordOnlySampler struct{}

func (recordOnlySampler) ShouldSample(trace.SamplingParameters) trace.SamplingResult {
	return trace.SamplingResult{Decision: trace.RecordOnly}
}

func (recordOnlySampler) Description() string { return "recordOnlySampler" }

func TestSpanLogSinkEdgeCases(t *testing.T) {
	sampledTP := trace.NewTracerProvider(trace.WithSampler(trace.AlwaysSample()))

	t.Run("nil span clears active span logger", func(t *testing.T) {
		var buf bytes.Buffer
		baseCtx := klog.NewContext(context.Background(), textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(&buf))))
		ctx, span := sampledTP.Tracer(instrumentationScope).Start(baseCtx, "outer")
		defer span.End()
		ctx = ContextWithSpanLogger(ctx, span)
		ctx = ContextWithSpanLogger(ctx, nil)
		klog.FromContext(ctx).Info("after nil span")
		if out := buf.String(); strings.Contains(out, "trace_id=") || strings.Contains(out, "span_id=") {
			t.Errorf("expected nil span to clear trace_id/span_id, got: %s", out)
		}
		if ContextWithSpanLogger(nil, nil) != nil { //nolint:staticcheck // intentionally verifying nil-ctx safety
			t.Errorf("expected nil context to return nil")
		}
	})

	t.Run("record-only span does not attach IDs", func(t *testing.T) {
		var buf bytes.Buffer
		baseCtx := klog.NewContext(context.Background(), textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(&buf))))
		recordOnlyTP := trace.NewTracerProvider(trace.WithSampler(recordOnlySampler{}))
		ctx, span := recordOnlyTP.Tracer(instrumentationScope).Start(baseCtx, "record-only")
		defer span.End()
		if !span.IsRecording() || span.SpanContext().IsSampled() {
			t.Fatalf("expected recording=true sampled=false, got recording=%v sampled=%v", span.IsRecording(), span.SpanContext().IsSampled())
		}
		ctx = ContextWithSpanLogger(ctx, span)
		klog.FromContext(ctx).Info("record only")
		if out := buf.String(); strings.Contains(out, "trace_id=") || strings.Contains(out, "span_id=") {
			t.Errorf("expected record-only span not to log trace_id/span_id, got: %s", out)
		}
	})

	t.Run("discard logger with nil sink does not panic", func(t *testing.T) {
		discardCtx := klog.NewContext(context.Background(), logr.Discard())
		ctx, span := sampledTP.Tracer(instrumentationScope).Start(discardCtx, "discard")
		defer span.End()
		ctx = ContextWithSpanLogger(ctx, span)
		klog.FromContext(ctx).Info("discarded")
	})

	t.Run("vmodule and WithCallDepth/WithCallStackHelper preserve caller attribution", func(t *testing.T) {
		var buf bytes.Buffer
		cfg := textlogger.NewConfig(textlogger.Output(&buf))
		if err := cfg.VModule().Set("tracing_test=5"); err != nil {
			t.Fatalf("failed to set vmodule: %v", err)
		}
		baseCtx := klog.NewContext(context.Background(), textlogger.NewLogger(cfg))
		ctx, span := sampledTP.Tracer(instrumentationScope).Start(baseCtx, "vmodule")
		defer span.End()
		ctx = ContextWithSpanLogger(ctx, span)

		logViaHelper := func(l klog.Logger, msg string) {
			helper, l := l.WithCallStackHelper()
			helper()
			l.V(5).Info(msg)
		}
		_, _, callLine, _ := runtime.Caller(0)
		logViaHelper(klog.FromContext(ctx), "vmodule message")
		callLine++

		out := buf.String()
		wantCaller := fmt.Sprintf("tracing_test.go:%d", callLine)
		if !strings.Contains(out, "vmodule message") || !strings.Contains(out, wantCaller) {
			t.Errorf("expected vmodule message with caller %q, got:\n%s", wantCaller, out)
		}
		if !strings.Contains(out, fmt.Sprintf("trace_id=%q", span.SpanContext().TraceID().String())) {
			t.Errorf("expected trace_id in vmodule output, got:\n%s", out)
		}
	})

	t.Run("slog round-trip does not race or corrupt caller depth", func(t *testing.T) {
		for _, format := range []string{"text", "json"} {
			t.Run(format, func(t *testing.T) {
				var mu sync.Mutex
				var buf bytes.Buffer
				syncWriter := &lockedBuffer{mu: &mu, buf: &buf}
				var baseLogger klog.Logger
				if format == "text" {
					baseLogger = textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(syncWriter)))
				} else {
					baseLogger, _ = logsjson.NewJSONLogger(0, zapcore.AddSync(syncWriter), nil, nil)
				}
				baseCtx := klog.NewContext(context.Background(), baseLogger)
				parentCtx, parentSpan := sampledTP.Tracer(instrumentationScope).Start(baseCtx, "parent")
				defer parentSpan.End()
				parentCtx = ContextWithSpanLogger(parentCtx, parentSpan)
				childCtx, childSpan := Start(parentCtx, "child")
				defer childSpan.End(10 * time.Second)

				var wg sync.WaitGroup
				wg.Add(2)
				go func() {
					defer wg.Done()
					for i := 0; i < 20; i++ {
						klog.FromContext(parentCtx).Info("parent log", "i", i)
					}
				}()
				go func() {
					defer wg.Done()
					for i := 0; i < 20; i++ {
						roundTripped := logr.FromSlogHandler(logr.ToSlogHandler(klog.FromContext(childCtx)))
						roundTripped.Info("round-tripped log", "i", i)
					}
				}()
				wg.Wait()

				_, _, checkLine, _ := runtime.Caller(0)
				klog.FromContext(parentCtx).Info("post-roundtrip check")
				checkLine++

				mu.Lock()
				out := buf.String()
				mu.Unlock()
				wantCaller := fmt.Sprintf("tracing_test.go:%d", checkLine)
				if !strings.Contains(out, wantCaller) {
					t.Errorf("expected caller %q after slog round-trip, got:\n%s", wantCaller, out)
				}
			})
		}
	})
}

type lockedBuffer struct {
	mu  *sync.Mutex
	buf *bytes.Buffer
}

func (l *lockedBuffer) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

func BenchmarkTracingContextualLogging(b *testing.B) {
	noopTP := NewNoopTracerProvider()
	unsampledTP := trace.NewTracerProvider(trace.WithSampler(trace.NeverSample()))
	sampledTP := trace.NewTracerProvider(trace.WithSampler(trace.AlwaysSample()))

	for _, tc := range []struct {
		name string
		tp   oteltrace.TracerProvider
		log  bool
	}{
		{name: "noop_no_log", tp: noopTP, log: false},
		{name: "unsampled_no_log", tp: unsampledTP, log: false},
		{name: "sampled_no_log", tp: sampledTP, log: false},
		{name: "noop_with_log", tp: noopTP, log: true},
		{name: "unsampled_with_log", tp: unsampledTP, log: true},
		{name: "sampled_with_log", tp: sampledTP, log: true},
	} {
		b.Run(tc.name, func(b *testing.B) {
			cfg := textlogger.NewConfig(textlogger.Output(io.Discard))
			logger := textlogger.NewLogger(cfg)
			baseCtx := klog.NewContext(context.Background(), logger)
			tracer := tc.tp.Tracer(instrumentationScope)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ctx, rootSpan := tracer.Start(baseCtx, "root")
				ctx = ContextWithSpanLogger(ctx, rootSpan)
				childCtx, childSpan := Start(ctx, "child")
				if tc.log {
					klog.FromContext(childCtx).Info("benchmark message", "iteration", i)
				} else {
					klog.FromContext(childCtx).V(5).Info("debug message", "iteration", i)
				}
				childSpan.End(time.Minute)
				rootSpan.End()
			}
		})
	}
}
