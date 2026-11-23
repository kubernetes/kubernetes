/*
Copyright The Kubernetes Authors.

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

package filters

import (
	"bytes"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	oteltrace "go.opentelemetry.io/otel/trace"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/filterlatency"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/httplog"
	tracing "k8s.io/component-base/tracing"
	"k8s.io/klog/v2"
)

func TestWithTracingLogging(t *testing.T) {
	state := klog.CaptureState()
	defer state.Restore()

	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	klog.InitFlags(fs)
	if err := fs.Set("logtostderr", "false"); err != nil {
		t.Fatalf("failed to set logtostderr: %v", err)
	}
	if err := fs.Set("v", "3"); err != nil {
		t.Fatalf("failed to set v=3: %v", err)
	}

	for _, tc := range []struct {
		name        string
		newTP       func(*tracetest.SpanRecorder) oteltrace.TracerProvider
		traceparent string
		sampled     bool
	}{
		{
			name: "sampled request includes trace_id and span_id in contextual logs and httplog",
			newTP: func(recorder *tracetest.SpanRecorder) oteltrace.TracerProvider {
				return sdktrace.NewTracerProvider(
					sdktrace.WithSampler(sdktrace.AlwaysSample()),
					sdktrace.WithSpanProcessor(recorder),
				)
			},
			sampled: true,
		},
		{
			name: "unsampled request omits trace_id and span_id",
			newTP: func(recorder *tracetest.SpanRecorder) oteltrace.TracerProvider {
				return sdktrace.NewTracerProvider(
					sdktrace.WithSampler(sdktrace.NeverSample()),
					sdktrace.WithSpanProcessor(recorder),
				)
			},
			sampled: false,
		},
		{
			name: "noop tracer provider with sampled traceparent omits trace_id and span_id",
			newTP: func(_ *tracetest.SpanRecorder) oteltrace.TracerProvider {
				return tracing.NewNoopTracerProvider()
			},
			traceparent: "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
			sampled:     false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)
			defer klog.SetOutput(&bytes.Buffer{})

			recorder := tracetest.NewSpanRecorder()
			tp := tc.newTP(recorder)

			innerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				klog.FromContext(r.Context()).Info("handling request")
				w.WriteHeader(http.StatusOK)
			})
			handler := httplog.WithLogging(WithTracing(innerHandler, tp), httplog.DefaultStacktracePred)

			req, err := http.NewRequest(http.MethodGet, "http://example.com/api/v1/namespaces", nil)
			if err != nil {
				t.Fatalf("failed to create request: %v", err)
			}
			if tc.traceparent != "" {
				req.Header.Set("traceparent", tc.traceparent)
			}
			handler.ServeHTTP(httptest.NewRecorder(), req)
			klog.Flush()

			output := buf.String()
			lines := strings.Split(strings.TrimSpace(output), "\n")
			if len(lines) != 2 {
				t.Fatalf("expected 2 log lines (contextual + HTTP), got %d:\n%s", len(lines), output)
			}
			if !strings.Contains(lines[0], "handling request") {
				t.Errorf("expected first log line to contain contextual message, got:\n%s", lines[0])
			}
			if !strings.Contains(lines[0], "traces_test.go:") {
				t.Errorf("expected first log line caller to point to traces_test.go, got:\n%s", lines[0])
			}
			if !strings.Contains(lines[1], `"HTTP"`) {
				t.Errorf("expected second log line to contain HTTP log message, got:\n%s", lines[1])
			}

			if tc.sampled {
				ended := recorder.Ended()
				if len(ended) != 1 {
					t.Fatalf("expected 1 ended span, got %d", len(ended))
				}
				spanCtx := ended[0].SpanContext()
				wantTraceID := fmt.Sprintf("trace_id=%q", spanCtx.TraceID().String())
				wantSpanID := fmt.Sprintf("span_id=%q", spanCtx.SpanID().String())

				for _, line := range lines {
					if !strings.Contains(line, wantTraceID) || !strings.Contains(line, wantSpanID) {
						t.Errorf("expected log line to contain %s and %s, got:\n%s", wantTraceID, wantSpanID, line)
					}
				}
			} else if strings.Contains(output, "trace_id=") || strings.Contains(output, "span_id=") {
				t.Errorf("expected unsampled log output not to contain trace_id or span_id, got:\n%s", output)
			}
		})
	}
}

type spanNameSampler struct {
	sampleName string
}

func (s spanNameSampler) ShouldSample(p sdktrace.SamplingParameters) sdktrace.SamplingResult {
	if p.Name == s.sampleName {
		return sdktrace.SamplingResult{Decision: sdktrace.RecordAndSample}
	}
	return sdktrace.SamplingResult{Decision: sdktrace.Drop}
}

func (s spanNameSampler) Description() string {
	return "spanNameSampler"
}

func buildTestFilterChain(inner http.Handler, afterWithTracing func(r *http.Request), tp oteltrace.TracerProvider, reqUser user.Info) http.Handler {
	handler := inner
	for _, name := range []string{"authorization", "priorityandfairness", "impersonation", "audit"} {
		handler = filterlatency.TrackCompleted(handler)
		handler = filterlatency.TrackStarted(handler, tp, name)
	}
	if afterWithTracing != nil {
		next := handler
		handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			afterWithTracing(r)
			next.ServeHTTP(w, r)
		})
	}
	handler = WithTracing(handler, tp)
	handler = filterlatency.TrackCompleted(handler)
	withUser := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if reqUser != nil {
			r = r.WithContext(request.WithUser(r.Context(), reqUser))
		}
		handler.ServeHTTP(w, r)
	})
	return filterlatency.TrackStarted(withUser, tp, "authentication")
}

// recordingLogSink is a minimal non-deduplicating logr.LogSink (matching zapr's
// append-only WithValues behavior) used to verify that nested filter spans emit
// a single trace_id/span_id pair and preserve caller attribution without adding
// new module dependencies to k8s.io/apiserver.
type recordingLogSink struct {
	records   *[]recordedLogEntry
	values    []any
	callDepth int
}

type recordedLogEntry struct {
	msg    string
	caller string
	kvs    []any
}

func (s *recordingLogSink) Init(info logr.RuntimeInfo) {
	s.callDepth += info.CallDepth
}

func (s *recordingLogSink) Enabled(int) bool { return true }

func (s *recordingLogSink) Info(_ int, msg string, keysAndValues ...any) {
	_, file, line, _ := runtime.Caller(s.callDepth + 1)
	kvs := make([]any, 0, len(s.values)+len(keysAndValues))
	kvs = append(kvs, s.values...)
	kvs = append(kvs, keysAndValues...)
	*s.records = append(*s.records, recordedLogEntry{
		msg:    msg,
		caller: fmt.Sprintf("%s:%d", file, line),
		kvs:    kvs,
	})
}

func (s *recordingLogSink) Error(_ error, msg string, keysAndValues ...any) {
	s.Info(0, msg, keysAndValues...)
}

func (s *recordingLogSink) WithValues(keysAndValues ...any) logr.LogSink {
	clone := *s
	clone.values = append(append([]any(nil), s.values...), keysAndValues...)
	return &clone
}

func (s *recordingLogSink) WithName(string) logr.LogSink {
	clone := *s
	return &clone
}

func (s *recordingLogSink) WithCallDepth(depth int) logr.LogSink {
	clone := *s
	clone.callDepth += depth
	return &clone
}

func TestWithTracingFilterChain(t *testing.T) {
	t.Run("ordinary user with sampled auth span and unsampled HTTP span omits stale auth IDs", func(t *testing.T) {
		state := klog.CaptureState()
		defer state.Restore()

		fs := flag.NewFlagSet("test", flag.ContinueOnError)
		klog.InitFlags(fs)
		if err := fs.Set("logtostderr", "false"); err != nil {
			t.Fatalf("failed to set logtostderr: %v", err)
		}
		if err := fs.Set("v", "3"); err != nil {
			t.Fatalf("failed to set v=3: %v", err)
		}

		var buf bytes.Buffer
		klog.SetOutput(&buf)
		defer klog.SetOutput(&bytes.Buffer{})

		recorder := tracetest.NewSpanRecorder()
		tp := sdktrace.NewTracerProvider(
			sdktrace.WithSampler(sdktrace.ParentBased(spanNameSampler{sampleName: "authentication"})),
			sdktrace.WithSpanProcessor(recorder),
		)

		inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			klog.FromContext(r.Context()).Info("handling request")
			w.WriteHeader(http.StatusOK)
		})
		afterWithTracing := func(r *http.Request) {
			klog.FromContext(r.Context()).Info("immediately after WithTracing")
		}
		handler := httplog.WithLogging(
			buildTestFilterChain(inner, afterWithTracing, tp, &user.DefaultInfo{Name: "alice", Groups: []string{user.AllAuthenticated}}),
			httplog.DefaultStacktracePred,
		)

		req, err := http.NewRequest(http.MethodGet, "http://example.com/api/v1/namespaces", nil)
		if err != nil {
			t.Fatalf("failed to create request: %v", err)
		}
		handler.ServeHTTP(httptest.NewRecorder(), req)
		klog.Flush()

		if len(recorder.Ended()) != 1 || recorder.Ended()[0].Name() != "authentication" {
			t.Fatalf("expected only authentication span to be sampled, got %d spans", len(recorder.Ended()))
		}
		output := buf.String()
		for _, want := range []string{"immediately after WithTracing", "handling request", `"HTTP"`} {
			if !strings.Contains(output, want) {
				t.Errorf("expected log output to contain %s, got:\n%s", want, output)
			}
		}
		if strings.Contains(output, "trace_id=") || strings.Contains(output, "span_id=") {
			t.Errorf("expected unsampled HTTP request not to log stale auth trace_id/span_id, got:\n%s", output)
		}
	})

	t.Run("privileged user with sampled auth span and unsampled traceparent omits stale auth IDs", func(t *testing.T) {
		state := klog.CaptureState()
		defer state.Restore()

		fs := flag.NewFlagSet("test", flag.ContinueOnError)
		klog.InitFlags(fs)
		if err := fs.Set("logtostderr", "false"); err != nil {
			t.Fatalf("failed to set logtostderr: %v", err)
		}
		if err := fs.Set("v", "3"); err != nil {
			t.Fatalf("failed to set v=3: %v", err)
		}

		var buf bytes.Buffer
		klog.SetOutput(&buf)
		defer klog.SetOutput(&bytes.Buffer{})

		recorder := tracetest.NewSpanRecorder()
		tp := sdktrace.NewTracerProvider(
			sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.AlwaysSample())),
			sdktrace.WithSpanProcessor(recorder),
		)

		inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			klog.FromContext(r.Context()).Info("handling request")
			w.WriteHeader(http.StatusOK)
		})
		afterWithTracing := func(r *http.Request) {
			klog.FromContext(r.Context()).Info("immediately after WithTracing")
		}
		handler := httplog.WithLogging(
			buildTestFilterChain(inner, afterWithTracing, tp, &user.DefaultInfo{Name: "admin", Groups: []string{user.SystemPrivilegedGroup}}),
			httplog.DefaultStacktracePred,
		)

		req, err := http.NewRequest(http.MethodGet, "http://example.com/api/v1/namespaces", nil)
		if err != nil {
			t.Fatalf("failed to create request: %v", err)
		}
		req.Header.Set("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00")
		handler.ServeHTTP(httptest.NewRecorder(), req)
		klog.Flush()

		if len(recorder.Ended()) != 1 || recorder.Ended()[0].Name() != "authentication" {
			t.Fatalf("expected only authentication span to be sampled, got %d spans", len(recorder.Ended()))
		}
		output := buf.String()
		for _, want := range []string{"immediately after WithTracing", "handling request", `"HTTP"`} {
			if !strings.Contains(output, want) {
				t.Errorf("expected log output to contain %s, got:\n%s", want, output)
			}
		}
		if strings.Contains(output, "trace_id=") || strings.Contains(output, "span_id=") {
			t.Errorf("expected unsampled traceparent request not to log stale auth trace_id/span_id, got:\n%s", output)
		}
	})

	t.Run("sampled filter chain with non-deduplicating sink emits single trace_id and span_id pair", func(t *testing.T) {
		var records []recordedLogEntry
		logger := logr.New(&recordingLogSink{records: &records})

		recorder := tracetest.NewSpanRecorder()
		tp := sdktrace.NewTracerProvider(
			sdktrace.WithSampler(sdktrace.AlwaysSample()),
			sdktrace.WithSpanProcessor(recorder),
		)

		inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			klog.FromContext(r.Context()).Info("handling request")
			w.WriteHeader(http.StatusOK)
		})
		handler := buildTestFilterChain(inner, nil, tp, &user.DefaultInfo{Name: "alice", Groups: []string{user.AllAuthenticated}})

		req, err := http.NewRequestWithContext(klog.NewContext(t.Context(), logger), http.MethodGet, "http://example.com/api/v1/namespaces", nil)
		if err != nil {
			t.Fatalf("failed to create request: %v", err)
		}
		handler.ServeHTTP(httptest.NewRecorder(), req)

		ended := recorder.Ended()
		if len(ended) != 6 {
			t.Fatalf("expected 6 ended spans (authorization, priorityandfairness, impersonation, audit, HTTP, authentication), got %d", len(ended))
		}
		authzSC := ended[0].SpanContext()
		authSC := ended[len(ended)-1].SpanContext()
		if authzSC.TraceID() == authSC.TraceID() {
			t.Fatalf("expected WithNewRoot HTTP span to start a different trace from authentication")
		}

		if len(records) != 1 {
			t.Fatalf("expected 1 log entry, got %d", len(records))
		}
		entry := records[0]
		var traceIDs, spanIDs []string
		for i := 0; i+1 < len(entry.kvs); i += 2 {
			switch entry.kvs[i] {
			case "trace_id":
				traceIDs = append(traceIDs, fmt.Sprint(entry.kvs[i+1]))
			case "span_id":
				spanIDs = append(spanIDs, fmt.Sprint(entry.kvs[i+1]))
			}
		}
		if len(traceIDs) != 1 || traceIDs[0] != authzSC.TraceID().String() {
			t.Errorf("expected single trace_id=%s, got %v", authzSC.TraceID(), traceIDs)
		}
		if len(spanIDs) != 1 || spanIDs[0] != authzSC.SpanID().String() {
			t.Errorf("expected single span_id=%s, got %v", authzSC.SpanID(), spanIDs)
		}
		if !strings.Contains(entry.caller, "traces_test.go:") {
			t.Errorf("expected caller to point to traces_test.go, got %s", entry.caller)
		}
	})
}
