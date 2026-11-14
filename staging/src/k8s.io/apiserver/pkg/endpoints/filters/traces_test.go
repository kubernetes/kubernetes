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
	"strings"
	"testing"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"k8s.io/apiserver/pkg/server/httplog"
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
		name    string
		sampler sdktrace.Sampler
		sampled bool
	}{
		{
			name:    "sampled request includes trace_id and span_id in contextual logs and httplog",
			sampler: sdktrace.AlwaysSample(),
			sampled: true,
		},
		{
			name:    "unsampled request omits trace_id and span_id",
			sampler: sdktrace.NeverSample(),
			sampled: false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)
			defer klog.SetOutput(&bytes.Buffer{})

			recorder := tracetest.NewSpanRecorder()
			tp := sdktrace.NewTracerProvider(
				sdktrace.WithSampler(tc.sampler),
				sdktrace.WithSpanProcessor(recorder),
			)

			innerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				klog.FromContext(r.Context()).Info("handling request")
				w.WriteHeader(http.StatusOK)
			})
			handler := httplog.WithLogging(WithTracing(innerHandler, tp), httplog.DefaultStacktracePred)

			req, err := http.NewRequest(http.MethodGet, "http://example.com/api/v1/namespaces", nil)
			if err != nil {
				t.Fatalf("failed to create request: %v", err)
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
