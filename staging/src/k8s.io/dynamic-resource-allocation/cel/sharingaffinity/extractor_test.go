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

package sharingaffinity

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	sigsjson "sigs.k8s.io/json"
)

const testDriver = "dra.example.com"

func TestExtract_ParametersEnvironment(t *testing.T) {
	for _, tc := range []struct {
		name, expression string
		wantErr          error
	}{
		{"parameters", `parameters.subnetID`, nil},
		{"optional parameters", `parameters.?subnetID.orValue("")`, nil},
		{"guarded parameters", `has(parameters.subnetID) ? parameters.subnetID : ""`, nil},
		{"no object alias", `object.subnetID`, ErrInvalidDefinition},
		{"no device variable", `device.attributes["dra.example.com"].model`, ErrInvalidDefinition},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", tc.expression)})
			got, err := newTestExtractor().Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{
				opaqueConfig(nil, `{"subnetID":"blue","private":{"token":"not-a-selector"}}`),
			})
			if tc.wantErr != nil {
				assertExtractError(t, got, err, tc.wantErr, OutcomeInvalidDefinition)
				return
			}
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}})
		})
	}
}

func TestExtract_Outcomes(t *testing.T) {
	value := testExtractor("subnet", `parameters.?subnetID.orValue("")`)
	empty := testExtractor("pkey", `""`)
	exact := value
	exact.APIGroup, exact.Kind = "example.com", "Config"
	blue := opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","subnetID":"blue"}`)
	green := opaqueConfig(nil, `{"apiVersion":"example.com/v2","kind":"Config","subnetID":"green"}`)
	for _, tc := range []struct {
		name       string
		extractors []resourceapi.SharingAffinityExtractor
		configs    []resourceapi.DeviceClaimConfiguration
		want       Result
		wantErr    error
		outcome    Outcome
	}{
		{name: "empty set", want: Result{Satisfied: true}},
		{name: "empty set skips malformed parameters", configs: []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{`)}, want: Result{Satisfied: true}},
		{name: "no configs", extractors: []resourceapi.SharingAffinityExtractor{value}, want: Result{Constrained: true, Unsatisfied: []string{"subnet"}}},
		{name: "no metadata matches", extractors: []resourceapi.SharingAffinityExtractor{exact}, configs: []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{}`)}, want: Result{Constrained: true, Unsatisfied: []string{"subnet"}}},
		{name: "all empty", extractors: []resourceapi.SharingAffinityExtractor{empty}, configs: []resourceapi.DeviceClaimConfiguration{blue}, want: Result{Constrained: true, Unsatisfied: []string{"pkey"}}},
		{name: "partial satisfaction", extractors: []resourceapi.SharingAffinityExtractor{value, empty}, configs: []resourceapi.DeviceClaimConfiguration{blue}, want: Result{Constrained: true, Values: map[string]string{"subnet": "blue"}, Unsatisfied: []string{"pkey"}}},
		{name: "full satisfaction", extractors: []resourceapi.SharingAffinityExtractor{value, testExtractor("pkey", `"p0"`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, want: Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue", "pkey": "p0"}}},
		{name: "equal contributions", extractors: []resourceapi.SharingAffinityExtractor{value}, configs: []resourceapi.DeviceClaimConfiguration{blue, blue}, want: Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}}},
		{name: "empty then value", extractors: []resourceapi.SharingAffinityExtractor{value}, configs: []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{}`), blue}, want: Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}}},
		{name: "value then empty", extractors: []resourceapi.SharingAffinityExtractor{value}, configs: []resourceapi.DeviceClaimConfiguration{blue, opaqueConfig(nil, `{}`)}, want: Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}}},
		{name: "conflicting contributions", extractors: []resourceapi.SharingAffinityExtractor{value}, configs: []resourceapi.DeviceClaimConfiguration{blue, green}, wantErr: ErrClaimSelfInconsistent, outcome: OutcomeClaimInconsistent},
		{name: "duplicate equal expressions", extractors: []resourceapi.SharingAffinityExtractor{value, value}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrInvalidDefinition, outcome: OutcomeInvalidDefinition},
		{name: "duplicate equal outputs", extractors: []resourceapi.SharingAffinityExtractor{value, testExtractor("subnet", `"blue"`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrInvalidDefinition, outcome: OutcomeInvalidDefinition},
		{name: "duplicate unequal outputs", extractors: []resourceapi.SharingAffinityExtractor{value, testExtractor("subnet", `"green"`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrInvalidDefinition, outcome: OutcomeInvalidDefinition},
		{name: "duplicate without configs", extractors: []resourceapi.SharingAffinityExtractor{value, value}, wantErr: ErrInvalidDefinition, outcome: OutcomeInvalidDefinition},
		{name: "invalid CEL without configs", extractors: []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `parameters.[`)}, wantErr: ErrInvalidDefinition, outcome: OutcomeInvalidDefinition},
		{name: "empty expression", extractors: []resourceapi.SharingAffinityExtractor{testExtractor("subnet", ``)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrInvalidDefinition, outcome: OutcomeInvalidDefinition},
		{name: "static non-string", extractors: []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `42`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrInvalidDefinition, outcome: OutcomeInvalidDefinition},
		{name: "missing field", extractors: []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `parameters.missing`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrMissingField, outcome: OutcomeRuntime},
		{name: "dynamic non-string", extractors: []resourceapi.SharingAffinityExtractor{value}, configs: []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{"subnetID":42}`)}, wantErr: ErrNonStringReturn, outcome: OutcomeRuntime},
		{name: "runtime evaluation failure", extractors: []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `string(1 / parameters.zero)`)}, configs: []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{"zero":0}`)}, wantErr: ErrCELEval, outcome: OutcomeRuntime},
		{name: "later failure discards previous values", extractors: []resourceapi.SharingAffinityExtractor{value, testExtractor("pkey", `parameters.missing`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrMissingField, outcome: OutcomeRuntime},
		{name: "unsatisfied does not hide later runtime error", extractors: []resourceapi.SharingAffinityExtractor{empty, testExtractor("subnet", `parameters.missing`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrMissingField, outcome: OutcomeRuntime},
		{name: "unsatisfied names sorted", extractors: []resourceapi.SharingAffinityExtractor{testExtractor("z-last", `""`), testExtractor("a-first", `""`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, want: Result{Constrained: true, Unsatisfied: []string{"a-first", "z-last"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := newTestExtractor().Extract(testContext(t), testPool(1, tc.extractors), Request{Name: "req"}, tc.configs)
			if tc.wantErr != nil {
				assertExtractError(t, got, err, tc.wantErr, tc.outcome)
				return
			}
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			assertExtractResult(t, got, tc.want)
		})
	}
}

func TestExtract_DefinitionBeforeMatching(t *testing.T) {
	for _, expression := range []string{`parameters.[`, `object.x`, `device.x`, `42`, ``} {
		t.Run(expression, func(t *testing.T) {
			entry := testExtractor("subnet", expression)
			entry.APIGroup, entry.Kind = "example.com", "Config"
			for _, raw := range []string{`{"apiVersion":"other.example.com/v1","kind":"Other"}`, `{`} {
				got, err := newTestExtractor().Extract(testContext(t), testPool(1, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, raw)})
				assertExtractError(t, got, err, ErrInvalidDefinition, OutcomeInvalidDefinition)
			}
		})
	}
}

func TestExtract_NineCombinedExtractors(t *testing.T) {
	slices := make([]resourceapi.ResourceSlice, 2)
	want := map[string]string{}
	for i := range 9 {
		name := fmt.Sprintf("key-%d", i)
		slices[i/8].Spec.SharingAffinity = append(slices[i/8].Spec.SharingAffinity, testExtractor(name, `"value"`))
		want[name] = "value"
	}
	combined := append(slices[0].Spec.SharingAffinity, slices[1].Spec.SharingAffinity...)
	if err := ValidateExtractorSet(combined); err != nil {
		t.Fatalf("ValidateExtractorSet(8 + 1 entries) error = %v; eight is a per-slice cap", err)
	}
	got, err := newTestExtractor().Extract(testContext(t), testPool(1, combined), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{}`)})
	if err != nil {
		t.Fatalf("Extract(9 combined entries) error = %v", err)
	}
	assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: want})
}

func TestExtract_Scoping(t *testing.T) {
	for _, tc := range []struct {
		name     string
		req      Request
		requests []string
		driver   string
		opaque   bool
		matches  bool
	}{
		{"whole claim", Request{Name: "main"}, nil, testDriver, true, true},
		{"empty driver", Request{Name: "main"}, nil, "", true, true},
		{"top level", Request{Name: "main"}, []string{"main"}, testDriver, true, true},
		{"unrelated top level", Request{Name: "main"}, []string{"other"}, testDriver, true, false},
		{"subrequest whole claim", Request{Name: "slice", ParentName: "main"}, nil, testDriver, true, true},
		{"subrequest parent", Request{Name: "slice", ParentName: "main"}, []string{"main"}, testDriver, true, true},
		{"qualified subrequest", Request{Name: "slice", ParentName: "main"}, []string{"main/slice"}, testDriver, true, true},
		{"multiple targets", Request{Name: "slice", ParentName: "main"}, []string{"other", "main/slice"}, testDriver, true, true},
		{"bare subrequest not qualified", Request{Name: "slice", ParentName: "main"}, []string{"slice"}, testDriver, true, false},
		{"sibling subrequest", Request{Name: "slice", ParentName: "main"}, []string{"main/other"}, testDriver, true, false},
		{"other parent", Request{Name: "slice", ParentName: "main"}, []string{"other/slice"}, testDriver, true, false},
		{"top level excludes subrequest target", Request{Name: "main"}, []string{"main/slice"}, testDriver, true, false},
		{"other driver", Request{Name: "main"}, nil, "other.example.com", true, false},
		{"non opaque", Request{Name: "main"}, nil, testDriver, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := opaqueConfig(tc.requests, `{"subnetID":"blue"}`)
			cfg.Opaque.Driver = tc.driver
			if !tc.opaque {
				cfg.Opaque = nil
			}
			pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `parameters.subnetID`)})
			got, err := newTestExtractor().Extract(testContext(t), pool, tc.req, []resourceapi.DeviceClaimConfiguration{cfg})
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			want := Result{Constrained: true, Unsatisfied: []string{"subnet"}}
			if tc.matches {
				want = Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}}
			}
			assertExtractResult(t, got, want)
		})
	}
}

func TestExtract_MatchedButMissingFields(t *testing.T) {
	for _, tc := range []struct {
		name, expression, fields string
		wantErr                  error
		wantValue                string
	}{
		{"optional absence", `parameters.?subnetID.orValue("")`, ``, nil, ""},
		{"guarded absence", `has(parameters.subnetID) ? parameters.subnetID : ""`, ``, nil, ""},
		{"unguarded absence", `parameters.subnetID`, ``, ErrMissingField, ""},
		{"renamed field takes precedence", `has(parameters.subnet) ? parameters.subnet : (has(parameters.subnetID) ? parameters.subnetID : "")`, `,"subnet":"blue","subnetID":"old"`, nil, "blue"},
		{"renamed field fallback", `has(parameters.subnet) ? parameters.subnet : (has(parameters.subnetID) ? parameters.subnetID : "")`, `,"subnetID":"blue"`, nil, "blue"},
		{"optional present integer", `parameters.?subnetID.orValue("")`, `,"subnetID":7`, ErrNonStringReturn, ""},
		{"present boolean", `parameters.subnetID`, `,"subnetID":true`, ErrNonStringReturn, ""},
		{"present null", `parameters.subnetID`, `,"subnetID":null`, ErrNonStringReturn, ""},
		{"present object", `parameters.subnetID`, `,"subnetID":{}`, ErrNonStringReturn, ""},
		{"present list", `parameters.subnetID`, `,"subnetID":[]`, ErrNonStringReturn, ""},
		{"wrong type method", `parameters.subnetID.lowerAscii()`, `,"subnetID":7`, ErrCELEval, ""},
		{"private nested field available", `parameters.private.subnetID`, `,"private":{"subnetID":"blue"}`, nil, "blue"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			entry := testExtractor("subnet", tc.expression)
			entry.APIGroup, entry.Kind = "example.com", "Config"
			raw := `{"apiVersion":"example.com/v2","kind":"Config"` + tc.fields + `}`
			got, err := newTestExtractor().Extract(testContext(t), testPool(1, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, raw)})
			if tc.wantErr != nil {
				assertExtractError(t, got, err, tc.wantErr, OutcomeRuntime)
				return
			}
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			want := Result{Constrained: true, Unsatisfied: []string{"subnet"}}
			if tc.wantValue != "" {
				want = Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": tc.wantValue}}
			}
			assertExtractResult(t, got, want)
		})
	}
}

func TestExtract_WildcardKindGuard(t *testing.T) {
	entry := testExtractor("subnet", `has(parameters.kind) && (parameters.kind == "NICConfig" || parameters.kind == "NICConfig2") ? parameters.subnetID : ""`)
	entry.APIGroup = "example.com"
	for _, tc := range []struct {
		name, raw string
		satisfied bool
	}{
		{"first kind", `{"apiVersion":"example.com/v1","kind":"NICConfig","subnetID":"blue"}`, true},
		{"second kind", `{"apiVersion":"example.com/v2","kind":"NICConfig2","subnetID":"blue"}`, true},
		{"missing kind", `{"apiVersion":"example.com/v1"}`, false},
		{"unrelated kind", `{"apiVersion":"example.com/v1","kind":"Other"}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := newTestExtractor().Extract(testContext(t), testPool(1, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, tc.raw)})
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			want := Result{Constrained: true, Unsatisfied: []string{"subnet"}}
			if tc.satisfied {
				want = Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}}
			}
			assertExtractResult(t, got, want)
		})
	}
}

func TestExtract_MultiVersionGuard(t *testing.T) {
	entry := testExtractor("subnet", `parameters.apiVersion == "example.com/v1" ? parameters.subnetID : ""`)
	entry.APIGroup, entry.Kind = "example.com", "Config"
	got, err := newTestExtractor().Extract(testContext(t), testPool(1, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1beta1","kind":"Config","subnetID":"old"}`),
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","subnetID":"blue"}`),
	})
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}
	assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}})
}

func TestExtract_LargeIntegers(t *testing.T) {
	for _, id := range []string{"9007199254740993", "9223372036854775807", "-9223372036854775808"} {
		t.Run(id, func(t *testing.T) {
			pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("id", `string(parameters.id)`)})
			got, err := newTestExtractor().Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{"id":`+id+`}`)})
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"id": id}})
		})
	}
}

func TestExtract_MalformedParameters(t *testing.T) {
	for _, tc := range []struct{ name, raw string }{
		{"empty", ""}, {"whitespace", " \t\n"}, {"null", "null"}, {"array", "[]"},
		{"string", `"s3cr3t-marker"`}, {"number", "1"}, {"boolean", "true"},
		{"truncated", `{"secret":"s3cr3t-marker"`}, {"trailing data", `{} {}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, scope := range []string{"in scope", "other driver", "other request"} {
				t.Run(scope, func(t *testing.T) {
					cfg := opaqueConfig(nil, tc.raw)
					if scope == "other driver" {
						cfg.Opaque.Driver = "other.example.com"
					}
					if scope == "other request" {
						cfg.Requests = []string{"other"}
					}
					pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `"blue"`)})
					got, err := newTestExtractor().Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{cfg})
					if scope == "in scope" {
						assertExtractError(t, got, err, ErrParameters, OutcomeRuntime)
						if strings.Contains(err.Error(), "s3cr3t-marker") {
							t.Error("error disclosed opaque parameters")
						}
						return
					}
					if err != nil {
						t.Fatalf("out-of-scope parameters were decoded: %v", err)
					}
					assertExtractResult(t, got, Result{Constrained: true, Unsatisfied: []string{"subnet"}})
				})
			}
		})
	}
}

func TestExtract_ErrorRedaction(t *testing.T) {
	// Integer-preserving JSON falls back to float64 beyond int64, so the
	// token must overflow float64 to exercise a decoder error.
	overflow := strings.Repeat("1234567890", 40)
	for _, tc := range []struct {
		name, expression, raw, secret  string
		wantErr                        error
		jsonTypeError, jsonSyntaxError bool
	}{
		{"numeric overflow", `"blue"`, `{"n":` + overflow + `}`, overflow, ErrParameters, true, false},
		{"dynamic missing key", `parameters[parameters.secret]`, `{"secret":"s3cr3t-marker"}`, "s3cr3t-marker", ErrMissingField, false, false},
		{"JSON syntax", `"blue"`, `{"secret":"s3cr3t-marker","broken":}`, "s3cr3t-marker", ErrParameters, false, true},
		{"dynamic conversion", `string(int(parameters.secret))`, `{"secret":"s3cr3t-marker"}`, "s3cr3t-marker", ErrCELEval, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true), ktesting.Verbosity(10)))
			ctx := klog.NewContext(context.Background(), logger)
			pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", tc.expression)})
			got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, tc.raw)})
			assertExtractError(t, got, err, tc.wantErr, OutcomeRuntime)
			if strings.Contains(err.Error(), tc.secret) || strings.Contains(err.Error(), tc.raw) {
				t.Error("error disclosed opaque parameters")
			}
			buffer := logger.GetSink().(ktesting.Underlier).GetBuffer()
			logs := buffer.String()
			if logs == "" {
				t.Error("expected captured extraction logs at verbosity 10")
			}
			if strings.Contains(logs, tc.secret) || strings.Contains(logs, tc.raw) {
				t.Error("logs disclosed opaque parameters")
			}
			if tc.jsonTypeError {
				var cause *json.UnmarshalTypeError
				if !errors.As(err, &cause) {
					t.Fatalf("error lost JSON type-error cause: %v", err)
				}
				if !strings.Contains(cause.Value, tc.secret) {
					t.Error("redaction discarded the original numeric decoder cause")
				}
			}
			if tc.jsonSyntaxError {
				found := false
				for cause := err; cause != nil; cause = errors.Unwrap(cause) {
					if syntax, offset := sigsjson.SyntaxErrorOffset(cause); syntax && offset > 0 {
						found = true
					}
				}
				if !found {
					t.Fatalf("error lost JSON syntax-error cause: %v", err)
				}
			}
			if tc.wantErr == ErrMissingField {
				found := false
				for cause := errors.Unwrap(err); cause != nil; cause = errors.Unwrap(cause) {
					if strings.Contains(cause.Error(), tc.secret) {
						found = true
					}
				}
				if !found {
					t.Error("redaction discarded the underlying CEL missing-key error")
				}
			}
		})
	}
}

func TestExtract_LoggingRedaction(t *testing.T) {
	logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true), ktesting.Verbosity(10)))
	ctx := klog.NewContext(context.Background(), logger)
	entry := testExtractor("subnet", `parameters.subnetID`)
	raw := `{"subnetID":"blue","privateToken":"s3cr3t-marker"}`
	got, err := newTestExtractor().Extract(ctx, testPool(7, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, raw)})
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}
	assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}})
	logs := logger.GetSink().(ktesting.Underlier).GetBuffer().String()
	if logs == "" {
		t.Fatal("expected captured extraction logs at verbosity 10")
	}
	for _, forbidden := range []string{raw, "s3cr3t-marker", entry.Expression} {
		if strings.Contains(logs, forbidden) {
			t.Errorf("logs contain private payload or expression %q", forbidden)
		}
	}
}

func TestExtract_Cancellation(t *testing.T) {
	entry := testExtractor("subnet", `parameters.subnetID`)
	entry.APIGroup, entry.Kind = "example.com", "Config"
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{entry})
	matching := opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","subnetID":"blue"}`)
	for _, tc := range []struct {
		name    string
		configs []resourceapi.DeviceClaimConfiguration
	}{
		{"matching", []resourceapi.DeviceClaimConfiguration{matching}},
		{"zero configs", nil},
		{"out of scope", []resourceapi.DeviceClaimConfiguration{opaqueConfig([]string{"other"}, `{`)}},
		{"metadata nonmatch", []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{"kind":"Other"}`)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cause := errors.New("test cancellation cause")
			ctx, cancel := context.WithCancelCause(testContext(t))
			cancel(cause)
			got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, tc.configs)
			assertExtractError(t, got, err, context.Canceled, OutcomeRuntime)
			if !errors.Is(err, cause) {
				t.Fatalf("Extract() error = %v, want cancellation cause", err)
			}
		})
	}
	t.Run("deadline and cause", func(t *testing.T) {
		cause := errors.New("test deadline cause")
		ctx, cancel := context.WithDeadlineCause(testContext(t), time.Now().Add(-time.Second), cause)
		defer cancel()
		got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{matching})
		assertExtractError(t, got, err, context.DeadlineExceeded, OutcomeRuntime)
		if !errors.Is(err, cause) {
			t.Fatalf("Extract() error = %v, want deadline cause", err)
		}
	})
	t.Run("empty extractor fast path", func(t *testing.T) {
		ctx, cancel := context.WithCancel(testContext(t))
		cancel()
		got, err := newTestExtractor().Extract(ctx, testPool(1, nil), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{`)})
		if err != nil {
			t.Fatalf("empty extractor set consulted cancellation: %v", err)
		}
		assertExtractResult(t, got, Result{Satisfied: true})
	})
}

func TestExtract_CancellationDuringEvaluation(t *testing.T) {
	// CEL obtains Done inside ContextEval, after Extract's pre-evaluation
	// checks. Cancel there to exercise the real interpreter interrupt without
	// racing a timer against compilation or exhausting the cost budget.
	cause := errors.New("interpreter cancellation cause")
	ctx, cancel := context.WithCancelCause(testContext(t))
	defer cancel(nil)
	interrupt := &cancelOnEvalContext{Context: ctx, cancel: func() { cancel(cause) }}
	items := strings.TrimSuffix(strings.Repeat(`1,`, 4096), ",")
	entry := testExtractor("subnet", `[`+items+`].all(x, x == 1) ? "blue" : ""`)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{entry})
	if err := ValidateExtractorSet(pool.Extractors); err != nil {
		t.Fatalf("cancellation fixture must be statically valid: %v", err)
	}
	got, err := newTestExtractor().Extract(interrupt, pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{}`)})
	assertExtractError(t, got, err, context.Canceled, OutcomeRuntime)
	if !errors.Is(err, cause) {
		t.Fatalf("Extract() error = %v, want interpreter cancellation cause", err)
	}
	var detail *Error
	if !errors.As(err, &detail) || detail.Extractor != "subnet" || detail.ConfigIndex != 0 {
		t.Fatalf("cancellation lost evaluation attribution: %#v", detail)
	}
}

type cancelOnEvalContext struct {
	context.Context
	once   sync.Once
	cancel func()
}

func (c *cancelOnEvalContext) Done() <-chan struct{} {
	c.once.Do(c.cancel)
	return c.Context.Done()
}

func TestExtract_TypedErrorPropagation(t *testing.T) {
	for _, tc := range []struct {
		name, expression string
		raws             []string
		want             error
		index            int
	}{
		{"missing field", `parameters.subnetID`, []string{`{"subnetID":"blue"}`, `{}`}, ErrMissingField, 1},
		{"non-string", `parameters.subnetID`, []string{`{"subnetID":7}`}, ErrNonStringReturn, 0},
		{"evaluation", `string(1 / parameters.zero)`, []string{`{"zero":0}`}, ErrCELEval, 0},
		{"parameters", `"blue"`, []string{`{}`, `{`}, ErrParameters, 1},
		{"inconsistent", `parameters.subnetID`, []string{`{"subnetID":"blue"}`, `{"subnetID":"green"}`}, ErrClaimSelfInconsistent, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pool := testPool(7, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", tc.expression)})
			configs := []resourceapi.DeviceClaimConfiguration{opaqueConfig([]string{"unrelated"}, `{`)}
			for _, raw := range tc.raws {
				configs = append(configs, opaqueConfig(nil, raw))
			}
			got, err := newTestExtractor().Extract(testContext(t), pool, Request{Name: "slice", ParentName: "main"}, configs)
			outcome := OutcomeRuntime
			if tc.want == ErrClaimSelfInconsistent {
				outcome = OutcomeClaimInconsistent
			}
			assertExtractError(t, got, err, tc.want, outcome)
			var detail *Error
			if !errors.As(err, &detail) {
				t.Fatalf("Extract() error %v does not wrap *Error", err)
			}
			if detail.Pool != pool.Key || detail.Request != "main/slice" || detail.ConfigIndex != tc.index+1 {
				t.Errorf("error attribution = %#v, want pool %#v, request main/slice, original config index %d", detail, pool.Key, tc.index+1)
			}
			if tc.want != ErrParameters && detail.Extractor != "subnet" {
				t.Errorf("extractor attribution = %q, want subnet", detail.Extractor)
			}
		})
	}
}

func TestError_WrappingAndAttribution(t *testing.T) {
	for _, kind := range []error{ErrClaimSelfInconsistent, ErrInvalidDefinition, ErrParameters, ErrCELEval, ErrNonStringReturn, ErrCostExceeded, ErrMissingField} {
		t.Run(kind.Error(), func(t *testing.T) {
			cause := errors.New("underlying cause")
			detail := &Error{Kind: kind, Pool: PoolKey{Driver: testDriver, Name: "pool-a", Generation: 7}, Request: "main/slice", Extractor: "subnet", ConfigIndex: 2, Err: cause}
			err := fmt.Errorf("outer wrapper: %w", detail)
			if !errors.Is(err, kind) || !errors.Is(err, cause) {
				t.Errorf("wrapped error does not preserve kind and cause: %v", err)
			}
			var got *Error
			if !errors.As(err, &got) || got != detail || errors.Unwrap(detail) != cause {
				t.Error("typed error or direct cause was lost")
			}
			for _, text := range []string{testDriver, "pool-a", "7", "main/slice", "subnet", "2"} {
				if !strings.Contains(err.Error(), text) {
					t.Errorf("error %q lacks attribution %q", err, text)
				}
			}
		})
	}
}

func TestClassify_Outcomes(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want Outcome
	}{
		{"claim inconsistency", ErrClaimSelfInconsistent, OutcomeClaimInconsistent},
		{"invalid definition", ErrInvalidDefinition, OutcomeInvalidDefinition},
		{"estimated cost is definition error", &Error{Kind: ErrInvalidDefinition, Err: ErrCostExceeded}, OutcomeInvalidDefinition},
		{"definition takes precedence", errors.Join(ErrClaimSelfInconsistent, ErrInvalidDefinition), OutcomeInvalidDefinition},
		{"parameters", ErrParameters, OutcomeRuntime},
		{"evaluation", ErrCELEval, OutcomeRuntime},
		{"missing field", ErrMissingField, OutcomeRuntime},
		{"non-string", ErrNonStringReturn, OutcomeRuntime},
		{"runtime cost", ErrCostExceeded, OutcomeRuntime},
		{"cancellation", context.Canceled, OutcomeRuntime},
		{"deadline", context.DeadlineExceeded, OutcomeRuntime},
		{"unknown", errors.New("unexpected failure"), OutcomeRuntime},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, err := range []error{tc.err, fmt.Errorf("wrapped: %w", tc.err)} {
				if got := Classify(err); got != tc.want {
					t.Errorf("Classify(%v) = %v, want %v", err, got, tc.want)
				}
			}
		})
	}
}

func TestExtract_StoredExpressionLengthCompatibility(t *testing.T) {
	value := strings.Repeat("x", resourceapi.SharingAffinityCELExpressionMaxLength+1)
	entry := testExtractor("subnet", `"`+value+`"`)
	if compiled, err := compileExpression(entry.Expression, environment.StoredExpressions); err != nil || compiled == nil {
		t.Fatalf("stored long expression must compile without admission length check: %v", err)
	}
	got, err := newTestExtractor().Extract(testContext(t), testPool(1, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{}`)})
	if err != nil {
		t.Fatalf("Extract() newly rejected stored expression by length: %v", err)
	}
	assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": value}})
}

func testContext(t *testing.T) context.Context {
	t.Helper()
	_, ctx := ktesting.NewTestContext(t)
	return ctx
}

func newTestExtractor() *Extractor {
	return NewExtractor(32)
}

func testPool(generation int64, extractors []resourceapi.SharingAffinityExtractor) PoolExtractors {
	return PoolExtractors{Key: PoolKey{Driver: testDriver, Name: "pool-a", Generation: generation}, Extractors: extractors}
}

func testExtractor(name, expression string) resourceapi.SharingAffinityExtractor {
	return resourceapi.SharingAffinityExtractor{Name: name, APIGroup: "*", Kind: "*", Expression: expression}
}

func opaqueConfig(requests []string, raw string) resourceapi.DeviceClaimConfiguration {
	return resourceapi.DeviceClaimConfiguration{
		Requests: requests,
		DeviceConfiguration: resourceapi.DeviceConfiguration{
			Opaque: &resourceapi.OpaqueDeviceConfiguration{Driver: testDriver, Parameters: runtime.RawExtension{Raw: []byte(raw)}},
		},
	}
}

func assertExtractResult(t *testing.T, got, want Result) {
	t.Helper()
	if got.Constrained != want.Constrained || got.Satisfied != want.Satisfied || !maps.Equal(got.Values, want.Values) || !slices.Equal(got.Unsatisfied, want.Unsatisfied) {
		t.Errorf("Extract() = %#v, want %#v", got, want)
	}
}

func assertExtractError(t *testing.T, got Result, err, want error, outcome Outcome) {
	t.Helper()
	if !errors.Is(err, want) {
		t.Fatalf("Extract() error = %v, want errors.Is(..., %v)", err, want)
	}
	if !reflect.DeepEqual(got, Result{}) {
		t.Errorf("failed extraction returned nonzero Result: %#v", got)
	}
	if actual := Classify(err); actual != outcome {
		t.Errorf("Classify(%v) = %v, want %v", err, actual, outcome)
	}
}
