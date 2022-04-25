/*
Copyright 2023 The Kubernetes Authors.

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

package runtime_test

import (
	"errors"
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestIsNotRegisteredError(t *testing.T) {
	testcases := []struct {
		name   string
		err    error
		expect bool
	}{
		{name: "nil", err: nil, expect: false},
		{name: "other", err: fmt.Errorf("different error"), expect: false},
		{name: "direct", err: runtime.NewNotRegisteredErrForKind("test", schema.GroupVersionKind{}), expect: true},
		{name: "embedded", err: fmt.Errorf("embedded: %w", runtime.NewNotRegisteredErrForKind("test", schema.GroupVersionKind{})), expect: true},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			if actual := runtime.IsNotRegisteredError(tc.err); actual != tc.expect {
				t.Fatalf("expected %v, got %v", tc.expect, actual)
			}
		})
	}
}

func TestIsMissingKind(t *testing.T) {
	testcases := []struct {
		name   string
		err    error
		expect bool
	}{
		{name: "nil", err: nil, expect: false},
		{name: "other", err: fmt.Errorf("different error"), expect: false},
		{name: "direct", err: runtime.NewMissingKindErr("test"), expect: true},
		{name: "embedded", err: fmt.Errorf("embedded: %w", runtime.NewMissingKindErr("test")), expect: true},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			if actual := runtime.IsMissingKind(tc.err); actual != tc.expect {
				t.Fatalf("expected %v, got %v", tc.expect, actual)
			}
		})
	}
}

func TestIsMissingVersion(t *testing.T) {
	testcases := []struct {
		name   string
		err    error
		expect bool
	}{
		{name: "nil", err: nil, expect: false},
		{name: "other", err: fmt.Errorf("different error"), expect: false},
		{name: "direct", err: runtime.NewMissingVersionErr("test"), expect: true},
		{name: "embedded", err: fmt.Errorf("embedded: %w", runtime.NewMissingVersionErr("test")), expect: true},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			if actual := runtime.IsMissingVersion(tc.err); actual != tc.expect {
				t.Fatalf("expected %v, got %v", tc.expect, actual)
			}
		})
	}
}

func TestIsStrictDecodingError(t *testing.T) {
	testcases := []struct {
		name   string
		err    error
		expect bool
	}{
		{name: "nil", err: nil, expect: false},
		{name: "other", err: fmt.Errorf("different error"), expect: false},
		{name: "direct", err: runtime.NewStrictDecodingError([]error{errors.New("test")}), expect: true},
		{name: "embedded", err: fmt.Errorf("embedded: %w", runtime.NewStrictDecodingError([]error{errors.New("test")})), expect: true},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			if actual := runtime.IsStrictDecodingError(tc.err); actual != tc.expect {
				t.Fatalf("expected %v, got %v", tc.expect, actual)
			}
		})
	}
}

func TestAsStrictDecodingError(t *testing.T) {
	strictDecodingError := runtime.NewStrictDecodingError([]error{errors.New("test")})
	testcases := []struct {
		name   string
		err    error
		expect bool
	}{
		{name: "nil", err: nil, expect: false},
		{name: "other", err: fmt.Errorf("different error"), expect: false},
		{name: "direct", err: strictDecodingError, expect: true},
		{name: "embedded", err: fmt.Errorf("embedded: %w", strictDecodingError), expect: true},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			actualErr, actual := runtime.AsStrictDecodingError(tc.err)
			if actual != tc.expect {
				t.Fatalf("expected %v, got %v", tc.expect, actual)
			}

			if actual && strictDecodingError != actualErr {
				t.Fatalf("expected error: %v, got %v", strictDecodingError, actualErr)
			}
		})
	}
}
