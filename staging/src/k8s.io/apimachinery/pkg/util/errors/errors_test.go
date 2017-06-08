/*
Copyright 2015 The Kubernetes Authors.

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

package errors

import (
	"fmt"
	"reflect"
	"sort"
	"testing"
)

func TestEmptyAggregate(t *testing.T) {
	var slice []error
	var agg Aggregate
	var err error

	agg = NewAggregate(slice)
	if agg != nil {
		t.Errorf("expected nil, got %#v", agg)
	}
	err = NewAggregate(slice)
	if err != nil {
		t.Errorf("expected nil, got %#v", err)
	}

	// This is not normally possible, but pedantry demands I test it.
	agg = aggregate(slice) // empty aggregate
	if s := agg.Error(); s != "" {
		t.Errorf("expected empty string, got %q", s)
	}
	if s := agg.Errors(); len(s) != 0 {
		t.Errorf("expected empty slice, got %#v", s)
	}
	err = agg.(error)
	if s := err.Error(); s != "" {
		t.Errorf("expected empty string, got %q", s)
	}
}

func TestAggregateWithNil(t *testing.T) {
	var slice []error
	slice = []error{nil}
	var agg Aggregate
	var err error

	agg = NewAggregate(slice)
	if agg != nil {
		t.Errorf("expected nil, got %#v", agg)
	}
	err = NewAggregate(slice)
	if err != nil {
		t.Errorf("expected nil, got %#v", err)
	}

	// Append a non-nil error
	slice = append(slice, fmt.Errorf("err"))
	agg = NewAggregate(slice)
	if agg == nil {
		t.Errorf("expected non-nil")
	}
	if s := agg.Error(); s != "err" {
		t.Errorf("expected 'err', got %q", s)
	}
	if s := agg.Errors(); len(s) != 1 {
		t.Errorf("expected one-element slice, got %#v", s)
	}
	if s := agg.Errors()[0].Error(); s != "err" {
		t.Errorf("expected 'err', got %q", s)
	}

	err = agg.(error)
	if err == nil {
		t.Errorf("expected non-nil")
	}
	if s := err.Error(); s != "err" {
		t.Errorf("expected 'err', got %q", s)
	}
}

func TestSingularAggregate(t *testing.T) {
	var slice []error = []error{fmt.Errorf("err")}
	var agg Aggregate
	var err error

	agg = NewAggregate(slice)
	if agg == nil {
		t.Errorf("expected non-nil")
	}
	if s := agg.Error(); s != "err" {
		t.Errorf("expected 'err', got %q", s)
	}
	if s := agg.Errors(); len(s) != 1 {
		t.Errorf("expected one-element slice, got %#v", s)
	}
	if s := agg.Errors()[0].Error(); s != "err" {
		t.Errorf("expected 'err', got %q", s)
	}

	err = agg.(error)
	if err == nil {
		t.Errorf("expected non-nil")
	}
	if s := err.Error(); s != "err" {
		t.Errorf("expected 'err', got %q", s)
	}
}

func TestPluralAggregate(t *testing.T) {
	var slice []error = []error{fmt.Errorf("abc"), fmt.Errorf("123")}
	var agg Aggregate
	var err error

	agg = NewAggregate(slice)
	if agg == nil {
		t.Errorf("expected non-nil")
	}
	if s := agg.Error(); s != "[abc, 123]" {
		t.Errorf("expected '[abc, 123]', got %q", s)
	}
	if s := agg.Errors(); len(s) != 2 {
		t.Errorf("expected two-elements slice, got %#v", s)
	}
	if s := agg.Errors()[0].Error(); s != "abc" {
		t.Errorf("expected '[abc, 123]', got %q", s)
	}

	err = agg.(error)
	if err == nil {
		t.Errorf("expected non-nil")
	}
	if s := err.Error(); s != "[abc, 123]" {
		t.Errorf("expected '[abc, 123]', got %q", s)
	}
}

func TestFilterOut(t *testing.T) {
	testCases := []struct {
		err      error
		filter   []Matcher
		expected error
	}{
		{
			nil,
			[]Matcher{},
			nil,
		},
		{
			aggregate{},
			[]Matcher{},
			nil,
		},
		{
			aggregate{fmt.Errorf("abc")},
			[]Matcher{},
			aggregate{fmt.Errorf("abc")},
		},
		{
			aggregate{fmt.Errorf("abc")},
			[]Matcher{func(err error) bool { return false }},
			aggregate{fmt.Errorf("abc")},
		},
		{
			aggregate{fmt.Errorf("abc")},
			[]Matcher{func(err error) bool { return true }},
			nil,
		},
		{
			aggregate{fmt.Errorf("abc")},
			[]Matcher{func(err error) bool { return false }, func(err error) bool { return false }},
			aggregate{fmt.Errorf("abc")},
		},
		{
			aggregate{fmt.Errorf("abc")},
			[]Matcher{func(err error) bool { return false }, func(err error) bool { return true }},
			nil,
		},
		{
			aggregate{fmt.Errorf("abc"), fmt.Errorf("def"), fmt.Errorf("ghi")},
			[]Matcher{func(err error) bool { return err.Error() == "def" }},
			aggregate{fmt.Errorf("abc"), fmt.Errorf("ghi")},
		},
		{
			aggregate{aggregate{fmt.Errorf("abc")}},
			[]Matcher{},
			aggregate{aggregate{fmt.Errorf("abc")}},
		},
		{
			aggregate{aggregate{fmt.Errorf("abc"), aggregate{fmt.Errorf("def")}}},
			[]Matcher{},
			aggregate{aggregate{fmt.Errorf("abc"), aggregate{fmt.Errorf("def")}}},
		},
		{
			aggregate{aggregate{fmt.Errorf("abc"), aggregate{fmt.Errorf("def")}}},
			[]Matcher{func(err error) bool { return err.Error() == "def" }},
			aggregate{aggregate{fmt.Errorf("abc")}},
		},
	}
	for i, testCase := range testCases {
		err := FilterOut(testCase.err, testCase.filter...)
		if !reflect.DeepEqual(testCase.expected, err) {
			t.Errorf("%d: expected %v, got %v", i, testCase.expected, err)
		}
	}
}

func TestFlatten(t *testing.T) {
	testCases := []struct {
		agg      Aggregate
		expected Aggregate
	}{
		{
			nil,
			nil,
		},
		{
			aggregate{},
			nil,
		},
		{
			aggregate{fmt.Errorf("abc")},
			aggregate{fmt.Errorf("abc")},
		},
		{
			aggregate{fmt.Errorf("abc"), fmt.Errorf("def"), fmt.Errorf("ghi")},
			aggregate{fmt.Errorf("abc"), fmt.Errorf("def"), fmt.Errorf("ghi")},
		},
		{
			aggregate{aggregate{fmt.Errorf("abc")}},
			aggregate{fmt.Errorf("abc")},
		},
		{
			aggregate{aggregate{aggregate{fmt.Errorf("abc")}}},
			aggregate{fmt.Errorf("abc")},
		},
		{
			aggregate{aggregate{fmt.Errorf("abc"), aggregate{fmt.Errorf("def")}}},
			aggregate{fmt.Errorf("abc"), fmt.Errorf("def")},
		},
		{
			aggregate{aggregate{aggregate{fmt.Errorf("abc")}, fmt.Errorf("def"), aggregate{fmt.Errorf("ghi")}}},
			aggregate{fmt.Errorf("abc"), fmt.Errorf("def"), fmt.Errorf("ghi")},
		},
	}
	for i, testCase := range testCases {
		agg := Flatten(testCase.agg)
		if !reflect.DeepEqual(testCase.expected, agg) {
			t.Errorf("%d: expected %v, got %v", i, testCase.expected, agg)
		}
	}
}

func TestCreateAggregateFromMessageCountMap(t *testing.T) {
	testCases := []struct {
		name     string
		mcp      MessageCountMap
		expected Aggregate
	}{
		{
			"input has single instance of one message",
			MessageCountMap{"abc": 1},
			aggregate{fmt.Errorf("abc")},
		},
		{
			"input has multiple messages",
			MessageCountMap{"abc": 2, "ghi": 1},
			aggregate{fmt.Errorf("abc (repeated 2 times)"), fmt.Errorf("ghi")},
		},
	}

	var expected, agg []error
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			if testCase.expected != nil {
				expected = testCase.expected.Errors()
				sort.Slice(expected, func(i, j int) bool { return expected[i].Error() < expected[j].Error() })
			}
			if testCase.mcp != nil {
				agg = CreateAggregateFromMessageCountMap(testCase.mcp).Errors()
				sort.Slice(agg, func(i, j int) bool { return agg[i].Error() < agg[j].Error() })
			}
			if !reflect.DeepEqual(expected, agg) {
				t.Errorf("expected %v, got %v", expected, agg)
			}
		})
	}
}

func TestAggregateGoroutines(t *testing.T) {
	testCases := []struct {
		errs     []error
		expected map[string]bool // can't compare directly to Aggregate due to non-deterministic ordering
	}{
		{
			[]error{},
			nil,
		},
		{
			[]error{nil},
			nil,
		},
		{
			[]error{nil, nil},
			nil,
		},
		{
			[]error{fmt.Errorf("1")},
			map[string]bool{"1": true},
		},
		{
			[]error{fmt.Errorf("1"), nil},
			map[string]bool{"1": true},
		},
		{
			[]error{fmt.Errorf("1"), fmt.Errorf("267")},
			map[string]bool{"1": true, "267": true},
		},
		{
			[]error{fmt.Errorf("1"), nil, fmt.Errorf("1234")},
			map[string]bool{"1": true, "1234": true},
		},
		{
			[]error{nil, fmt.Errorf("1"), nil, fmt.Errorf("1234"), fmt.Errorf("22")},
			map[string]bool{"1": true, "1234": true, "22": true},
		},
	}
	for i, testCase := range testCases {
		funcs := make([]func() error, len(testCase.errs))
		for i := range testCase.errs {
			err := testCase.errs[i]
			funcs[i] = func() error { return err }
		}
		agg := AggregateGoroutines(funcs...)
		if agg == nil {
			if len(testCase.expected) > 0 {
				t.Errorf("%d: expected %v, got nil", i, testCase.expected)
			}
			continue
		}
		if len(agg.Errors()) != len(testCase.expected) {
			t.Errorf("%d: expected %d errors in aggregate, got %v", i, len(testCase.expected), agg)
			continue
		}
		for _, err := range agg.Errors() {
			if !testCase.expected[err.Error()] {
				t.Errorf("%d: expected %v, got aggregate containing %v", i, testCase.expected, err)
			}
		}
	}
}
