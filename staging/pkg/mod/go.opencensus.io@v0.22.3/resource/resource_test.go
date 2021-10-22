// Copyright 2018, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package resource

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func TestMerge(t *testing.T) {
	cases := []struct {
		a, b, want *Resource
	}{
		{
			a: &Resource{
				Type:   "t1",
				Labels: map[string]string{"a": "1", "b": "2"},
			},
			b: &Resource{
				Type:   "t2",
				Labels: map[string]string{"a": "1", "b": "3", "c": "4"},
			},
			want: &Resource{
				Type:   "t1",
				Labels: map[string]string{"a": "1", "b": "2", "c": "4"},
			},
		},
		{
			a: nil,
			b: &Resource{
				Type:   "t1",
				Labels: map[string]string{"a": "1"},
			},
			want: &Resource{
				Type:   "t1",
				Labels: map[string]string{"a": "1"},
			},
		},
		{
			a: &Resource{
				Type:   "t1",
				Labels: map[string]string{"a": "1"},
			},
			b: nil,
			want: &Resource{
				Type:   "t1",
				Labels: map[string]string{"a": "1"},
			},
		},
	}
	for i, c := range cases {
		t.Run(fmt.Sprintf("case-%d", i), func(t *testing.T) {
			res := merge(c.a, c.b)
			if !reflect.DeepEqual(res, c.want) {
				t.Fatalf("unwanted result: want %+v, got %+v", c.want, res)
			}
		})
	}
}

func TestDecodeLabels(t *testing.T) {
	cases := []struct {
		encoded    string
		wantLabels map[string]string
		wantFail   bool
	}{
		{
			encoded:    `example.org/test-1="test $ \"" ,  Abc="Def"`,
			wantLabels: map[string]string{"example.org/test-1": "test $ \"", "Abc": "Def"},
		}, {
			encoded:    `single="key"`,
			wantLabels: map[string]string{"single": "key"},
		},
		{encoded: `invalid-char-ü="test"`, wantFail: true},
		{encoded: `invalid-char="ü-test"`, wantFail: true},
		{encoded: `missing="trailing-quote`, wantFail: true},
		{encoded: `missing=leading-quote"`, wantFail: true},
		{encoded: `extra="chars", a`, wantFail: true},
	}
	for i, c := range cases {
		t.Run(fmt.Sprintf("case-%d", i), func(t *testing.T) {
			res, err := DecodeLabels(c.encoded)
			if err != nil && !c.wantFail {
				t.Fatalf("unwanted error: %s", err)
			}
			if c.wantFail && err == nil {
				t.Fatalf("wanted failure but got none, result: %v", res)
			}
			if !reflect.DeepEqual(res, c.wantLabels) {
				t.Fatalf("wanted result %v, got %v", c.wantLabels, res)
			}
		})
	}
}

func TestEncodeLabels(t *testing.T) {
	got := EncodeLabels(map[string]string{
		"example.org/test-1": "test ¥ \"",
		"un":                 "quøted",
		"Abc":                "Def",
	})
	if want := `Abc="Def",example.org/test-1="test ¥ \"",un="quøted"`; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestMultiDetector(t *testing.T) {
	got, err := MultiDetector(
		func(context.Context) (*Resource, error) {
			return &Resource{
				Type:   "t1",
				Labels: map[string]string{"a": "1", "b": "2"},
			}, nil
		},
		func(context.Context) (*Resource, error) {
			return &Resource{
				Type:   "t2",
				Labels: map[string]string{"a": "11", "c": "3"},
			}, nil
		},
	)(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %s", err)
	}
	want := &Resource{
		Type:   "t1",
		Labels: map[string]string{"a": "1", "b": "2", "c": "3"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected resource: want %v, got %v", want, got)
	}

	wantErr := errors.New("err1")
	_, err = MultiDetector(
		func(context.Context) (*Resource, error) {
			return &Resource{
				Type:   "t1",
				Labels: map[string]string{"a": "1", "b": "2"},
			}, nil
		},
		func(context.Context) (*Resource, error) {
			return nil, wantErr
		},
	)(context.Background())
	if err != wantErr {
		t.Fatalf("unexpected error: want %v, got %v", wantErr, err)
	}
}
