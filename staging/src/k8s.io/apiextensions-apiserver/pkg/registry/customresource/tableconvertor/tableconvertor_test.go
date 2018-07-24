/*
Copyright 2018 The Kubernetes Authors.

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

package tableconvertor

import (
	"fmt"
	"reflect"
	"testing"
	"time"
)

func Test_cellForJSONValue(t *testing.T) {
	tests := []struct {
		headerType string
		value      interface{}
		want       interface{}
	}{
		{"integer", int64(42), int64(42)},
		{"integer", float64(3.14), int64(3)},
		{"integer", true, nil},
		{"integer", "foo", nil},

		{"number", int64(42), float64(42)},
		{"number", float64(3.14), float64(3.14)},
		{"number", true, nil},
		{"number", "foo", nil},

		{"boolean", int64(42), nil},
		{"boolean", float64(3.14), nil},
		{"boolean", true, true},
		{"boolean", "foo", nil},

		{"string", int64(42), nil},
		{"string", float64(3.14), nil},
		{"string", true, nil},
		{"string", "foo", "foo"},

		{"date", int64(42), nil},
		{"date", float64(3.14), nil},
		{"date", true, nil},
		{"date", time.Now().Add(-time.Hour*12 - 30*time.Minute).UTC().Format(time.RFC3339), "12h"},
		{"date", time.Now().Add(+time.Hour*12 + 30*time.Minute).UTC().Format(time.RFC3339), "<invalid>"},
		{"date", "", "<unknown>"},

		{"unknown", "foo", nil},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%#v of type %s", tt.value, tt.headerType), func(t *testing.T) {
			if got := cellForJSONValue(tt.headerType, tt.value); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("cellForJSONValue() = %#v, want %#v", got, tt.want)
			}
		})
	}
}
