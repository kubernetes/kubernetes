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

package validators

import (
	"strings"
	"testing"

	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

func TestParseNumericLimit(t *testing.T) {
	duration := &types.Type{Name: durationType, Kind: types.Alias, Underlying: types.Int64}
	cases := []struct {
		name    string
		typ     *types.Type
		tag     codetags.Tag
		want    any
		wantErr string
	}{{
		name: "valid duration",
		typ:  duration,
		tag:  codetags.Tag{Value: "100ms", ValueType: codetags.ValueTypeString},
		want: int64(100000000),
	}, {
		name:    "int payload on duration",
		typ:     duration,
		tag:     codetags.Tag{Value: "1", ValueType: codetags.ValueTypeInt},
		wantErr: "field is a time.Duration, but payload is of type int",
	}, {
		name:    "duration without unit",
		typ:     duration,
		tag:     codetags.Tag{Value: "1", ValueType: codetags.ValueTypeString},
		wantErr: "missing unit",
	}, {
		name: "valid integer",
		typ:  types.Int64,
		tag:  codetags.Tag{Value: "1", ValueType: codetags.ValueTypeInt},
		want: int64(1),
	}, {
		name:    "string payload on integer",
		typ:     types.Int64,
		tag:     codetags.Tag{Value: "1", ValueType: codetags.ValueTypeString},
		wantErr: "field is an integer, but payload is of type string",
	}}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseNumericLimit(Context{Type: tc.typ}, tc.tag)
			if (tc.wantErr == "") != (err == nil) {
				t.Fatalf("got error %v, want %q", err, tc.wantErr)
			}
			if err != nil && !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("got error %v, want %q", err, tc.wantErr)
			}
			if got != tc.want {
				t.Errorf("got %v (%T), want %v (%T)", got, got, tc.want, tc.want)
			}
		})
	}
}
