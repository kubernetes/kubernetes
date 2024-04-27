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

package benchmark

import (
	"reflect"
	"testing"
)

func Test_uniqueLVCombos(t *testing.T) {
	type args struct {
		lvs []*labelValues
	}
	tests := []struct {
		name string
		args args
		want []map[string]string
	}{
		{
			name: "empty input",
			args: args{
				lvs: []*labelValues{},
			},
			want: []map[string]string{{}},
		},
		{
			name: "single label, multiple values",
			args: args{
				lvs: []*labelValues{
					{"A", []string{"a1", "a2"}},
				},
			},
			want: []map[string]string{
				{"A": "a1"},
				{"A": "a2"},
			},
		},
		{
			name: "multiple labels, single value each",
			args: args{
				lvs: []*labelValues{
					{"A", []string{"a1"}},
					{"B", []string{"b1"}},
				},
			},
			want: []map[string]string{
				{"A": "a1", "B": "b1"},
			},
		},
		{
			name: "multiple labels, multiple values",
			args: args{
				lvs: []*labelValues{
					{"A", []string{"a1", "a2"}},
					{"B", []string{"b1", "b2"}},
				},
			},
			want: []map[string]string{
				{"A": "a1", "B": "b1"},
				{"A": "a1", "B": "b2"},
				{"A": "a2", "B": "b1"},
				{"A": "a2", "B": "b2"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := uniqueLVCombos(tt.args.lvs); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("uniqueLVCombos() = %v, want %v", got, tt.want)
			}
		})
	}
}
