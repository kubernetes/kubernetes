/*
Copyright 2021 The Kubernetes Authors.

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

package set

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
)

func Test_updateEnv(t *testing.T) {
	var (
		env1 = v1.EnvVar{
			Name:  "env1",
			Value: "env1",
		}
		env2 = v1.EnvVar{
			Name:  "env2",
			Value: "env2",
		}
		env3 = v1.EnvVar{
			Name:  "env3",
			Value: "env3",
		}
	)

	type args struct {
		existing []v1.EnvVar
		add      []v1.EnvVar
		remove   []string
	}
	tests := []struct {
		name string
		args args
		want []v1.EnvVar
	}{
		{
			name: "case 1: add a new and remove another one",
			args: args{
				existing: []v1.EnvVar{env1},
				add:      []v1.EnvVar{env2},
				remove:   []string{env1.Name},
			},
			want: []v1.EnvVar{env2},
		},
		{
			name: "case 2: in a collection of multiple env, add a new and remove another one",
			args: args{
				existing: []v1.EnvVar{env1, env2},
				add:      []v1.EnvVar{env3},
				remove:   []string{env1.Name},
			},
			want: []v1.EnvVar{env2, env3},
		},
		{
			name: "case 3: items added are deduplicated",
			args: args{
				existing: []v1.EnvVar{env1},
				add:      []v1.EnvVar{env2, env2},
				remove:   []string{env1.Name},
			},
			want: []v1.EnvVar{env2},
		},
		{
			name: "case 4: multi add and single remove",
			args: args{
				existing: []v1.EnvVar{env1},
				add:      []v1.EnvVar{env2, env2, env2, env3},
				remove:   []string{env1.Name},
			},
			want: []v1.EnvVar{env2, env3},
		},
		{
			name: "case 5: add and remove the same env",
			args: args{
				existing: []v1.EnvVar{env1},
				add:      []v1.EnvVar{env2, env1},
				remove:   []string{env1.Name, env1.Name},
			},
			want: []v1.EnvVar{env2},
		},
		{
			name: "case 6: existing duplicate unmodified by unrelated addition",
			args: args{
				existing: []v1.EnvVar{env1, env1},
				add:      []v1.EnvVar{env2},
				remove:   nil,
			},
			want: []v1.EnvVar{env1, env1, env2},
		},
		{
			name: "case 7: existing duplicate removed when added yet again",
			args: args{
				existing: []v1.EnvVar{env1, env1, env2},
				add:      []v1.EnvVar{env1},
				remove:   nil,
			},
			want: []v1.EnvVar{env1, env2},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := updateEnv(tt.args.existing, tt.args.add, tt.args.remove); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("updateEnv() = %v, want %v", got, tt.want)
			}
		})
	}
}
