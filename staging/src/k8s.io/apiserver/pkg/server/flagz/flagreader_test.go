/*
Copyright 2024 The Kubernetes Authors.

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

package flagz

import (
	"reflect"
	"testing"

	"github.com/spf13/pflag"

	"k8s.io/component-base/cli/flag"
)

func TestConvertNamedFlagSetToFlags(t *testing.T) {
	tests := []struct {
		name     string
		flagSets *flag.NamedFlagSets
		want     map[string]string
	}{
		{
			name: "basic flags",
			flagSets: &flag.NamedFlagSets{
				FlagSets: map[string]*pflag.FlagSet{
					"test": flagSet(t, map[string]flagValue{
						"flag1": {value: "value1", sensitive: false},
						"flag2": {value: "value2", sensitive: false},
					}),
				},
			},
			want: map[string]string{
				"flag1": "value1",
				"flag2": "value2",
			},
		},
		{
			name: "classified flags",
			flagSets: &flag.NamedFlagSets{
				FlagSets: map[string]*pflag.FlagSet{
					"test": flagSet(t, map[string]flagValue{
						"secret1": {value: "value1", sensitive: true},
						"flag2":   {value: "value2", sensitive: false},
					}),
				},
			},
			want: map[string]string{
				"flag2":   "value2",
				"secret1": "CLASSIFIED",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertNamedFlagSetToFlags(tt.flagSets)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ConvertNamedFlagSetToFlags() = %v, want %v", got, tt.want)
			}
		})
	}
}

type flagValue struct {
	value     string
	sensitive bool
}

func flagSet(t *testing.T, flags map[string]flagValue) *pflag.FlagSet {
	fs := pflag.NewFlagSet("test-set", pflag.ContinueOnError)
	for flagName, flagVal := range flags {
		flagValue := ""
		fs.StringVar(&flagValue, flagName, flagVal.value, "test-usage")
		if flagVal.sensitive {
			err := fs.SetAnnotation(flagName, "classified", []string{"true"})
			if err != nil {
				t.Fatalf("unexpected error when setting flag annotation: %v", err)
			}
		}
	}

	return fs
}
