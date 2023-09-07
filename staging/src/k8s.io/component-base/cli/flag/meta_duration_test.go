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

package flag

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/spf13/pflag"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestMetaDuration(t *testing.T) {
	type config struct {
		timeout *metav1.Duration
	}

	tests := []struct {
		desc       string
		argument   string
		variable   *metav1.Duration
		expected   *metav1.Duration
		parseError string
	}{
		{
			desc: "empty",
		},
		{
			desc:       "missing unit in durations",
			argument:   "5",
			parseError: "missing unit in duration",
		},
		{
			desc:     "set 5s as string",
			argument: "5s",
			expected: &metav1.Duration{Duration: 5 * time.Second},
		},
		{
			desc:     "override 1s by 5s",
			variable: &metav1.Duration{Duration: 1 * time.Second},
			argument: "5s",
			expected: &metav1.Duration{Duration: 5 * time.Second},
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			c := &config{
				timeout: test.variable,
			}
			v := NewMetaDuration(c.timeout)
			fs := pflag.NewFlagSet("testMetaDuration", pflag.ContinueOnError)
			fs.Var(&v, "duration", "usage")

			args := []string{}
			if test.argument != "" {
				args = append(args, fmt.Sprintf("--duration=%s", test.argument))
			}

			err := fs.Parse(args)
			if test.parseError != "" {
				if err == nil {
					t.Errorf("expected error %q, got nil", test.parseError)
				} else if !strings.Contains(err.Error(), test.parseError) {
					t.Errorf("expected error %q, got %q", test.parseError, err)
				}
			} else if err != nil {
				t.Errorf("expected nil error, got %v", err)
			}

			if !reflect.DeepEqual(v.m, test.expected) {
				t.Errorf("expected %+v, got %+v", test.expected, v)
			}

		})
	}
}
