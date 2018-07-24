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

package templates

import (
	"strings"
	"testing"

	"github.com/spf13/pflag"
)

func Test_flagsUsages_not_deprecated(t *testing.T) {
	flags := pflag.NewFlagSet("deprecatedflagtest", pflag.ContinueOnError)
	flags.String("new-flag", "", "The new flag for the some command")

	usage := flagsUsages(flags)

	if strings.Contains(usage, "DEPRECATED") {
		t.Errorf("template should not contain DEPRECATED text, got '%s'", usage)
	}
}

func Test_flagsUsages_deprecated(t *testing.T) {
	flags := pflag.NewFlagSet("deprecatedflagtest", pflag.ContinueOnError)
	oldFlag := "old-flag"
	useInstead := "Use --new-flag instead"
	flags.String(oldFlag, "", "Old flag for some command")
	flags.MarkDeprecated(oldFlag, useInstead)
	tests := []struct {
		name   string
		hidden bool
	}{
		{
			name:   "hidden",
			hidden: true,
		},
		{
			name:   "visible",
			hidden: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags.Lookup(oldFlag).Hidden = tt.hidden

			usage := flagsUsages(flags)

			if tt.hidden && usage != "" {
				t.Errorf("%v: template for hidden deprecated flag should be empty, got '%s'", tt.name, usage)
			} else if !tt.hidden {
				if !strings.Contains(usage, "DEPRECATED") {
					t.Errorf("%v: template for deprecated flag should contain DEPRECATED text, got '%s'", tt.name, usage)
				}
				if !strings.Contains(usage, useInstead) {
					t.Errorf("%v: template for deprecated flag should contain the provided use instead message, got '%s'", tt.name, usage)
				}
			}
		})
	}
}
