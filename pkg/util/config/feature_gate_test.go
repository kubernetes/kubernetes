/*
Copyright 2016 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"strings"
	"testing"

	"github.com/spf13/pflag"
)

func TestFeatureGateFlag(t *testing.T) {
	tests := []struct {
		arg        string
		allAlpha   bool
		parseError error
	}{
		{fmt.Sprintf("--%s=fooBarBaz=maybeidk", flagName), false, fmt.Errorf("unrecognized key: fooBarBaz")},
		{fmt.Sprintf("--%s=", flagName), false, nil},
		{fmt.Sprintf("--%s=allAlpha=false", flagName), false, nil},
		{fmt.Sprintf("--%s=allAlpha=true", flagName), true, nil},
		{fmt.Sprintf("--%s=allAlpha=banana", flagName), false, fmt.Errorf("invalid value of allAlpha")},
	}
	for i, test := range tests {
		fs := pflag.NewFlagSet("testfeaturegateflag", pflag.ContinueOnError)
		f := &featureGate{}
		f.AddFlag(fs)

		err := fs.Parse([]string{test.arg})
		if test.parseError != nil {
			if !strings.Contains(err.Error(), test.parseError.Error()) {
				t.Errorf("%d: Parse() Expected %v, Got %v", i, test.parseError, err)
			}
		} else if err != nil {
			t.Errorf("%d: Parse() Expected nil, Got %v", i, err)
		}
		if alpha := f.AllAlpha(); alpha != test.allAlpha {
			t.Errorf("%d: AlphaEnabled() expected %v, Got %v", i, test.allAlpha, alpha)
		}
	}
}
