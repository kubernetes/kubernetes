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

package register

import (
	"bytes"
	"testing"

	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-base/logs"
	"k8s.io/klog/v2"
)

func TestJSONFlag(t *testing.T) {
	o := logs.NewOptions()
	fs := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	output := bytes.Buffer{}
	o.AddFlags(fs)
	fs.SetOutput(&output)
	fs.PrintDefaults()
	wantSubstring := `Permitted formats: "json", "text".`
	if !assert.Contains(t, output.String(), wantSubstring) {
		t.Errorf("JSON logging format flag is not available. expect to contain %q, got %q", wantSubstring, output.String())
	}
}

func TestJSONFormatRegister(t *testing.T) {
	newOptions := logs.NewOptions()
	klogr := klog.Background()
	testcases := []struct {
		name              string
		args              []string
		contextualLogging bool
		want              *logs.Options
		errs              field.ErrorList
	}{
		{
			name: "JSON log format",
			args: []string{"--logging-format=json"},
			want: func() *logs.Options {
				c := newOptions.Config.DeepCopy()
				c.Format = logs.JSONLogFormat
				return &logs.Options{*c}
			}(),
		},
		{
			name:              "JSON direct",
			args:              []string{"--logging-format=json"},
			contextualLogging: true,
			want: func() *logs.Options {
				c := newOptions.Config.DeepCopy()
				c.Format = logs.JSONLogFormat
				return &logs.Options{*c}
			}(),
		},
		{
			name: "Unsupported log format",
			args: []string{"--logging-format=test"},
			want: func() *logs.Options {
				c := newOptions.Config.DeepCopy()
				c.Format = "test"
				return &logs.Options{*c}
			}(),
			errs: field.ErrorList{&field.Error{
				Type:     "FieldValueInvalid",
				Field:    "format",
				BadValue: "test",
				Detail:   "Unsupported log format",
			}},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			o := logs.NewOptions()
			fs := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
			o.AddFlags(fs)
			fs.Parse(tc.args)
			if !assert.Equal(t, tc.want, o) {
				t.Errorf("Wrong Validate() result for %q. expect %v, got %v", tc.name, tc.want, o)
			}
			featureGate := featuregate.NewFeatureGate()
			logs.AddFeatureGates(featureGate)
			err := featureGate.SetFromMap(map[string]bool{string(logs.ContextualLogging): tc.contextualLogging})
			require.NoError(t, err)
			errs := o.ValidateAndApply(featureGate)
			defer klog.ClearLogger()
			if !assert.ElementsMatch(t, tc.errs, errs) {
				t.Errorf("Wrong Validate() result for %q.\n expect:\t%+v\n got:\t%+v", tc.name, tc.errs, errs)

			}
			currentLogger := klog.Background()
			isKlogr := currentLogger == klogr
			if tc.contextualLogging && isKlogr {
				t.Errorf("Expected to get zapr as logger, got: %T", currentLogger)
			}
			if !tc.contextualLogging && !isKlogr {
				t.Errorf("Expected to get klogr as logger, got: %T", currentLogger)
			}
		})
	}
}
