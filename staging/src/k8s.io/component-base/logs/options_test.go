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

package logs

import (
	"bytes"
	"testing"

	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestFlags(t *testing.T) {
	o := NewOptions()
	fs := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	output := bytes.Buffer{}
	o.AddFlags(fs)
	fs.SetOutput(&output)
	fs.PrintDefaults()
	want := `      --experimental-logging-sanitization   [Experimental] When enabled prevents logging of fields tagged as sensitive (passwords, keys, tokens).
                                            Runtime log sanitization may introduce significant computation overhead and therefore should not be enabled in production.
      --log-flush-frequency duration        Maximum number of seconds between log flushes (default 5s)
      --logging-format string               Sets the log format. Permitted formats: "text".
                                            Non-default formats don't honor these flags: --add-dir-header, --alsologtostderr, --log-backtrace-at, --log-dir, --log-file, --log-file-max-size, --logtostderr, --one-output, --skip-headers, --skip-log-headers, --stderrthreshold, --vmodule.
                                            Non-default choices are currently alpha and subject to change without warning. (default "text")
  -v, --v Level                             number for the log level verbosity
      --vmodule pattern=N,...               comma-separated list of pattern=N settings for file-filtered logging (only works for text log format)
`
	if !assert.Equal(t, want, output.String()) {
		t.Errorf("Wrong list of flags. expect %q, got %q", want, output.String())
	}
}

func TestOptions(t *testing.T) {
	newOptions := NewOptions()
	testcases := []struct {
		name string
		args []string
		want *Options
		errs field.ErrorList
	}{
		{
			name: "Default log format",
			want: newOptions,
		},
		{
			name: "Text log format",
			args: []string{"--logging-format=text"},
			want: newOptions,
		},
		{
			name: "log sanitization",
			args: []string{"--experimental-logging-sanitization"},
			want: func() *Options {
				c := newOptions.Config.DeepCopy()
				c.Sanitization = true
				return &Options{*c}
			}(),
		},
		{
			name: "Unsupported log format",
			args: []string{"--logging-format=test"},
			want: func() *Options {
				c := newOptions.Config.DeepCopy()
				c.Format = "test"
				return &Options{*c}
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
			o := NewOptions()
			fs := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
			o.AddFlags(fs)
			fs.Parse(tc.args)
			if !assert.Equal(t, tc.want, o) {
				t.Errorf("Wrong Validate() result for %q. expect %v, got %v", tc.name, tc.want, o)
			}
			err := o.ValidateAndApply()

			if !assert.ElementsMatch(t, tc.errs.ToAggregate(), err) {
				t.Errorf("Wrong Validate() result for %q.\n expect:\t%+v\n got:\t%+v", tc.name, tc.errs, err)

			}
		})
	}
}
