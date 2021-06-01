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
	"fmt"
	"testing"

	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
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
      --logging-format string               Sets the log format. Permitted formats: "json", "text".
                                            Non-default formats don't honor these flags: --add_dir_header, --alsologtostderr, --log_backtrace_at, --log_dir, --log_file, --log_file_max_size, --logtostderr, --one_output, --skip_headers, --skip_log_headers, --stderrthreshold, --vmodule, --log-flush-frequency.
                                            Non-default choices are currently alpha and subject to change without warning. (default "text")
`
	if !assert.Equal(t, want, output.String()) {
		t.Errorf("Wrong list of flags. expect %q, got %q", want, output.String())
	}
}

func TestOptions(t *testing.T) {
	testcases := []struct {
		name string
		args []string
		want *Options
		errs []error
	}{
		{
			name: "Default log format",
			want: NewOptions(),
		},
		{
			name: "Text log format",
			args: []string{"--logging-format=text"},
			want: NewOptions(),
		},
		{
			name: "JSON log format",
			args: []string{"--logging-format=json"},
			want: &Options{
				LogFormat: jsonLogFormat,
			},
		},
		{
			name: "log sanitization",
			args: []string{"--experimental-logging-sanitization"},
			want: &Options{
				LogFormat:       defaultLogFormat,
				LogSanitization: true,
			},
		},
		{
			name: "Unsupported log format",
			args: []string{"--logging-format=test"},
			want: &Options{
				LogFormat: "test",
			},
			errs: []error{fmt.Errorf("unsupported log format: test")},
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
			errs := o.Validate()
			if !assert.ElementsMatch(t, tc.errs, errs) {
				t.Errorf("Wrong Validate() result for %q. expect %v, got %v", tc.name, tc.errs, errs)
			}
		})
	}
}
