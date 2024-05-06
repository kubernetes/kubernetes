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

package v1

import (
	"bytes"
	"context"
	"flag"
	"testing"

	"github.com/go-logr/logr"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
)

func TestReapply(t *testing.T) {
	oldReapplyHandling := ReapplyHandling
	defer func() {
		ReapplyHandling = oldReapplyHandling
		if err := ResetForTest(nil /* feature gates */); err != nil {
			t.Errorf("Unexpected error resetting the logging configuration: %v", err)
		}
	}()

	newOptions := NewLoggingConfiguration()
	if err := ValidateAndApply(newOptions, nil); err != nil {
		t.Errorf("unexpected error for first ValidateAndApply: %v", err)
	}
	ReapplyHandling = ReapplyHandlingError
	if err := ValidateAndApply(newOptions, nil); err == nil {
		t.Error("did not get expected error for second ValidateAndApply")
	}
	ReapplyHandling = ReapplyHandlingIgnoreUnchanged
	if err := ValidateAndApply(newOptions, nil); err != nil {
		t.Errorf("unexpected error for third ValidateAndApply: %v", err)
	}
	modifiedOptions := newOptions.DeepCopy()
	modifiedOptions.Verbosity = 100
	if err := ValidateAndApply(modifiedOptions, nil); err == nil {
		t.Errorf("unexpected success for forth ValidateAndApply, should have complained about modified config")
	}
}

func TestOptions(t *testing.T) {
	newOptions := NewLoggingConfiguration()
	testcases := []struct {
		name string
		args []string
		want *LoggingConfiguration
		errs field.ErrorList
	}{
		{
			name: "Default log format",
			want: newOptions.DeepCopy(),
		},
		{
			name: "Text log format",
			args: []string{"--logging-format=text"},
			want: newOptions.DeepCopy(),
		},
		{
			name: "Unsupported log format",
			args: []string{"--logging-format=test"},
			want: func() *LoggingConfiguration {
				c := newOptions.DeepCopy()
				c.Format = "test"
				return c
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
			c := NewLoggingConfiguration()
			fs := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
			AddFlags(c, fs)
			fs.Parse(tc.args)
			if !assert.Equal(t, tc.want, c) {
				t.Errorf("Wrong Validate() result for %q. expect %v, got %v", tc.name, tc.want, c)
			}
			defer func() {
				if err := ResetForTest(nil /* feature gates */); err != nil {
					t.Errorf("Unexpected error resetting the logging configuration: %v", err)
				}
			}()
			errs := ValidateAndApply(c, nil /* We don't care about feature gates here. */)
			defer klog.StopFlushDaemon()
			if !assert.ElementsMatch(t, tc.errs, errs) {
				t.Errorf("Wrong Validate() result for %q.\n expect:\t%+v\n got:\t%+v", tc.name, tc.errs, errs)
			}
		})
	}
}

func TestFlagSet(t *testing.T) {
	t.Run("pflag", func(t *testing.T) {
		newOptions := NewLoggingConfiguration()
		var fs pflag.FlagSet
		AddFlags(newOptions, &fs)
		var buffer bytes.Buffer
		fs.SetOutput(&buffer)
		fs.PrintDefaults()
		// Expected (Go 1.19, pflag v1.0.5):
		//     --logging-format string          Sets the log format. Permitted formats: "text". (default "text")
		//     --log-flush-frequency duration   Maximum number of seconds between log flushes (default 5s)
		// -v, --v Level                        number for the log level verbosity
		//     --vmodule pattern=N,...          comma-separated list of pattern=N settings for file-filtered logging (only works for text log format)
		//     --log-text-split-stream                [Alpha] In text format, write error messages to stderr and info messages to stdout. The default is to write a single stream to stdout. Enable the LoggingAlphaOptions feature gate to use this.
		//     --log-text-info-buffer-size quantity   [Alpha] In text format with split output streams, the info messages can be buffered for a while to increase performance. The default value of zero bytes disables buffering. The size can be specified as number of bytes (512), multiples of 1000 (1K), multiples of 1024 (2Ki), or powers of those (3M, 4G, 5Mi, 6Gi). Enable the LoggingAlphaOptions feature gate to use this.
		assert.Regexp(t, `^.*--logging-format.*default.*text.*
.*--log-flush-frequency.*default 5s.*
.*-v.*--v.*
.*--vmodule.*pattern=N.*
.*--log-text-split-stream.*
.*--log-text-info-buffer-size quantity.*
$`, buffer.String())
	})

	t.Run("flag", func(t *testing.T) {
		newOptions := NewLoggingConfiguration()
		var pfs pflag.FlagSet
		AddFlags(newOptions, &pfs)
		var fs flag.FlagSet
		pfs.VisitAll(func(f *pflag.Flag) {
			fs.Var(f.Value, f.Name, f.Usage)
		})
		var buffer bytes.Buffer
		fs.SetOutput(&buffer)
		fs.PrintDefaults()
		// Expected (Go 1.19):
		// -log-flush-frequency value
		//   	Maximum number of seconds between log flushes (default 5s)
		// -log-text-info-buffer-size value
		//      [Alpha] In text format with split output streams, the info messages can be buffered for a while to increase performance. The default value of zero bytes disables buffering. The size can be specified as number of bytes (512), multiples of 1000 (1K), multiples of 1024 (2Ki), or powers of those (3M, 4G, 5Mi, 6Gi). Enable the LoggingAlphaOptions feature gate to use this.
		// -log-text-split-stream
		//      [Alpha] In text format, write error messages to stderr and info messages to stdout. The default is to write a single stream to stdout. Enable the LoggingAlphaOptions feature gate to use this.
		// -logging-format value
		//   	Sets the log format. Permitted formats: "text". (default text)
		// -v value
		//   	number for the log level verbosity
		// -vmodule value
		//   	comma-separated list of pattern=N settings for file-filtered logging (only works for text log format)
		assert.Regexp(t, `^.*-log-flush-frequency.*
.*default 5s.*
.*-log-text-info-buffer-size.*
.*
.*-log-text-split-stream.*
.*
.*-logging-format.*
.*default.*text.*
.*-v.*
.*
.*-vmodule.*
.*
$`, buffer.String())
	})

	t.Run("AddGoFlags", func(t *testing.T) {
		newOptions := NewLoggingConfiguration()
		var fs flag.FlagSet
		var buffer bytes.Buffer
		AddGoFlags(newOptions, &fs)
		fs.SetOutput(&buffer)
		fs.PrintDefaults()
		// In contrast to copying through VisitAll, the type of some options is now
		// known:
		// -log-flush-frequency duration
		//   	Maximum number of seconds between log flushes (default 5s)
		// -log-text-info-buffer-size value
		//      [Alpha] In text format with split output streams, the info messages can be buffered for a while to increase performance. The default value of zero bytes disables buffering. The size can be specified as number of bytes (512), multiples of 1000 (1K), multiples of 1024 (2Ki), or powers of those (3M, 4G, 5Mi, 6Gi). Enable the LoggingAlphaOptions feature gate to use this.
		// -log-text-split-stream
		//      [Alpha] In text format, write error messages to stderr and info messages to stdout. The default is to write a single stream to stdout. Enable the LoggingAlphaOptions feature gate to use this.
		// -logging-format string
		//   	Sets the log format. Permitted formats: "text". (default "text")
		// -v value
		//   	number for the log level verbosity
		// -vmodule value
		//   	comma-separated list of pattern=N settings for file-filtered logging (only works for text log format)
		assert.Regexp(t, `^.*-log-flush-frequency.*duration.*
.*default 5s.*
.*-log-text-info-buffer-size.*
.*
.*-log-text-split-stream.*
.*
.*-logging-format.*string.*
.*default.*text.*
.*-v.*
.*
.*-vmodule.*
.*
$`, buffer.String())
	})
}

func TestContextualLogging(t *testing.T) {
	t.Run("enabled", func(t *testing.T) {
		testContextualLogging(t, true)
	})

	t.Run("disabled", func(t *testing.T) {
		testContextualLogging(t, false)
	})
}

func testContextualLogging(t *testing.T, enabled bool) {
	var err error

	c := NewLoggingConfiguration()
	featureGate := featuregate.NewFeatureGate()
	AddFeatureGates(featureGate)
	err = featureGate.SetFromMap(map[string]bool{string(ContextualLogging): enabled})
	require.NoError(t, err)
	defer func() {
		if err := ResetForTest(nil /* feature gates */); err != nil {
			t.Errorf("Unexpected error resetting the logging configuration: %v", err)
		}
	}()
	err = ValidateAndApply(c, featureGate)
	require.NoError(t, err)
	defer klog.StopFlushDaemon()
	defer klog.EnableContextualLogging(true)

	ctx := context.Background()
	// nolint:logcheck // This intentionally adds a name independently of the feature gate.
	logger := klog.NewKlogr().WithName("contextual")
	// nolint:logcheck // This intentionally creates a new context independently of the feature gate.
	ctx = logr.NewContext(ctx, logger)
	if enabled {
		assert.Equal(t, logger, klog.FromContext(ctx), "FromContext")
		assert.NotEqual(t, ctx, klog.NewContext(ctx, logger), "NewContext")
		assert.NotEqual(t, logger, klog.LoggerWithName(logger, "foo"), "LoggerWithName")
		assert.NotEqual(t, logger, klog.LoggerWithValues(logger, "x", "y"), "LoggerWithValues")
	} else {
		assert.NotEqual(t, logger, klog.FromContext(ctx), "FromContext")
		assert.Equal(t, ctx, klog.NewContext(ctx, logger), "NewContext")
		assert.Equal(t, logger, klog.LoggerWithName(logger, "foo"), "LoggerWithName")
		assert.Equal(t, logger, klog.LoggerWithValues(logger, "x", "y"), "LoggerWithValues")
	}
}
