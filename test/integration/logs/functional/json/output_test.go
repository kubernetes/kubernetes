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

package json

import (
	"io"
	"os"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"go.uber.org/zap/zapcore"

	logsapi "k8s.io/component-base/logs/api/v1"
	logsjson "k8s.io/component-base/logs/json"
	"k8s.io/klog/v2/test"
)

func init() {
	test.InitKlog()
}

// TestJsonOutput tests the JSON logger, directly and as backend for klog.
func TestJSONOutput(t *testing.T) {
	newLogger := func(out io.Writer, v int, vmodule string) logr.Logger {
		logger, _ := logsjson.NewJSONLogger(logsapi.VerbosityLevel(v), logsjson.AddNopSync(out), nil,
			&zapcore.EncoderConfig{
				MessageKey:     "msg",
				CallerKey:      "caller",
				NameKey:        "logger",
				EncodeDuration: zapcore.StringDurationEncoder,
				EncodeCaller:   zapcore.ShortCallerEncoder,
			})
		return logger
	}

	// If Go modules are turned off (for example, as in "make test-integration"),
	// references to klog like k8s.io/klog/v2.ObjectRef.MarshalLog become
	// k8s.io/kubernetes/vendor/k8s.io/klog/v2.ObjectRef.MarshalLog.
	injectVendor := func(mapping map[string]string) map[string]string {
		if os.Getenv("GO111MODULE") != "off" {
			return mapping
		}
		for key, value := range mapping {
			mapping[key] = strings.ReplaceAll(value, "k8s.io/klog/v2", "k8s.io/kubernetes/vendor/k8s.io/klog/v2")
		}
		return mapping
	}

	t.Run("direct", func(t *testing.T) {
		test.Output(t, test.OutputConfig{
			NewLogger:             newLogger,
			ExpectedOutputMapping: injectVendor(test.ZaprOutputMappingDirect()),
		})
	})

	t.Run("klog-backend", func(t *testing.T) {
		test.Output(t, test.OutputConfig{
			NewLogger:             newLogger,
			AsBackend:             true,
			ExpectedOutputMapping: injectVendor(test.ZaprOutputMappingIndirect()),
		})
	})
}
