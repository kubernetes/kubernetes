/*
Copyright The Kubernetes Authors.

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

package runtime

import (
	"bytes"
	"context"
	"net/http"
	"strings"
	"testing"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/textlogger"
)

// crashHandlers defers each HandleCrash variant with the given logger and
// then calls trigger. The handler passed to HandleCrashWithContext records
// the panic value in *handled.
var crashHandlers = map[string]func(logger klog.Logger, handled *interface{}, trigger func()){
	"HandleCrash": func(logger klog.Logger, _ *interface{}, trigger func()) {
		klog.SetLogger(logger)
		defer klog.ClearLogger()
		defer HandleCrash()
		trigger()
	},
	"HandleCrashWithContext": func(logger klog.Logger, handled *interface{}, trigger func()) {
		defer HandleCrashWithContext(klog.NewContext(context.Background(), logger), func(_ context.Context, r interface{}) {
			*handled = r
		})
		trigger()
	},
	"HandleCrashWithLogger": func(logger klog.Logger, _ *interface{}, trigger func()) {
		defer HandleCrashWithLogger(logger)
		trigger()
	},
}

func TestHandleCrash(t *testing.T) {
	for name, handleCrash := range crashHandlers {
		for _, tc := range []struct {
			name      string
			panicWith interface{}
			wantLog   string
		}{
			{name: "string", panicWith: "fake", wantLog: `"Observed a panic" panic="fake" stacktrace=`},
			{name: "int", panicWith: 42, wantLog: `"Observed a panic" panic="42" stacktrace=`},
			{name: "ErrAbortHandler", panicWith: http.ErrAbortHandler},
			{name: "none"},
		} {
			t.Run(name+"/"+tc.name, func(t *testing.T) {
				var buffer bytes.Buffer
				logger := textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(&buffer)))
				var recovered, handled interface{}
				func() {
					defer func() { recovered = recover() }()
					handleCrash(logger, &handled, func() {
						if tc.panicWith != nil {
							panic(tc.panicWith)
						}
					})
				}()
				if recovered != tc.panicWith {
					t.Errorf("expected panic %v to continue, got %v", tc.panicWith, recovered)
				}
				if name == "HandleCrashWithContext" && handled != tc.panicWith {
					t.Errorf("expected the additional handler to see %v, got %v", tc.panicWith, handled)
				}
				log := buffer.String()
				if tc.wantLog == "" && log != "" {
					t.Errorf("expected no log entry, got:\n%s", log)
				}
				// The entry must be attributed to the function which deferred the variant.
				if tc.wantLog != "" && (!strings.Contains(log, tc.wantLog) || !strings.Contains(log, "runtime_test.go:")) {
					t.Errorf("expected %q attributed to the test, got:\n%s", tc.wantLog, log)
				}
			})
		}
	}
}
