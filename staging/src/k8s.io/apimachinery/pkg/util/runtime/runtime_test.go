/*
Copyright 2014 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/textlogger"
)

func TestHandleCrash(t *testing.T) {
	defer func() {
		if x := recover(); x == nil {
			t.Errorf("Expected a panic to recover from")
		}
	}()
	//nolint:logcheck // Intentionally uses the old API.
	defer HandleCrash()
	panic("Test Panic")
}

func TestCustomHandleCrash(t *testing.T) {
	old := PanicHandlers
	defer func() { PanicHandlers = old }()
	var result interface{}
	PanicHandlers = []func(context.Context, interface{}){
		func(_ context.Context, r interface{}) {
			result = r
		},
	}
	func() {
		defer func() {
			if x := recover(); x == nil {
				t.Errorf("Expected a panic to recover from")
			}
		}()
		//nolint:logcheck // Intentionally uses the old API.
		defer HandleCrash()
		panic("test")
	}()
	if result != "test" {
		t.Errorf("did not receive custom handler")
	}
}

func TestCustomHandleError(t *testing.T) {
	old := ErrorHandlers
	defer func() { ErrorHandlers = old }()
	var result error
	ErrorHandlers = []ErrorHandler{
		func(_ context.Context, err error, msg string, keysAndValues ...interface{}) {
			result = err
		},
	}
	err := fmt.Errorf("test")
	//nolint:logcheck // Intentionally uses the old API.
	HandleError(err)
	if result != err {
		t.Errorf("did not receive custom handler")
	}
}

func TestHandleCrashLog(t *testing.T) {
	log, err := captureStderr(func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("expected a panic to recover from")
			}
		}()
		//nolint:logcheck // Intentionally uses the old API.
		defer HandleCrash()
		panic("test panic")
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	// Example log:
	//
	// ...] Observed a panic: test panic
	// goroutine 6 [running]:
	// command-line-arguments.logPanic(0x..., 0x...)
	// 	.../src/k8s.io/kubernetes/staging/src/k8s.io/apimachinery/pkg/util/runtime/runtime.go:69 +0x...
	lines := strings.Split(log, "\n")
	if len(lines) < 4 {
		t.Fatalf("panic log should have 1 line of message, 1 line per goroutine and 2 lines per function call")
	}
	t.Logf("Got log output:\n%s", strings.Join(lines, "\n"))
	if match, _ := regexp.MatchString(`"Observed a panic" panic="test panic"`, lines[0]); !match {
		t.Errorf("mismatch panic message: %s", lines[0])
	}
	// The following regexp's verify that Kubernetes panic log matches Golang stdlib
	// stacktrace pattern. We need to update these regexp's if stdlib changes its pattern.
	if match, _ := regexp.MatchString(`goroutine [0-9]+ \[.+\]:`, lines[1]); !match {
		t.Errorf("mismatch goroutine: %s", lines[1])
	}
	if match, _ := regexp.MatchString(`logPanic(.*)`, lines[2]); !match {
		t.Errorf("mismatch symbolized function name: %s", lines[2])
	}
	if match, _ := regexp.MatchString(`runtime\.go:[0-9]+ \+0x`, lines[3]); !match {
		t.Errorf("mismatch file/line/offset information: %s", lines[3])
	}
}

func TestHandleCrashContextual(t *testing.T) {
	for name, handleCrash := range map[string]func(logger klog.Logger, trigger func(), additionalHandlers ...func(context.Context, interface{})){
		"WithLogger": func(logger klog.Logger, trigger func(), additionalHandlers ...func(context.Context, interface{})) {
			logger = logger.WithCallDepth(2) // This function *and* the trigger helper.
			defer HandleCrashWithLogger(logger, additionalHandlers...)
			trigger()
		},
		"WithContext": func(logger klog.Logger, trigger func(), additionalHandlers ...func(context.Context, interface{})) {
			logger = logger.WithCallDepth(2)
			defer HandleCrashWithContext(klog.NewContext(context.Background(), logger), additionalHandlers...)
			trigger()
		},
	} {
		t.Run(name, func(t *testing.T) {
			for name, tt := range map[string]struct {
				trigger     func()
				expectPanic string
			}{
				"no-panic": {
					trigger:     func() {},
					expectPanic: "",
				},
				"string-panic": {
					trigger:     func() { panic("fake") },
					expectPanic: "fake",
				},
				"int-panic": {
					trigger:     func() { panic(42) },
					expectPanic: "42",
				},
			} {
				t.Run(name, func(t *testing.T) {
					var buffer bytes.Buffer
					timeInUTC := time.Date(2009, 12, 1, 13, 30, 40, 42000, time.UTC)
					timeString := "1201 13:30:40.000042"
					logger := textlogger.NewLogger(textlogger.NewConfig(
						textlogger.FixedTime(timeInUTC),
						textlogger.Output(&buffer),
					))
					ReallyCrash = false
					defer func() { ReallyCrash = true }()

					handler := func(ctx context.Context, r interface{}) {
						// Same formatting as in HandleCrash.
						str, ok := r.(string)
						if !ok {
							str = fmt.Sprintf("%v", r)
						}
						klog.FromContext(ctx).Info("handler called", "panic", str)
					}

					_, _, line, _ := runtime.Caller(0)
					handleCrash(logger, tt.trigger, handler)
					if tt.expectPanic != "" {
						assert.Contains(t, buffer.String(), fmt.Sprintf(`E%s %7d runtime_test.go:%d] "Observed a panic" panic=%q`, timeString, os.Getpid(), line+1, tt.expectPanic))
						assert.Contains(t, buffer.String(), fmt.Sprintf(`I%s %7d runtime_test.go:%d] "handler called" panic=%q
`, timeString, os.Getpid(), line+1, tt.expectPanic))
					} else {
						assert.Empty(t, buffer.String())
					}
				})
			}
		})
	}
}

func TestHandleCrashLogSilenceHTTPErrAbortHandler(t *testing.T) {
	log, err := captureStderr(func() {
		defer func() {
			if r := recover(); r != http.ErrAbortHandler {
				t.Fatalf("expected to recover from http.ErrAbortHandler")
			}
		}()
		//nolint:logcheck // Intentionally uses the old API.
		defer HandleCrash()
		panic(http.ErrAbortHandler)
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	if len(log) > 0 {
		t.Fatalf("expected no stderr log, got: %s", log)
	}
}

// captureStderr redirects stderr to result string, and then restore stderr from backup
func captureStderr(f func()) (string, error) {
	r, w, err := os.Pipe()
	if err != nil {
		return "", err
	}
	bak := os.Stderr
	os.Stderr = w
	defer func() { os.Stderr = bak }()

	resultCh := make(chan string)
	// copy the output in a separate goroutine so printing can't block indefinitely
	go func() {
		var buf bytes.Buffer
		io.Copy(&buf, r)
		resultCh <- buf.String()
	}()

	f()
	w.Close()

	return <-resultCh, nil
}

func Test_rudimentaryErrorBackoff_OnError_ParallelSleep(t *testing.T) {
	r := &rudimentaryErrorBackoff{
		minPeriod: time.Second,
	}

	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < 30; i++ {
		wg.Add(1)
		go func() {
			<-start
			r.OnError()
			wg.Done()
		}()
	}
	st := time.Now()
	close(start)
	wg.Wait()

	if since := time.Since(st); since > 5*time.Second {
		t.Errorf("OnError slept for too long: %s", since)
	}
}

func TestHandleError(t *testing.T) {
	for name, handleError := range map[string]func(logger klog.Logger, err error, msg string, keysAndValues ...interface{}){
		"WithLogger": func(logger klog.Logger, err error, msg string, keysAndValues ...interface{}) {
			helper, logger := logger.WithCallStackHelper()
			helper()
			HandleErrorWithLogger(logger, err, msg, keysAndValues...)
		},
		"WithContext": func(logger klog.Logger, err error, msg string, keysAndValues ...interface{}) {
			helper, logger := logger.WithCallStackHelper()
			helper()
			HandleErrorWithContext(klog.NewContext(context.Background(), logger), err, msg, keysAndValues...)
		},
	} {
		t.Run(name, func(t *testing.T) {
			for name, tc := range map[string]struct {
				err           error
				msg           string
				keysAndValues []interface{}
				expectLog     string
			}{
				"no-error": {
					msg:       "hello world",
					expectLog: `"hello world" logger="UnhandledError"`,
				},
				"complex": {
					err:           errors.New("fake error"),
					msg:           "ignore",
					keysAndValues: []interface{}{"a", 1, "b", "c"},
					expectLog:     `"ignore" err="fake error" logger="UnhandledError" a=1 b="c"`,
				},
			} {
				t.Run(name, func(t *testing.T) {
					var buffer bytes.Buffer
					timeInUTC := time.Date(2009, 12, 1, 13, 30, 40, 42000, time.UTC)
					timeString := "1201 13:30:40.000042"
					logger := textlogger.NewLogger(textlogger.NewConfig(
						textlogger.FixedTime(timeInUTC),
						textlogger.Output(&buffer),
					))

					_, _, line, _ := runtime.Caller(0)
					handleError(logger, tc.err, tc.msg, tc.keysAndValues...)
					assert.Equal(t, fmt.Sprintf("E%s %7d runtime_test.go:%d] %s\n", timeString, os.Getpid(), line+1, tc.expectLog), buffer.String())
				})
			}
		})
	}
}
