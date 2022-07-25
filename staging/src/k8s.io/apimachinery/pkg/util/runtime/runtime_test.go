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
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
	"testing"
)

func TestHandleCrash(t *testing.T) {
	defer func() {
		if x := recover(); x == nil {
			t.Errorf("Expected a panic to recover from")
		}
	}()
	defer HandleCrash()
	panic("Test Panic")
}

func TestCustomHandleCrash(t *testing.T) {
	old := PanicHandlers
	defer func() { PanicHandlers = old }()
	var result interface{}
	PanicHandlers = []func(interface{}){
		func(r interface{}) {
			result = r
		},
	}
	func() {
		defer func() {
			if x := recover(); x == nil {
				t.Errorf("Expected a panic to recover from")
			}
		}()
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
	ErrorHandlers = []func(error){
		func(err error) {
			result = err
		},
	}
	err := fmt.Errorf("test")
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
	if match, _ := regexp.MatchString("Observed a panic: test panic", lines[0]); !match {
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

func TestHandleCrashLogSilenceHTTPErrAbortHandler(t *testing.T) {
	log, err := captureStderr(func() {
		defer func() {
			if r := recover(); r != http.ErrAbortHandler {
				t.Fatalf("expected to recover from http.ErrAbortHandler")
			}
		}()
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
