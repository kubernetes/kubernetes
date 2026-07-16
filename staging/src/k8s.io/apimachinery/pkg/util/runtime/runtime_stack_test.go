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
	"flag"
	"fmt"
	"regexp"

	"k8s.io/klog/v2"
)

//nolint:logcheck // Several functions are normally not okay in a package.
func ExampleHandleErrorWithContext() {
	state := klog.CaptureState()
	defer state.Restore()
	var fs flag.FlagSet
	klog.InitFlags(&fs)
	for flag, value := range map[string]string{
		"one_output":  "true",
		"logtostderr": "false",
	} {
		if err := fs.Set(flag, value); err != nil {
			fmt.Printf("Unexpected error configuring klog: %v", err)
			return
		}
	}
	var buffer bytes.Buffer
	klog.SetOutput(&buffer)

	logger := klog.Background()
	logger = klog.LoggerWithValues(logger, "request", 42)
	ctx := klog.NewContext(context.Background(), logger)

	// The line number of the next call must be at line 60. Here are some
	// blank lines that can be removed to keep the line unchanged.
	//
	//
	//
	//
	//
	//
	HandleErrorWithContext(ctx, errors.New("fake error"), "test")

	klog.Flush()
	// Strip varying header. Code location should be constant and something
	// that needs to be tested.
	output := buffer.String()
	output = regexp.MustCompile(`^.* ([^[:space:]]*.go:[[:digit:]]*)\] `).ReplaceAllString(output, `xxx $1] `)
	fmt.Print(output)

	// Output:
	// xxx runtime_stack_test.go:60] "test" err="fake error" logger="UnhandledError" request=42
}
