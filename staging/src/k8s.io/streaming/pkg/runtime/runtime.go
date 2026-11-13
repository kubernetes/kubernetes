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
	"context"
	"fmt"
	"net/http"
	"runtime"

	"k8s.io/klog/v2"
)

// HandleError logs an asynchronous error.
func HandleError(err error) {
	if err == nil {
		return
	}
	klog.Background().Error(err, "Unhandled Error")
}

// HandleErrorWithContext logs an asynchronous error with contextual logging when available.
func HandleErrorWithContext(ctx context.Context, err error, msg string, keysAndValues ...interface{}) {
	if err == nil {
		return
	}
	klog.FromContext(ctx).Error(err, msg, keysAndValues...)
}

// HandleCrash logs a recovered panic with a stack trace and panics again.
// Meant to be called via defer.
//
// Contextual logging: HandleCrashWithContext or HandleCrashWithLogger should be
// used instead of HandleCrash in code which supports contextual logging.
func HandleCrash() {
	if r := recover(); r != nil {
		handleCrash(context.Background(), r)
	}
}

// HandleCrashWithContext is HandleCrash with the logger taken from the context.
// The additional handlers run after logging, before the panic continues.
func HandleCrashWithContext(ctx context.Context, additionalHandlers ...func(context.Context, interface{})) {
	if r := recover(); r != nil {
		handleCrash(ctx, r, additionalHandlers...)
	}
}

// HandleCrashWithLogger is HandleCrash with the given logger.
func HandleCrashWithLogger(logger klog.Logger) {
	if r := recover(); r != nil {
		handleCrash(klog.NewContext(context.Background(), logger), r)
	}
}

// Each variant calls recover itself, because recover only works in the
// deferred function. The rest is shared so that the call depth is the same.
func handleCrash(ctx context.Context, r interface{}, additionalHandlers ...func(context.Context, interface{})) {
	// Skip handleCrash, the variant and the Go panic handler to reach the
	// function which deferred the variant.
	logger := klog.FromContext(ctx).WithCallDepth(3)
	// net/http panics with ErrAbortHandler to abort a handler quietly.
	if r != http.ErrAbortHandler {
		// Same buffer size as the net/http server.
		stacktrace := make([]byte, 64<<10)
		stacktrace = stacktrace[:runtime.Stack(stacktrace, false)]
		logger.Error(nil, "Observed a panic", "panic", fmt.Sprintf("%v", r), "stacktrace", string(stacktrace))
	}
	ctx = klog.NewContext(ctx, logger)
	for _, fn := range additionalHandlers {
		fn(ctx, r)
	}
	panic(r)
}
