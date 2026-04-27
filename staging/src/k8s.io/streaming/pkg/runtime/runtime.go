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

// HandleCrash recovers from panic and logs it.
func HandleCrash() {
	HandleCrashWithLogger(klog.Background())
}

// HandleCrashWithContext recovers from panic and logs it with the context logger.
func HandleCrashWithContext(ctx context.Context, additionalHandlers ...func(context.Context, interface{})) {
	if r := recover(); r != nil {
		for _, fn := range additionalHandlers {
			fn(ctx, r)
		}
		klog.FromContext(ctx).Error(fmt.Errorf("%v", r), "Observed a panic")
	}
}

// HandleCrashWithLogger recovers from panic and logs it using the provided logger.
func HandleCrashWithLogger(logger klog.Logger) {
	if r := recover(); r != nil {
		logger.Error(fmt.Errorf("%v", r), "Observed a panic")
	}
}
