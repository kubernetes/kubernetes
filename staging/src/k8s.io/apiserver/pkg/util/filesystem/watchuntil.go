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

package filesystem

import (
	"context"
	"fmt"
	"os"
	"time"

	"k8s.io/utils/fswatch"
)

// WatchUntil watches the specified path for changes and blocks until ctx is canceled.
// eventHandler() must be non-nil, and pollInterval must be greater than 0.
// eventHandler() is invoked whenever a change event is observed or pollInterval elapses.
// errorHandler() is invoked (if non-nil) whenever an error occurs initializing or watching the specified path.
// If path is a directory, only the directory and immediate children are watched.
// If path does not exist or cannot be watched, an error is passed to errorHandler() and eventHandler() is called at pollInterval.
// eventHandler() is invoked immediately after successful initialization of the filesystem watch.
//
// Implemented as a thin wrapper around k8s.io/utils/fswatch so fsnotify stays out of the apiserver Linux build closure.
func WatchUntil(ctx context.Context, pollInterval time.Duration, path string, eventHandler func(), errorHandler func(err error)) {
	if pollInterval <= 0 {
		panic(fmt.Errorf("pollInterval must be > 0"))
	}
	if eventHandler == nil {
		panic(fmt.Errorf("eventHandler must be non-nil"))
	}
	if errorHandler == nil {
		errorHandler = func(error) {}
	}

	// Dispatch directory vs file at startup; fswatch has separate
	// helpers and the original contract treats them differently.
	info, statErr := os.Stat(path)
	if statErr != nil {
		errorHandler(statErr)
	}
	if statErr == nil && info.IsDir() {
		eventHandler() // match the post-init fire of the original WatchUntil
		_ = fswatch.WatchDir(ctx, path, eventHandler,
			fswatch.WithDirRecheckInterval(pollInterval),
			fswatch.WithDirErrorHandler(errorHandler),
		)
		return
	}
	_ = fswatch.WatchFile(ctx, path, eventHandler,
		fswatch.WithRecheckInterval(pollInterval),
		fswatch.WithFallbackPolling(pollInterval),
		fswatch.WithInitialCallback(),
		fswatch.WithErrorHandler(errorHandler),
	)
}
