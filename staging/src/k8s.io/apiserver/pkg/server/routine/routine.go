/*
Copyright 2023 The Kubernetes Authors.

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

package routine

import (
	"context"
	"net/http"

	"k8s.io/apiserver/pkg/endpoints/request"
)

type taskKeyType int

const taskKey taskKeyType = iota

type Task struct {
	Func func()
}

func WithTask(parent context.Context, t *Task) context.Context {
	return request.WithValue(parent, taskKey, t)
}

// AppendTask appends a task executed after completion of existing task.
// It is a no-op if there is no existing task.
func AppendTask(ctx context.Context, t *Task) bool {
	if existTask := TaskFrom(ctx); existTask != nil && existTask.Func != nil {
		existFunc := existTask.Func
		existTask.Func = func() {
			existFunc()
			t.Func()
		}
		return true
	}
	return false
}

func TaskFrom(ctx context.Context) *Task {
	t, _ := ctx.Value(taskKey).(*Task)
	return t
}

// WithRoutine returns an http.Handler that executes preparation of long running requests (i.e. watches)
// in a separate Goroutine and then serves the long running request in the main Goroutine. Doing so allows
// freeing stack memory used in preparation Goroutine for better memory efficiency.
func WithRoutine(handler http.Handler, longRunning request.LongRunningRequestCheck) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		requestInfo, _ := request.RequestInfoFrom(ctx)
		if !longRunning(req, requestInfo) {
			handler.ServeHTTP(w, req)
			return
		}

		req = req.WithContext(WithTask(ctx, &Task{}))
		panicCh := make(chan any, 1)
		go func() {
			defer func() {
				if r := recover(); r != nil {
					panicCh <- r
				}
				close(panicCh)
			}()
			handler.ServeHTTP(w, req)
		}()

		if p, ok := <-panicCh; ok {
			panic(p)
		}

		ctx = req.Context()
		if t := TaskFrom(ctx); t != nil && t.Func != nil {
			t.Func()
		}

	})
}
