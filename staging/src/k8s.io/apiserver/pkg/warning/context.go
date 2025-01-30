/*
Copyright 2020 The Kubernetes Authors.

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

package warning

import (
	"context"
)

// The key type is unexported to prevent collisions
type key int

const (
	// warningRecorderKey is the context key for the warning recorder.
	warningRecorderKey key = iota
)

// Recorder provides a method for recording warnings
type Recorder interface {
	// AddWarning adds the specified warning to the response.
	// agent must be valid UTF-8, and must not contain spaces, quotes, backslashes, or control characters.
	// text must be valid UTF-8, and must not contain control characters.
	AddWarning(agent, text string)
}

// WithWarningRecorder returns a new context that wraps the provided context and contains the provided Recorder implementation.
// The returned context can be passed to AddWarning().
func WithWarningRecorder(ctx context.Context, recorder Recorder) context.Context {
	return context.WithValue(ctx, warningRecorderKey, recorder)
}

func warningRecorderFrom(ctx context.Context) (Recorder, bool) {
	recorder, ok := ctx.Value(warningRecorderKey).(Recorder)
	return recorder, ok
}

// AddWarning records a warning for the specified agent and text to the Recorder added to the provided context using WithWarningRecorder().
// If no Recorder exists in the provided context, this is a no-op.
// agent must be valid UTF-8, and must not contain spaces, quotes, backslashes, or control characters.
// text must be valid UTF-8, and must not contain control characters.
func AddWarning(ctx context.Context, agent string, text string) {
	recorder, ok := warningRecorderFrom(ctx)
	if !ok {
		return
	}
	recorder.AddWarning(agent, text)
}
