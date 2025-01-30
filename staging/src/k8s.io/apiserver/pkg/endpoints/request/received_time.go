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

package request

import (
	"context"
	"time"
)

type requestReceivedTimestampKeyType int

// requestReceivedTimestampKey is the ReceivedTimestamp (the time the request reached the apiserver)
// key for the context.
const requestReceivedTimestampKey requestReceivedTimestampKeyType = iota

// WithReceivedTimestamp returns a copy of parent context in which the ReceivedTimestamp
// (the time the request reached the apiserver) is set.
//
// If the specified ReceivedTimestamp is zero, no value is set and the parent context is returned as is.
func WithReceivedTimestamp(parent context.Context, receivedTimestamp time.Time) context.Context {
	if receivedTimestamp.IsZero() {
		return parent
	}
	return WithValue(parent, requestReceivedTimestampKey, receivedTimestamp)
}

// ReceivedTimestampFrom returns the value of the ReceivedTimestamp key from the specified context.
func ReceivedTimestampFrom(ctx context.Context) (time.Time, bool) {
	info, ok := ctx.Value(requestReceivedTimestampKey).(time.Time)
	return info, ok
}
