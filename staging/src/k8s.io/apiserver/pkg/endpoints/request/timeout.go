/*
Copyright 2024 The Kubernetes Authors.

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
	"math/rand"
	"time"
)

// GetTimeoutForWatch calculates the timeout duration for a given WATCH request
//
// timeoutSeconds: timeout in seconds for the watch request, this refers to the
// value in seconds the user has specified in the request parameter 'timeoutSeconds'
// minRequestTimeout: the value obtained from the server run option
// '--min-request-timeout', if specified (in seconds), long running requests
// such as watch will be allocated a random timeout between this value,
// and twice this value.
//
// NOTE: GetTimeoutForWatch is placed here so it can be reused by
// endpoints/filters and endpoints/handlers packages. We want to avoid
// duplication of this logic.
func GetTimeoutForWatch(timeoutSeconds *int64, minRequestTimeout time.Duration) time.Duration {
	// TODO: Currently we explicitly ignore ?timeout= and use only ?timeoutSeconds=.
	timeout := time.Duration(0)

	// NOTE: we allow a negative value, in keeping with the
	// current (1.30) behavior.
	if timeoutSeconds != nil {
		timeout = time.Duration(*timeoutSeconds) * time.Second
	}

	// we don't distinguish between:
	//  a) user specifies a zero value 'timeoutSeconds=0' in the request parameter
	//  b) user does not specify any timeout in the request parameter
	// NOTE(1.30): this is in keeping with current behavior
	if timeout == 0 && minRequestTimeout > 0 {
		timeout = time.Duration(float64(minRequestTimeout) * (rand.Float64() + 1.0))
	}
	// at this point a timeout of zero value implies that:
	//  a) the user has not specified any non-zero timeout value in the request parameter
	//  b) the server option --min-request-timeout has a value of zero
	// and hence, the watch request will timeout immediately.
	// NOTE(1.30): this is in keeping with the current behavior
	return timeout
}
