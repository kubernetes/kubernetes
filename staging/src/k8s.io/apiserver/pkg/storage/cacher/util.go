/*
Copyright 2015 The Kubernetes Authors.

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

package cacher

import (
	"math"
	"time"
)

// calculateRetryAfterForUnreadyCache calculates the retry duration based on the cache downtime.
func calculateRetryAfterForUnreadyCache(downtime time.Duration) int {
	factor := 0.06
	result := math.Exp(factor * downtime.Seconds())
	result = math.Min(30, math.Max(1, result))
	return int(result)
}
