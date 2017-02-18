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

package aws

import "time"

type CachePolicy struct {
	Name     string
	Validity time.Duration
}

func (c *CachePolicy) IsValid(timestamp int64) bool {
	if c.Validity == 0 {
		return false
	}
	now := nanoTime()
	return now < (timestamp + c.Validity.Nanoseconds())

}

var NeverRefresh = &CachePolicy{Name: "NeverRefresh", Validity: time.Hour * 24 * 1000}

var NeverCache = &CachePolicy{Name: "NeverCache", Validity: 0}

// Returns the number of nanoseconds elapsed since the unix epoch
// When go gets a monotonic time source, we can switch to that
func nanoTime() int64 {
	return time.Now().UnixNano()
}
