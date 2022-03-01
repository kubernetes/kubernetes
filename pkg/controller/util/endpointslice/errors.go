/*
Copyright 2021 The Kubernetes Authors.

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

package endpointslice

// StaleInformerCache errors indicate that the informer cache includes out of
// date resources.
type StaleInformerCache struct {
	msg string
}

// NewStaleInformerCache return StaleInformerCache with error mes
func NewStaleInformerCache(msg string) *StaleInformerCache {
	return &StaleInformerCache{msg}
}

func (e *StaleInformerCache) Error() string { return e.msg }

func IsStaleInformerCacheErr(err error) bool {
	_, ok := err.(*StaleInformerCache)
	return ok
}
