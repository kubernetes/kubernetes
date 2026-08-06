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

package config

// ImperativeEvictionResponderControllerConfiguration contains elements configuring
// the imperative eviction responder controller.
type ImperativeEvictionResponderControllerConfiguration struct {
	// ConcurrentImperativeEvictionResponderSyncs is the number of eviction object syncing and imperative
	// eviction operations that will be done concurrently. Larger number = bigger throughput of imperative eviction
	// calls (pods/<name>/eviction subresource) and faster Eviction status updating, but more CPU (and network) load.
	ConcurrentImperativeEvictionResponderSyncs int32
}
