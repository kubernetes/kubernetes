/*
Copyright 2025 The Kubernetes Authors.

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

package framework

// PodGroupCycleState provides a mechanism for plugins to store and retrieve arbitrary data.
// StateData stored by one pod-group-scope plugin can be read, altered, or deleted by another plugin.
// PodGroupCycleState does not provide any data protection, as all plugins are assumed to be
// trusted.
type PodGroupCycleState interface {
	// See CycleState for notes on concurrency.
	Read(key StateKey) (StateData, error)
	// Write stores the given "val" in CycleState with the given "key".
	//
	// See CycleState for notes on concurrency.
	Write(key StateKey, val StateData)
	// Delete deletes data with the given key from CycleState.
	//
	// See CycleState for notes on concurrency.
	Delete(key StateKey)
	// Clone creates a copy of CycleState and returns its pointer. Clone returns
	// nil if the context being cloned is nil.
	Clone() CycleState
}
