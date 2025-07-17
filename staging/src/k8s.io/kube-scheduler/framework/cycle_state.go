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

import (
	"errors"

	"k8s.io/apimachinery/pkg/util/sets"
)

var (
	// ErrNotFound is the not found error message.
	ErrNotFound = errors.New("not found")
)

// StateData is a generic type for arbitrary data stored in CycleState.
type StateData interface {
	// Clone is an interface to make a copy of StateData. For performance reasons,
	// clone should make shallow copies for members (e.g., slices or maps) that are not
	// impacted by PreFilter's optional AddPod/RemovePod methods.
	Clone() StateData
}

// StateKey is the type of keys stored in CycleState.
type StateKey string

// CycleState provides a mechanism for plugins to store and retrieve arbitrary data.
// StateData stored by one plugin can be read, altered, or deleted by another plugin.
// CycleState does not provide any data protection, as all plugins are assumed to be
// trusted.
type CycleState interface {
	// ShouldRecordPluginMetrics returns whether metrics.PluginExecutionDuration metrics
	// should be recorded.
	// This function is mostly for the scheduling framework runtime, plugins usually don't have to use it.
	ShouldRecordPluginMetrics() bool
	// GetSkipFilterPlugins returns plugins that will be skipped in the Filter extension point.
	// This function is mostly for the scheduling framework runtime, plugins usually don't have to use it.
	GetSkipFilterPlugins() sets.Set[string]
	// SetSkipFilterPlugins sets plugins that should be skipped in the Filter extension point.
	// This function is mostly for the scheduling framework runtime, plugins usually don't have to use it.
	SetSkipFilterPlugins(plugins sets.Set[string])
	// GetSkipScorePlugins returns plugins that will be skipped in the Score extension point.
	// This function is mostly for the scheduling framework runtime, plugins usually don't have to use it.
	GetSkipScorePlugins() sets.Set[string]
	// SetSkipScorePlugins sets plugins that should be skipped in the Score extension point.
	// This function is mostly for the scheduling framework runtime, plugins usually don't have to use it.
	SetSkipScorePlugins(plugins sets.Set[string])
	// GetSkipPreBindPlugins returns plugins that will be skipped in the PreBind extension point.
	// This function is mostly for the scheduling framework runtime, plugins usually don't have to use it.
	GetSkipPreBindPlugins() sets.Set[string]
	// SetSkipPreBindPlugins sets plugins that should be skipped in the PerBind extension point.
	// This function is mostly for the scheduling framework runtime, plugins usually don't have to use it.
	SetSkipPreBindPlugins(plugins sets.Set[string])
	// Read retrieves data with the given "key" from CycleState. If the key is not
	// present, ErrNotFound is returned.
	//
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
