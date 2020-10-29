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

package server

// newIsTerminatingFunc returns a 'func() bool' that relies on the
// 'ShutdownInitiated' life cycle signal of answer if the apiserver
// has started the termination process.
func (c *Config) newIsTerminatingFunc() func() bool {
	var shutdownCh <-chan struct{}
	// TODO: a properly initialized Config object should always have lifecycleSignals
	//  initialized, but some config unit tests leave lifecycleSignals as nil.
	//  Fix the unit tests upstream and then we can remove this check.
	if c.lifecycleSignals.ShutdownInitiated != nil {
		shutdownCh = c.lifecycleSignals.ShutdownInitiated.Signaled()
	}

	return func() bool {
		select {
		case <-shutdownCh:
			return true
		default:
			return false
		}
	}
}
