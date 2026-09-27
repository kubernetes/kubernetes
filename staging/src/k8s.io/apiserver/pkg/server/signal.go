/*
Copyright 2017 The Kubernetes Authors.

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

import (
	"context"

	"k8s.io/apiserver/pkg/server/signals"
)

// SetupSignalHandler is signals.SetupSignalHandler. The wrappers in this file
// keep the published API and the single shared shutdown state for existing
// callers.
func SetupSignalHandler() <-chan struct{} {
	return signals.SetupSignalHandler()
}

// SetupSignalContext is signals.SetupSignalContext.
func SetupSignalContext() context.Context {
	return signals.SetupSignalContext()
}

// RequestShutdown is signals.RequestShutdown.
func RequestShutdown() bool {
	return signals.RequestShutdown()
}
