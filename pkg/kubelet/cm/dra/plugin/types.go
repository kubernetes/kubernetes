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

package plugin

import (
	"context"

	drahealthv1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
)

// StreamHandler defines the interface for handling DRA health streams.
// This interface is implemented by the DRA Manager to decouple the plugin
// package from the manager package, breaking the import cycle.
type StreamHandler interface {
	// HandleWatchResourcesStream processes health updates from a specific DRA plugin stream.
	HandleWatchResourcesStream(ctx context.Context, stream drahealthv1alpha1.DRAResourceHealth_NodeWatchResourcesClient, resourceName string) error
}
