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

package v1

import (
	"context"

	policy "k8s.io/api/policy/v1"
)

// The EvictionExpansion interface allows manually adding extra methods to the ScaleInterface.
type EvictionExpansion interface {
	Evict(ctx context.Context, eviction *policy.Eviction) error
}

func (c *evictions) Evict(ctx context.Context, eviction *policy.Eviction) error {
	return c.GetClient().Post().
		AbsPath("/api/v1").
		Namespace(eviction.Namespace).
		Resource("pods").
		Name(eviction.Name).
		SubResource("eviction").
		Body(eviction).
		Do(ctx).
		Error()
}
