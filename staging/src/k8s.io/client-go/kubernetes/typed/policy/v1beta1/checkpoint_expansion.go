/*
Copyright 2016 The Kubernetes Authors.

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

package v1beta1

import (
	"context"

	policy "k8s.io/api/policy/v1beta1"
)

// The CheckpointExpansion interface allows manually adding extra methods to the ScaleInterface.
type CheckpointExpansion interface {
	Checkpoint(ctx context.Context, checkpoint *policy.Checkpoint) error
}

func (c *checkpoints) Checkpoint(ctx context.Context, checkpoint *policy.Checkpoint) error {
	return c.client.Post().
		AbsPath("/api/v1").
		Namespace(checkpoint.Namespace).
		Resource("pods").
		Name(checkpoint.Name).
		SubResource("checkpoint").
		Body(checkpoint).
		Do(ctx).
		Error()
}
