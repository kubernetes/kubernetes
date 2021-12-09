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

package v1

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// The NodeExpansion interface allows manually adding extra methods to the NodeInterface.
type NodeExpansion interface {
	// PatchStatus modifies the status of an existing node. It returns the copy
	// of the node that the server returns, or an error.
	PatchStatus(ctx context.Context, nodeName string, data []byte) (*v1.Node, error)
}

// PatchStatus modifies the status of an existing node. It returns the copy of
// the node that the server returns, or an error.
func (c *nodes) PatchStatus(ctx context.Context, nodeName string, data []byte) (*v1.Node, error) {
	result := &v1.Node{}
	err := c.client.Patch(types.StrategicMergePatchType).
		Resource("nodes").
		Name(nodeName).
		SubResource("status").
		Body(data).
		Do(ctx).
		Into(result)
	return result, err
}
