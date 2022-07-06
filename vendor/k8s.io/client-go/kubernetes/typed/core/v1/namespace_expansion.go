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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	scheme "k8s.io/client-go/kubernetes/scheme"
)

// The NamespaceExpansion interface allows manually adding extra methods to the NamespaceInterface.
type NamespaceExpansion interface {
	Finalize(ctx context.Context, item *v1.Namespace, opts metav1.UpdateOptions) (*v1.Namespace, error)
}

// Finalize takes the representation of a namespace to update.  Returns the server's representation of the namespace, and an error, if it occurs.
func (c *namespaces) Finalize(ctx context.Context, namespace *v1.Namespace, opts metav1.UpdateOptions) (result *v1.Namespace, err error) {
	result = &v1.Namespace{}
	err = c.client.Put().Resource("namespaces").Name(namespace.Name).VersionedParams(&opts, scheme.ParameterCodec).SubResource("finalize").Body(namespace).Do(ctx).Into(result)
	return
}
