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

package internalversion

import (
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

// The ScaleExpansion interface allows manually adding extra methods to the ScaleInterface.
type ScaleExpansion interface {
	Get(kind string, name string) (*extensions.Scale, error)
	Update(kind string, scale *extensions.Scale) (*extensions.Scale, error)
}

// Get takes the reference to scale subresource and returns the subresource or error, if one occurs.
func (c *scales) Get(kind string, name string) (result *extensions.Scale, err error) {
	result = &extensions.Scale{}

	// TODO this method needs to take a proper unambiguous kind
	fullyQualifiedKind := schema.GroupVersionKind{Kind: kind}
	resource, _ := meta.UnsafeGuessKindToResource(fullyQualifiedKind)

	err = c.client.Get().
		Namespace(c.ns).
		Resource(resource.Resource).
		Name(name).
		SubResource("scale").
		Do().
		Into(result)
	return
}

func (c *scales) Update(kind string, scale *extensions.Scale) (result *extensions.Scale, err error) {
	result = &extensions.Scale{}

	// TODO this method needs to take a proper unambiguous kind
	fullyQualifiedKind := schema.GroupVersionKind{Kind: kind}
	resource, _ := meta.UnsafeGuessKindToResource(fullyQualifiedKind)

	err = c.client.Put().
		Namespace(scale.Namespace).
		Resource(resource.Resource).
		Name(scale.Name).
		SubResource("scale").
		Body(scale).
		Do().
		Into(result)
	return
}
