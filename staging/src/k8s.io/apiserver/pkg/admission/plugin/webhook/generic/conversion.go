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

package generic

import (
	"fmt"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// convertor converts objects to the desired version.
type convertor struct {
	Scheme *runtime.Scheme
}

// ConvertToGVK converts object to the desired gvk.
func (c *convertor) ConvertToGVK(obj runtime.Object, gvk schema.GroupVersionKind) (runtime.Object, error) {
	// Unlike other resources, custom resources do not have internal version, so
	// if obj is a custom resource, it should not need conversion.
	if obj.GetObjectKind().GroupVersionKind() == gvk {
		return obj, nil
	}
	out, err := c.Scheme.New(gvk)
	// This is a workaround for http://issue.k8s.io/73752 for sending multi-version custom resources to admission webhooks.
	// If we're being asked to convert an unstructured object for a kind that is not registered, it must be a custom resource.
	// In 1.13, only no-op custom resource conversion is possible, so setting the group version is sufficient to "convert".
	// In 1.14+, this was fixed correctly in http://issue.k8s.io/74154 by plumbing the actual object converter here.
	if runtime.IsNotRegisteredError(err) {
		if u, ok := obj.(*unstructured.Unstructured); ok {
			u.GetObjectKind().SetGroupVersionKind(gvk)
			return u, nil
		}
	}
	if err != nil {
		return nil, err
	}
	err = c.Scheme.Convert(obj, out, nil)
	if err != nil {
		return nil, err
	}
	// Explicitly set the GVK
	out.GetObjectKind().SetGroupVersionKind(gvk)
	return out, nil
}

// Validate checks if the conversion has a scheme.
func (c *convertor) Validate() error {
	if c.Scheme == nil {
		return fmt.Errorf("the convertor requires a scheme")
	}
	return nil
}
