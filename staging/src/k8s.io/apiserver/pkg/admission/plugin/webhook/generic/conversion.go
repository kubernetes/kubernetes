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
	if err != nil {
		return nil, err
	}
	err = c.Scheme.Convert(obj, out, nil)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// Validate checks if the conversion has a scheme.
func (c *convertor) Validate() error {
	if c.Scheme == nil {
		return fmt.Errorf("the convertor requires a scheme")
	}
	return nil
}
