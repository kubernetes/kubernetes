/*
Copyright 2018 The Kubernetes Authors.

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

package conversion

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/cr"
)

// NoConversionConverter is a converter that does only set the apiVersion and kind, but without actual conversions.
type NoConversionConverter struct {
	crConverter
}

// NewNoConversionConverter create a NoConversionConverter.
func NewNoConversionConverter(clusterScoped bool) NoConversionConverter {
	return NoConversionConverter{
		crConverter: crConverter{
			clusterScoped: clusterScoped,
		},
	}
}

func (c NoConversionConverter) ConvertToVersion(in runtime.Object, gv runtime.GroupVersioner) (out runtime.Object, err error) {
	// not actually converting anything. The embedded UnstructuredObjectConverter will set the apiVersion and kind
	// of the object. In the future other converters can use this method to do actual conversions.
	if list, ok := in.(*cr.CustomResourceList); ok {
		// TODO: Restructure converters so this list conversion would be done for all converters by default.
		list.EachListItem(func(object runtime.Object) error {
			object, err = c.crConverter.ConvertToVersion(object, gv)
			return err
		})
	}
	return c.crConverter.ConvertToVersion(in, gv)
}
