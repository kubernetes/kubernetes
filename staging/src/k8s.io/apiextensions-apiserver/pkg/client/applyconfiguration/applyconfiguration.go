/*
Copyright The Kubernetes Authors.

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

package applyconfiguration

import (
	applyconfiguration "k8s.io/apiextensions/pkg/client/applyconfiguration"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
)

//go:fix inline
func ForKind(kind schema.GroupVersionKind) interface{} {
	return applyconfiguration.ForKind(kind)
}

//go:fix inline
func NewTypeConverter(scheme *runtime.Scheme) managedfields.TypeConverter {
	return applyconfiguration.NewTypeConverter(scheme)
}
