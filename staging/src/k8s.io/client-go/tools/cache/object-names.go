/*
Copyright 2023 The Kubernetes Authors.

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

package cache

import (
	"k8s.io/apimachinery/pkg/types"
)

// ObjectName is a reference to an object of some implicit kind
type ObjectName struct {
	Namespace string
	Name      string
}

// NewObjectName constructs a new one
func NewObjectName(namespace, name string) ObjectName {
	return ObjectName{Namespace: namespace, Name: name}
}

// Parts is the inverse of the constructor
func (objName ObjectName) Parts() (namespace, name string) {
	return objName.Namespace, objName.Name
}

// String returns the standard string encoding,
// which is designed to match the historical behavior of MetaNamespaceKeyFunc.
// Note this behavior is different from the String method of types.NamespacedName.
func (objName ObjectName) String() string {
	if len(objName.Namespace) > 0 {
		return objName.Namespace + "/" + objName.Name
	}
	return objName.Name
}

// ParseObjectName tries to parse the standard encoding
func ParseObjectName(str string) (ObjectName, error) {
	var objName ObjectName
	var err error
	objName.Namespace, objName.Name, err = SplitMetaNamespaceKey(str)
	return objName, err
}

// NamespacedNameAsObjectName rebrands the given NamespacedName as an ObjectName
func NamespacedNameAsObjectName(nn types.NamespacedName) ObjectName {
	return NewObjectName(nn.Namespace, nn.Name)
}

// AsNamespacedName rebrands as a NamespacedName
func (objName ObjectName) AsNamespacedName() types.NamespacedName {
	return types.NamespacedName{Namespace: objName.Namespace, Name: objName.Name}
}
