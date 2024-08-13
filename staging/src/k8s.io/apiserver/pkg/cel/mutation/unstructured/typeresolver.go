/*
Copyright 2024 The Kubernetes Authors.

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

package unstructured

import (
	"strings"

	"k8s.io/apiserver/pkg/cel/mutation/common"
)

const object = common.RootTypeReferenceName

type TypeResolver struct {
}

// Resolve resolves the TypeRef for the given type name
// that starts with "Object".
// This is the unstructured version, which means the
// returned TypeRef does not refer to the schema.
func (r *TypeResolver) Resolve(name string) (common.TypeRef, bool) {
	if !strings.HasPrefix(name, object) {
		return nil, false
	}
	return NewTypeRef(name), true
}
