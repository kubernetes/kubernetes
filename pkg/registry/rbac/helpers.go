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

package rbac

import (
	"reflect"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
)

// IsOnlyMutatingGCFields checks finalizers and ownerrefs which GC manipulates
// and indicates that only those fields are changing
func IsOnlyMutatingGCFields(obj, old runtime.Object, equalities conversion.Equalities) bool {
	if old == nil || reflect.ValueOf(old).IsNil() {
		return false
	}

	// make a copy of the newObj so that we can stomp for comparison
	copied := obj.DeepCopyObject()
	copiedMeta, err := meta.Accessor(copied)
	if err != nil {
		return false
	}
	oldMeta, err := meta.Accessor(old)
	if err != nil {
		return false
	}
	copiedMeta.SetOwnerReferences(oldMeta.GetOwnerReferences())
	copiedMeta.SetFinalizers(oldMeta.GetFinalizers())
	copiedMeta.SetSelfLink(oldMeta.GetSelfLink())

	return equalities.DeepEqual(copied, old)
}
