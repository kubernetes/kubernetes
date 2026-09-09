/*
Copyright 2022 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/cache"
)

var _ Informer[runtime.Object] = informer[runtime.Object]{}

type informer[T runtime.Object] struct {
	cache.SharedIndexInformer
	lister[T]
}

// Creates a generic informer around a type-erased cache.SharedIndexInformer
// It is incumbent on the caller to ensure that the generic type argument is
// consistent with the type of the objects stored inside the SharedIndexInformer
// as they will be casted.
func NewInformer[T runtime.Object](informe cache.SharedIndexInformer) Informer[T] {
	return informer[T]{
		SharedIndexInformer: informe,
		lister:              NewLister[T](informe.GetIndexer()),
	}
}
