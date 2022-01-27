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
	"context"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/cache"
)

type Controller[T runtime.Object] interface {
	// Meant to be run inside a goroutine
	// Waits for and reacts to changes in whatever type the controller
	// is concerned with.
	//
	// Returns an error always non-nil explaining why the worker stopped
	Run(ctx context.Context) error

	// Retrieves the informer used to back this controller
	Informer() Informer[T]

	// Returns true if the informer cache has synced, and all the objects from
	// the initial list have been reconciled at least once.
	HasSynced() bool
}

type NamespacedLister[T any] interface {
	// List lists all ValidationRuleSets in the indexer for a given namespace.
	// Objects returned here must be treated as read-only.
	List(selector labels.Selector) (ret []T, err error)
	// Get retrieves the ValidationRuleSet from the indexer for a given namespace and name.
	// Objects returned here must be treated as read-only.
	Get(name string) (T, error)
}

type Informer[T any] interface {
	cache.SharedIndexInformer
	Lister[T]
}

// Lister[T] helps list Ts.
// All objects returned here must be treated as read-only.
type Lister[T any] interface {
	NamespacedLister[T]
	Namespaced(namespace string) NamespacedLister[T]
}
