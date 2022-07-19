/*
Copyright 2022 The KCP Authors.

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

package customresource

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
)

// Store is an interface used by the upstream CR registry instead of the concrete genericregistry.Store
// in order to allow alternative implementations to be used.
type Store interface {
	rest.StandardStorage
	rest.ResetFieldsStrategy
}

// NewStores is a constructor of the main and status subresource stores for custom resources.
type NewStores func(resource schema.GroupResource, kind, listKind schema.GroupVersionKind, strategy customResourceStrategy, optsGetter generic.RESTOptionsGetter, tableConvertor rest.TableConvertor) (main Store, status Store)

// CustomResourceStrategy makes customResourceStrategy public for downstream consumers.
type CustomResourceStrategy = customResourceStrategy
