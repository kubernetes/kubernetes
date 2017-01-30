/*
Copyright 2016 The Kubernetes Authors.

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

package generators

import "k8s.io/gengo/types"

var (
	apiNamespaceAll             = types.Name{Package: "k8s.io/kubernetes/pkg/api", Name: "NamespaceAll"}
	apiScheme                   = types.Name{Package: "k8s.io/kubernetes/pkg/api", Name: "Scheme"}
	cacheGenericLister          = types.Name{Package: "k8s.io/kubernetes/pkg/client/cache", Name: "GenericLister"}
	cacheIndexers               = types.Name{Package: "k8s.io/kubernetes/pkg/client/cache", Name: "Indexers"}
	cacheListWatch              = types.Name{Package: "k8s.io/kubernetes/pkg/client/cache", Name: "ListWatch"}
	cacheMetaNamespaceIndexFunc = types.Name{Package: "k8s.io/kubernetes/pkg/client/cache", Name: "MetaNamespaceIndexFunc"}
	cacheNamespaceIndex         = types.Name{Package: "k8s.io/kubernetes/pkg/client/cache", Name: "NamespaceIndex"}
	cacheNewGenericLister       = types.Name{Package: "k8s.io/kubernetes/pkg/client/cache", Name: "NewGenericLister"}
	cacheNewSharedIndexInformer = types.Name{Package: "k8s.io/kubernetes/pkg/client/cache", Name: "NewSharedIndexInformer"}
	cacheSharedIndexInformer    = types.Name{Package: "k8s.io/kubernetes/pkg/client/cache", Name: "SharedIndexInformer"}
	listOptions                 = types.Name{Package: "k8s.io/kubernetes/pkg/api", Name: "ListOptions"}
	reflectType                 = types.Name{Package: "reflect", Name: "Type"}
	runtimeObject               = types.Name{Package: "k8s.io/apimachinery/pkg/runtime", Name: "Object"}
	schemaGroupResource         = types.Name{Package: "k8s.io/apimachinery/pkg/runtime/schema", Name: "GroupResource"}
	schemaGroupVersionResource  = types.Name{Package: "k8s.io/apimachinery/pkg/runtime/schema", Name: "GroupVersionResource"}
	syncMutex                   = types.Name{Package: "sync", Name: "Mutex"}
	timeDuration                = types.Name{Package: "time", Name: "Duration"}
	v1ListOptions               = types.Name{Package: "k8s.io/kubernetes/pkg/api/v1", Name: "ListOptions"}
	v1NamespaceAll              = types.Name{Package: "k8s.io/kubernetes/pkg/api/v1", Name: "NamespaceAll"}
	watchInterface              = types.Name{Package: "k8s.io/apimachinery/pkg/watch", Name: "Interface"}
)
