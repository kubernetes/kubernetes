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

package informers

import (
	"reflect"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

type ClusterRoleInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() cache.ClusterRoleLister
}

type clusterRoleInformer struct {
	*sharedInformerFactory
}

func (f *clusterRoleInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&rbac.ClusterRole{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return f.client.Rbac().ClusterRoles().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return f.client.Rbac().ClusterRoles().Watch(options)
			},
		},
		&rbac.ClusterRole{},
		f.defaultResync,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)
	f.informers[informerType] = informer

	return informer
}

func (f *clusterRoleInformer) Lister() cache.ClusterRoleLister {
	return cache.NewClusterRoleLister(f.Informer().GetIndexer())
}

type ClusterRoleBindingInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() cache.ClusterRoleBindingLister
}

type clusterRoleBindingInformer struct {
	*sharedInformerFactory
}

func (f *clusterRoleBindingInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&rbac.ClusterRoleBinding{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return f.client.Rbac().ClusterRoleBindings().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return f.client.Rbac().ClusterRoleBindings().Watch(options)
			},
		},
		&rbac.ClusterRoleBinding{},
		f.defaultResync,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)
	f.informers[informerType] = informer

	return informer
}

func (f *clusterRoleBindingInformer) Lister() cache.ClusterRoleBindingLister {
	return cache.NewClusterRoleBindingLister(f.Informer().GetIndexer())
}

type RoleInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() cache.RoleLister
}

type roleInformer struct {
	*sharedInformerFactory
}

func (f *roleInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&rbac.Role{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return f.client.Rbac().Roles(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return f.client.Rbac().Roles(api.NamespaceAll).Watch(options)
			},
		},
		&rbac.Role{},
		f.defaultResync,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)
	f.informers[informerType] = informer

	return informer
}

func (f *roleInformer) Lister() cache.RoleLister {
	return cache.NewRoleLister(f.Informer().GetIndexer())
}

type RoleBindingInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() cache.RoleBindingLister
}

type roleBindingInformer struct {
	*sharedInformerFactory
}

func (f *roleBindingInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&rbac.RoleBinding{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return f.client.Rbac().RoleBindings(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return f.client.Rbac().RoleBindings(api.NamespaceAll).Watch(options)
			},
		},
		&rbac.RoleBinding{},
		f.defaultResync,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)
	f.informers[informerType] = informer

	return informer
}

func (f *roleBindingInformer) Lister() cache.RoleBindingLister {
	return cache.NewRoleBindingLister(f.Informer().GetIndexer())
}
