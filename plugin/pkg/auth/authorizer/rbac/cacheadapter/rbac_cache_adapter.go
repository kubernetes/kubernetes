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

package cacheadapter

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/client/cache"
	rbacinformerv1alpha1 "k8s.io/kubernetes/pkg/client/informers/informers_generated/rbac/v1alpha1"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
)

type RBACInternalCache struct {
	clusterRoleIndexHandler        clusterRoleIndexHandler
	clusterRoleBindingIndexHandler clusterRoleBindingIndexHandler
	roleIndexHandler               roleIndexHandler
	roleBindingIndexHandler        roleBindingIndexHandler
}

func NewFromV1alpha1(rbacInformer rbacinformerv1alpha1.Interface) RBACInternalCache {
	ret := RBACInternalCache{
		clusterRoleIndexHandler: clusterRoleIndexHandler{
			indexer: cache.NewIndexer(cache.DeletionHandlingMetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}),
		},
		clusterRoleBindingIndexHandler: clusterRoleBindingIndexHandler{
			indexer: cache.NewIndexer(cache.DeletionHandlingMetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}),
		},
		roleIndexHandler: roleIndexHandler{
			indexer: cache.NewIndexer(cache.DeletionHandlingMetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}),
		},
		roleBindingIndexHandler: roleBindingIndexHandler{
			indexer: cache.NewIndexer(cache.DeletionHandlingMetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}),
		},
	}

	rbacInformer.ClusterRoles().Informer().AddEventHandler(ret.clusterRoleIndexHandler)
	rbacInformer.ClusterRoleBindings().Informer().AddEventHandler(ret.clusterRoleBindingIndexHandler)
	rbacInformer.Roles().Informer().AddEventHandler(ret.roleIndexHandler)
	rbacInformer.RoleBindings().Informer().AddEventHandler(ret.roleBindingIndexHandler)

	return ret
}

func (o RBACInternalCache) ClusterRoleLister() cache.ClusterRoleLister {
	return o.clusterRoleIndexHandler.Lister()
}
func (o RBACInternalCache) ClusterRoleBindingLister() cache.ClusterRoleBindingLister {
	return o.clusterRoleBindingIndexHandler.Lister()
}
func (o RBACInternalCache) RoleLister() cache.RoleLister {
	return o.roleIndexHandler.Lister()
}
func (o RBACInternalCache) RoleBindingLister() cache.RoleBindingLister {
	return o.roleBindingIndexHandler.Lister()
}

type clusterRoleIndexHandler struct {
	indexer cache.Indexer
}

func (o clusterRoleIndexHandler) Lister() cache.ClusterRoleLister {
	return cache.NewClusterRoleLister(o.indexer)
}

func (o clusterRoleIndexHandler) OnAdd(obj interface{}) {
	out := &rbac.ClusterRole{}
	if err := api.Scheme.Convert(obj, out, nil); err != nil {
		utilruntime.HandleError(err)
		return
	}
	if err := o.indexer.Add(out); err != nil {
		utilruntime.HandleError(err)
	}
}

func (o clusterRoleIndexHandler) OnUpdate(_, obj interface{}) {
	out := &rbac.ClusterRole{}
	if err := api.Scheme.Convert(obj, out, nil); err != nil {
		utilruntime.HandleError(err)
		return
	}
	if err := o.indexer.Update(out); err != nil {
		utilruntime.HandleError(err)
	}
}

func (o clusterRoleIndexHandler) OnDelete(obj interface{}) {
	out := &rbac.ClusterRole{}
	if err := api.Scheme.Convert(obj, out, nil); err != nil {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
			return
		}
		if err := api.Scheme.Convert(tombstone.Obj, out, nil); err != nil {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not castable %#v", obj))
			return
		}
	}

	if err := o.indexer.Delete(out); err != nil {
		utilruntime.HandleError(err)
	}
}

type clusterRoleBindingIndexHandler struct {
	indexer cache.Indexer
}

func (o clusterRoleBindingIndexHandler) Lister() cache.ClusterRoleBindingLister {
	return cache.NewClusterRoleBindingLister(o.indexer)
}

func (o clusterRoleBindingIndexHandler) OnAdd(obj interface{}) {
	out := &rbac.ClusterRoleBinding{}
	if err := api.Scheme.Convert(obj, out, nil); err != nil {
		utilruntime.HandleError(err)
		return
	}
	if err := o.indexer.Add(out); err != nil {
		utilruntime.HandleError(err)
	}
}

func (o clusterRoleBindingIndexHandler) OnUpdate(_, obj interface{}) {
	out := &rbac.ClusterRoleBinding{}
	if err := api.Scheme.Convert(obj, out, nil); err != nil {
		utilruntime.HandleError(err)
		return
	}
	if err := o.indexer.Update(out); err != nil {
		utilruntime.HandleError(err)
	}
}

func (o clusterRoleBindingIndexHandler) OnDelete(obj interface{}) {
	out := &rbac.ClusterRoleBinding{}
	if err := api.Scheme.Convert(obj, out, nil); err != nil {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
			return
		}
		if err := api.Scheme.Convert(tombstone.Obj, out, nil); err != nil {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not castable %#v", obj))
			return
		}
	}

	if err := o.indexer.Delete(out); err != nil {
		utilruntime.HandleError(err)
	}
}

type roleIndexHandler struct {
	indexer cache.Indexer
}

func (o roleIndexHandler) Lister() cache.RoleLister {
	return cache.NewRoleLister(o.indexer)
}

func (o roleIndexHandler) OnAdd(obj interface{}) {
	out := &rbac.Role{}
	if err := api.Scheme.Convert(obj, out, nil); err != nil {
		utilruntime.HandleError(err)
		return
	}
	if err := o.indexer.Add(out); err != nil {
		utilruntime.HandleError(err)
	}
}

func (o roleIndexHandler) OnUpdate(_, obj interface{}) {
	out := &rbac.Role{}
	if err := api.Scheme.Convert(obj, out, nil); err != nil {
		utilruntime.HandleError(err)
		return
	}
	if err := o.indexer.Update(out); err != nil {
		utilruntime.HandleError(err)
	}
}

func (o roleIndexHandler) OnDelete(obj interface{}) {
	out := &rbac.Role{}
	if err := api.Scheme.Convert(obj, out, nil); err != nil {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
			return
		}
		if err := api.Scheme.Convert(tombstone.Obj, out, nil); err != nil {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not castable %#v", obj))
			return
		}
	}

	if err := o.indexer.Delete(out); err != nil {
		utilruntime.HandleError(err)
	}
}

type roleBindingIndexHandler struct {
	indexer cache.Indexer
}

func (o roleBindingIndexHandler) Lister() cache.RoleBindingLister {
	return cache.NewRoleBindingLister(o.indexer)
}

func (o roleBindingIndexHandler) OnAdd(obj interface{}) {
	out := &rbac.RoleBinding{}
	if err := api.Scheme.Convert(obj, out, nil); err != nil {
		utilruntime.HandleError(err)
		return
	}
	if err := o.indexer.Add(out); err != nil {
		utilruntime.HandleError(err)
	}
}

func (o roleBindingIndexHandler) OnUpdate(_, obj interface{}) {
	out := &rbac.RoleBinding{}
	if err := api.Scheme.Convert(obj, out, nil); err != nil {
		utilruntime.HandleError(err)
		return
	}
	if err := o.indexer.Update(out); err != nil {
		utilruntime.HandleError(err)
	}
}

func (o roleBindingIndexHandler) OnDelete(obj interface{}) {
	out := &rbac.RoleBinding{}
	if err := api.Scheme.Convert(obj, out, nil); err != nil {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
			return
		}
		if err := api.Scheme.Convert(tombstone.Obj, out, nil); err != nil {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not castable %#v", obj))
			return
		}
	}

	if err := o.indexer.Delete(out); err != nil {
		utilruntime.HandleError(err)
	}
}
